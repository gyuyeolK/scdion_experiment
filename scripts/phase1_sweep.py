"""
Phase 1C: Model size sweep diagnostic.

여러 모델 크기에서 진단해서 SC-Dion 가설이 모델 크기와 어떻게 변하는지 측정.

GPU 가속 (이전 CPU 진단 대비 ~10-50x 빠름):
- 1.7B: ~20s (이전 ~400s)
- 3B: ~40s 예상
- 7B: ~90s 예상

진단할 모델은 환경 변수로 지정. 각각 따로 돌리고 결과 합쳐서 분석.

사용:
    bash launch_phase1_sweep.sh
"""
import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from optimizers.diagnostic_gpu import diagnose_param_gpu


def log(msg):
    print(msg, flush=True)


def load_model_and_tokenizer(model_name: str, dtype: str = 'bfloat16'):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch_dtype = {'bfloat16': torch.bfloat16, 'float16': torch.float16,
                   'float32': torch.float32}[dtype]
    log(f"Loading {model_name} in {dtype}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch_dtype, attn_implementation='sdpa'
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch_dtype, attn_implementation='sdpa'
        )
    try:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={'use_reentrant': False}
        )
    except TypeError:
        model.gradient_checkpointing_enable()
    if hasattr(model.config, 'use_cache'):
        model.config.use_cache = False
    return model, tokenizer


def load_fineweb():
    try:
        from datasets import load_dataset
        ds = load_dataset('HuggingFaceFW/fineweb', name='sample-10BT',
                         split='train', streaming=True)
        return iter(ds)
    except Exception as e:
        log(f"FineWeb load failed: {e}")
        return None


def get_batch(it, tokenizer, batch_size, seq_len, device):
    if it is None:
        vocab = tokenizer.vocab_size
        ids = torch.randint(0, vocab, (batch_size, seq_len), device=device)
        return {'input_ids': ids, 'labels': ids.clone()}, 'random'
    texts = []
    while len(texts) < batch_size:
        try:
            ex = next(it)
        except StopIteration:
            return get_batch(None, tokenizer, batch_size, seq_len, device)
        t = ex.get('text', '')
        if t and len(t) > 50:
            texts.append(t)
    enc = tokenizer(texts, max_length=seq_len, truncation=True,
                    padding='max_length', return_tensors='pt').to(device)
    labels = enc['input_ids'].clone()
    labels[enc['attention_mask'] == 0] = -100
    return {'input_ids': enc['input_ids'],
            'attention_mask': enc['attention_mask'],
            'labels': labels}, 'real'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--seq_len', type=int, default=2048)
    parser.add_argument('--num_batches', type=int, default=4)
    parser.add_argument('--ks', type=int, nargs='+', default=[8, 32])
    parser.add_argument('--alphas', type=float, nargs='+',
                        default=[0.5, 0.25, 0.125])
    parser.add_argument('--cert_threshold', type=float, default=0.05)
    parser.add_argument('--dtype', type=str, default='bfloat16')
    parser.add_argument('--max_params', type=int, default=100)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()
    
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    free, total = torch.cuda.mem_get_info(0)
    log(f"GPU memory: {free/1e9:.1f} / {total/1e9:.1f} GB free")
    
    model, tokenizer = load_model_and_tokenizer(args.model_name, args.dtype)
    
    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    n_2d = sum(1 for _, p in model.named_parameters()
               if p.ndim == 2 and min(p.shape) >= 2)
    log(f"Total params: {total_params/1e9:.2f}B, 2D matrices: {n_2d}")
    
    model = model.to(device)
    model.train()
    
    free, total = torch.cuda.mem_get_info(0)
    log(f"After model load: {free/1e9:.1f} / {total/1e9:.1f} GB free")
    
    # Compute gradients on real data
    ds_iter = load_fineweb()
    log(f"\nComputing gradients on {args.num_batches} FineWeb batches "
        f"(bs={args.batch_size}, seq={args.seq_len})...")
    t0 = time.time()
    actual_data = None
    losses = []
    for b in range(args.num_batches):
        batch, kind = get_batch(ds_iter, tokenizer, args.batch_size,
                                 args.seq_len, device)
        actual_data = kind
        out = model(**batch)
        loss = out.loss / args.num_batches
        loss.backward()
        losses.append(out.loss.item())
        log(f"  batch {b+1}/{args.num_batches}: loss={out.loss.item():.4f}")
        del out, loss, batch
        torch.cuda.empty_cache()
    log(f"Gradient computation: {time.time()-t0:.1f}s")
    log(f"Mean loss: {sum(losses)/len(losses):.3f} ({actual_data} data)")
    
    free, total = torch.cuda.mem_get_info(0)
    log(f"After gradients: {free/1e9:.1f} / {total/1e9:.1f} GB free")
    
    # === Diagnostic (GPU accelerated) ===
    param_names = [n for n, p in model.named_parameters()
                   if p.ndim == 2 and min(p.shape) >= 2 and p.requires_grad]
    
    if len(param_names) > args.max_params:
        idx = [int(i * (len(param_names) - 1) / (args.max_params - 1))
               for i in range(args.max_params)]
        param_names = [param_names[i] for i in sorted(set(idx))]
        log(f"Sampling {len(param_names)} of total params")
    
    log(f"\nRunning GPU diagnostic on {len(param_names)} params...")
    results_params = {}
    name_to_param = dict(model.named_parameters())
    t0 = time.time()
    
    for i, name in enumerate(param_names):
        if (i + 1) % 20 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i+1) * (len(param_names) - i - 1)
            log(f"  [{i+1}/{len(param_names)}] elapsed {elapsed:.1f}s, "
                f"ETA {eta:.1f}s")
        
        p = name_to_param.get(name)
        if p is None or p.grad is None:
            continue
        
        # GPU에서 직접 진단 (CPU 옮기지 않음)
        try:
            results_params[name] = diagnose_param_gpu(
                p.grad, ks=args.ks, alphas=args.alphas,
                c_min=args.cert_threshold
            )
        except Exception as e:
            log(f"  [{name}] failed: {e}")
        
        # grad는 진단 후 해제
        p.grad = None
        
        if (i + 1) % 30 == 0:
            torch.cuda.empty_cache()
    
    elapsed = time.time() - t0
    log(f"Diagnostic time: {elapsed:.1f}s ({elapsed/len(param_names):.2f}s/param)")
    
    # === Summary ===
    log(f"\n{'='*70}")
    log(f"=== SUMMARY: {args.model_name} ({total_params/1e9:.2f}B) ===")
    log(f"{'='*70}")
    
    summary = {
        'model_name': args.model_name,
        'total_params_B': total_params / 1e9,
        'num_2d_matrices': n_2d,
        'mean_loss': sum(losses) / len(losses),
        'actual_data': actual_data,
        'diagnostic_time_sec': elapsed,
        'per_config': {},
    }
    
    for k in args.ks:
        for alpha in args.alphas:
            key = f"k{k}_a{alpha}"
            pc, tot = 0, 0
            cv, tv, ov = [], [], []
            for pd in results_params.values():
                if key in pd.get('certificates', {}):
                    cd = pd['certificates'][key]
                    tot += 1
                    if cd['would_pass']:
                        pc += 1
                    cv.append(cd['cert']); tv.append(cd['tau']); ov.append(cd['omega'])
            if tot > 0:
                summary['per_config'][key] = {
                    'pass_rate': pc/tot, 'mean_cert': sum(cv)/tot,
                    'mean_tau': sum(tv)/tot, 'mean_omega': sum(ov)/tot,
                    'total': tot,
                }
                log(f"  {key:<15} pass {pc:>3}/{tot} ({100*pc/tot:>5.1f}%)  "
                    f"τ={sum(tv)/tot:.3f}  ω={sum(ov)/tot:.3f}  "
                    f"cert={sum(cv)/tot:+.3f}")
    
    spreads = [pd.get('row_norm_spread', 0) for pd in results_params.values()
               if 'row_norm_spread' in pd]
    if spreads:
        sps = sorted(spreads)
        summary['row_norm_spread'] = {
            'median': sps[len(sps)//2], 'max': max(spreads),
            'frac_gt_20': sum(1 for s in spreads if s>20) / len(spreads),
        }
        log(f"\nRow-norm spread: median={sps[len(sps)//2]:.1f}x, "
            f"max={max(spreads):.1f}x, "
            f"frac>20x={sum(1 for s in spreads if s>20)/len(spreads):.2f}")
    
    srs = [pd.get('stable_rank', 0) for pd in results_params.values()
           if pd.get('stable_rank', -1) > 0]
    if srs:
        srss = sorted(srs)
        summary['stable_rank'] = {
            'median': srss[len(srss)//2], 'min': min(srs), 'max': max(srs)
        }
        log(f"Stable rank: median={srss[len(srss)//2]:.1f}, "
            f"min={min(srs):.1f}, max={max(srs):.1f}")
    
    # Save (summary only, not per-param details)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    log(f"\nSaved to {output_path}")


if __name__ == '__main__':
    main()
