"""
Phase 1: Pre-training diagnostic (메모리 효율 버전, 24GB GPU 친화).

핵심 변경 사항:
- grad_accum 별도 dict 제거 - p.grad를 in-place 사용
- 진단 끝난 grad는 즉시 None으로 해제
- gradient checkpointing 기본 활성화 (~50% activation 메모리 절약)
- 짧은 시퀀스 / 작은 배치 기본값
"""
import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

# Make sibling 'optimizers' package importable when running from scripts/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.distributed as dist


def is_main():
    return (not dist.is_initialized()) or dist.get_rank() == 0


def log(msg):
    if is_main():
        print(msg, flush=True)


def setup_distributed():
    if 'RANK' in os.environ:
        dist.init_process_group(backend='nccl')
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        torch.cuda.set_device(local_rank)
        return True, local_rank
    return False, 0


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def load_model_and_tokenizer(model_name: str, dtype: str = 'bfloat16',
                             use_gradient_checkpointing: bool = True):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch_dtype = {'bfloat16': torch.bfloat16, 'float16': torch.float16,
                   'float32': torch.float32}[dtype]
    log(f"Loading {model_name} in {dtype}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # transformers 신/구버전 호환
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch_dtype, attn_implementation='sdpa'
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch_dtype, attn_implementation='sdpa'
        )
    if use_gradient_checkpointing:
        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={'use_reentrant': False}
            )
        except TypeError:
            model.gradient_checkpointing_enable()
        if hasattr(model.config, 'use_cache'):
            model.config.use_cache = False
        log("  gradient checkpointing: ENABLED")
    return model, tokenizer


def get_dummy_batch(tokenizer, batch_size: int, seq_len: int, device: str):
    vocab_size = tokenizer.vocab_size
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    return {'input_ids': input_ids, 'labels': input_ids.clone()}


def get_fineweb_batch(tokenizer, batch_size: int, seq_len: int, device: str,
                     dataset_iter=None):
    if dataset_iter is None:
        return get_dummy_batch(tokenizer, batch_size, seq_len, device)
    texts = []
    while len(texts) < batch_size:
        try:
            ex = next(dataset_iter)
            texts.append(ex['text'])
        except StopIteration:
            break
    if len(texts) < batch_size:
        return get_dummy_batch(tokenizer, batch_size, seq_len, device)
    enc = tokenizer(texts, max_length=seq_len, truncation=True,
                    padding='max_length', return_tensors='pt').to(device)
    labels = enc['input_ids'].clone()
    labels[enc['attention_mask'] == 0] = -100
    return {'input_ids': enc['input_ids'],
            'attention_mask': enc['attention_mask'],
            'labels': labels}


def maybe_load_dataset(use_fineweb: bool):
    if not use_fineweb:
        return None
    try:
        from datasets import load_dataset
        ds = load_dataset('HuggingFaceFW/fineweb', name='sample-10BT',
                          split='train', streaming=True)
        return iter(ds)
    except Exception as e:
        log(f"Could not load fineweb (will use random tokens): {e}")
        return None


@torch.no_grad()
def diagnose_param(grad_cpu: torch.Tensor, ks: list, alphas: list,
                   c_min: float = 0.05) -> dict:
    """한 파라미터 진단. grad는 CPU float32로 전달받음."""
    from optimizers.sc_dion import (
        _randomized_subspace, _greedy_logdet_select, _certificate
    )
    
    S = grad_cpu  # already float32 cpu
    transposed = S.size(0) > S.size(1)
    if transposed:
        S = S.t().contiguous()
    m, n = S.shape
    
    out = {'shape_oriented': (m, n), 'transposed': transposed}
    
    row_norms = S.norm(dim=1)
    if row_norms.numel() > 0:
        mean_rn = row_norms.mean().item()
        max_rn = row_norms.max().item()
        out['row_norm_spread'] = max_rn / max(mean_rn, 1e-12)
    out['frob_norm'] = float(S.norm().item())
    
    # Stable rank
    try:
        v = torch.randn(n)
        v = v / (v.norm() + 1e-12)
        for _ in range(15):
            v = S.t() @ (S @ v)
            v = v / (v.norm() + 1e-12)
        sigma_max = float((S @ v).norm().item())
        out['stable_rank'] = (out['frob_norm'] ** 2) / max(sigma_max ** 2, 1e-20)
    except Exception:
        out['stable_rank'] = -1.0
    
    out['certificates'] = {}
    for k in ks:
        k_eff = min(k, m - 1, n - 1)
        if k_eff < 1:
            continue
        try:
            U_k, tau = _randomized_subspace(S, k=k_eff)
        except Exception:
            continue
        for alpha in alphas:
            num_select = max(1, int(round(alpha * m)))
            if num_select >= m:
                continue
            try:
                sel_idx = _greedy_logdet_select(U_k, num_select)
                cert, omega = _certificate(U_k, sel_idx, tau)
            except Exception:
                continue
            key = f"k{k}_a{alpha}"
            out['certificates'][key] = {
                'cert': float(cert), 'omega': float(omega), 'tau': float(tau),
                'k_effective': k_eff, 'num_selected': num_select,
                'would_pass': bool(cert >= c_min),
            }
        del U_k
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--seq_len', type=int, default=1024)
    parser.add_argument('--num_batches', type=int, default=2)
    parser.add_argument('--ks', type=int, nargs='+', default=[4, 8, 16, 32])
    parser.add_argument('--alphas', type=float, nargs='+',
                        default=[0.5, 0.25, 0.125])
    parser.add_argument('--cert_threshold', type=float, default=0.05)
    parser.add_argument('--dtype', type=str, default='bfloat16')
    parser.add_argument('--use_fineweb', action='store_true')
    parser.add_argument('--no_gradient_checkpointing', action='store_true')
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--max_params', type=int, default=200)
    args = parser.parse_args()
    
    # CUDA 메모리 단편화 완화
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    
    is_dist, local_rank = setup_distributed()
    device = f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu'
    
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info(local_rank)
        log(f"GPU {local_rank} memory: {free/1e9:.1f} / {total/1e9:.1f} GB free")
    
    model, tokenizer = load_model_and_tokenizer(
        args.model_name, args.dtype,
        use_gradient_checkpointing=not args.no_gradient_checkpointing
    )
    model = model.to(device)
    model.train()
    
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info(local_rank)
        log(f"After model: {free/1e9:.1f} / {total/1e9:.1f} GB free")
    
    param_names_2d = [n for n, p in model.named_parameters()
                      if p.ndim == 2 and min(p.shape) >= 2 and p.requires_grad]
    log(f"Found {len(param_names_2d)} 2D matrix parameters")
    
    ds_iter = maybe_load_dataset(args.use_fineweb)
    
    log(f"\nComputing gradients on {args.num_batches} batches "
        f"(bs={args.batch_size}, seq={args.seq_len})...")
    t0 = time.time()
    for b in range(args.num_batches):
        if ds_iter is not None:
            batch = get_fineweb_batch(tokenizer, args.batch_size, args.seq_len,
                                      device, ds_iter)
        else:
            batch = get_dummy_batch(tokenizer, args.batch_size, args.seq_len, device)
        out = model(**batch)
        loss = out.loss / args.num_batches
        loss.backward()
        log(f"  batch {b+1}/{args.num_batches}: loss={out.loss.item():.4f}")
        del out, loss, batch
        torch.cuda.empty_cache()
    log(f"Gradient computation: {time.time()-t0:.1f}s")
    
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info(local_rank)
        log(f"After gradients: {free/1e9:.1f} / {total/1e9:.1f} GB free")
    
    # Sample params
    if len(param_names_2d) > args.max_params:
        idx = [int(i * (len(param_names_2d) - 1) / (args.max_params - 1))
               for i in range(args.max_params)]
        param_names_2d = [param_names_2d[i] for i in sorted(set(idx))]
        log(f"  Sampling {len(param_names_2d)} of total params")
    
    log("\nRunning diagnostics (grads moved to CPU one-by-one, GPU freed)...")
    results = {'config': vars(args), 'model_name': args.model_name, 'params': {}}
    name_to_param = dict(model.named_parameters())
    
    t0 = time.time()
    for i, name in enumerate(param_names_2d):
        if (i + 1) % 20 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i+1) * (len(param_names_2d) - i - 1)
            log(f"  [{i+1}/{len(param_names_2d)}] {name[:55]} "
                f"(elapsed {elapsed:.0f}s, ETA {eta:.0f}s)")
        
        p = name_to_param.get(name)
        if p is None or p.grad is None:
            continue
        
        # Move to CPU float32 — frees GPU memory while we do CPU SVD
        grad_cpu = p.grad.detach().float().cpu()
        # 진단 끝난 grad는 GPU에서 즉시 해제
        p.grad = None
        
        try:
            results['params'][name] = diagnose_param(
                grad_cpu, ks=args.ks, alphas=args.alphas,
                c_min=args.cert_threshold
            )
        except Exception as e:
            log(f"    [{name}] diagnostic failed: {e}")
        
        del grad_cpu
        if (i + 1) % 20 == 0:
            torch.cuda.empty_cache()
            gc.collect()
    log(f"Diagnostic: {time.time()-t0:.1f}s")
    
    # Summary
    log("\n=== SUMMARY ===")
    summary = {}
    for k in args.ks:
        for alpha in args.alphas:
            key = f"k{k}_a{alpha}"
            pc, tot = 0, 0
            cv, tv, ov = [], [], []
            for pd in results['params'].values():
                if key in pd.get('certificates', {}):
                    cd = pd['certificates'][key]
                    tot += 1
                    if cd['would_pass']:
                        pc += 1
                    cv.append(cd['cert']); tv.append(cd['tau']); ov.append(cd['omega'])
            if tot > 0:
                summary[key] = {
                    'pass_rate': pc/tot, 'mean_cert': sum(cv)/tot,
                    'mean_tau': sum(tv)/tot, 'mean_omega': sum(ov)/tot,
                    'total_params': tot,
                }
                log(f"  k={k:3d}, α={alpha}: pass {pc}/{tot} ({100*pc/tot:.1f}%), "
                    f"cert={sum(cv)/tot:+.3f}, τ={sum(tv)/tot:.3f}, "
                    f"ω={sum(ov)/tot:.3f}")
    
    spreads = [pd.get('row_norm_spread', 0) for pd in results['params'].values()
               if 'row_norm_spread' in pd]
    if spreads:
        sps = sorted(spreads)
        log(f"\nRow-norm spread: median={sps[len(sps)//2]:.1f}x, "
            f"max={max(spreads):.1f}x, "
            f"frac>20x={sum(1 for s in spreads if s>20)/len(spreads):.2f}")
        summary['row_norm_spread'] = {
            'median': sps[len(sps)//2], 'max': max(spreads),
            'frac_gt_20': sum(1 for s in spreads if s>20) / len(spreads),
        }
    
    srs = [pd.get('stable_rank', 0) for pd in results['params'].values()
           if pd.get('stable_rank', -1) > 0]
    if srs:
        srss = sorted(srs)
        log(f"Stable rank: median={srss[len(srss)//2]:.1f}, "
            f"min={min(srs):.1f}, max={max(srs):.1f}")
        summary['stable_rank'] = {'median': srss[len(srss)//2],
                                   'min': min(srs), 'max': max(srs)}
    
    results['summary'] = summary
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    log(f"\nSaved to {output_path}")
    
    log("\n=== DECISION GUIDE ===")
    best_key, best_pr = None, -1
    for key, stats in summary.items():
        if key.startswith('k') and stats.get('pass_rate', -1) > best_pr:
            best_pr = stats['pass_rate']
            best_key = key
    if best_key is None:
        log("⚠️  No diagnostic data")
    elif best_pr >= 0.5:
        log(f"✅ 유망: best ({best_key}) pass rate = {100*best_pr:.0f}%")
        log("    → Phase 2 학습 실험 진행 권장")
    elif best_pr >= 0.2:
        log(f"⚠️  부분적: best ({best_key}) pass rate = {100*best_pr:.0f}%")
        log("    → SC-Dion이 일부 layer에서만 활성화")
    else:
        log(f"❌ 비관: best ({best_key}) pass rate = {100*best_pr:.0f}%")
        log("    → 논문의 F4 진단이 이 모델에도 적용. SC-Dion이 폴백할 가능성 큼.")
    
    cleanup_distributed()


if __name__ == '__main__':
    main()
