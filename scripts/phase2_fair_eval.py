"""
Phase 2 + Fair Eval: 같은 eval batches로 모든 옵티마이저 비교.

핵심: 학습 시작 전 고정 eval batches를 미리 생성. 학습 중 random state 영향
없이 항상 같은 batch들로 evaluation.

이걸 통해 loss 비교의 "데이터 차이" artifact 제거.

사용:
    python scripts/phase2_fair_eval.py --optimizer muon ...
"""
import argparse
import gc
import json
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from optimizers import Muon, Dion2Uniform, SCDionGPU, is_2d_param


def log(msg):
    print(msg, flush=True)


def load_model_and_tokenizer(model_name: str, dtype: str = 'bfloat16'):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch_dtype = {'bfloat16': torch.bfloat16}[dtype]
    log(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch_dtype, attn_implementation='sdpa',
            trust_remote_code=True,
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch_dtype, attn_implementation='sdpa',
            trust_remote_code=True,
        )
    try:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={'use_reentrant': False}
        )
    except (TypeError, ValueError):
        try:
            model.gradient_checkpointing_enable()
        except Exception:
            pass
    if hasattr(model.config, 'use_cache'):
        model.config.use_cache = False
    return model, tokenizer


def collect_batches(tokenizer, n_batches: int, batch_size: int, seq_len: int,
                    device: str, seed: int, skip_first: int = 0):
    """FineWeb에서 batches를 미리 수집. seed로 결정론적."""
    from datasets import load_dataset
    ds = load_dataset('HuggingFaceFW/fineweb', name='sample-10BT',
                     split='train', streaming=True)
    ds = ds.shuffle(buffer_size=1000, seed=seed)
    
    buf = []
    batches = []
    skipped = 0
    for ex in ds:
        text = ex.get('text', '')
        if not text:
            continue
        ids = tokenizer(text, truncation=False, add_special_tokens=False)['input_ids']
        buf.extend(ids)
        buf.append(tokenizer.eos_token_id or 0)
        while len(buf) >= seq_len * batch_size:
            chunk = buf[:seq_len * batch_size]
            buf = buf[seq_len * batch_size:]
            if skipped < skip_first:
                skipped += 1
                continue
            t = torch.tensor(chunk, device=device).view(batch_size, seq_len)
            batches.append(t)
            if len(batches) >= n_batches:
                return batches
    return batches  # 부족하면 그냥 있는만큼


def build_optimizer(name: str, model, args):
    matrix_params = [p for p in model.parameters() if is_2d_param(p) and p.requires_grad]
    other_params = [p for p in model.parameters() if not is_2d_param(p) and p.requires_grad]
    log(f"  Matrix params: {len(matrix_params)}, Other (AdamW): {len(other_params)}")
    
    adamw = torch.optim.AdamW(other_params, lr=args.lr_adamw,
                              weight_decay=0.0, betas=(0.9, 0.95))
    
    if name == 'muon':
        main_opt = Muon(matrix_params, lr=args.lr, momentum=0.95,
                       ns_steps=5, weight_decay=args.weight_decay)
        sc_dion_ref = None
    elif name == 'dion2_uniform':
        main_opt = Dion2Uniform(matrix_params, lr=args.lr, alpha=args.alpha,
                                mu=0.95, ns_steps=5, weight_decay=args.weight_decay)
        sc_dion_ref = None
    elif name == 'sc_dion':
        main_opt = SCDionGPU(matrix_params, lr=args.lr,
                             alpha_u=args.alpha, alpha_d=1.0, mu=0.95,
                             ns_steps=5, subspace_rank=args.subspace_rank,
                             cert_threshold=args.cert_threshold,
                             refresh_period=args.refresh_period,
                             selector=args.selector,
                             weight_decay=args.weight_decay)
        sc_dion_ref = main_opt
    else:
        raise ValueError(f"Unknown optimizer: {name}")
    return [main_opt, adamw], sc_dion_ref


def cosine_lr(step, args):
    if step < args.warmup_steps:
        return args.lr * (step + 1) / args.warmup_steps
    progress = (step - args.warmup_steps) / max(1, args.max_steps - args.warmup_steps)
    return args.lr_min + 0.5 * (args.lr - args.lr_min) * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(model, eval_batches):
    """Fixed eval batches로 loss 측정. 모델 mode 보존."""
    was_training = model.training
    model.eval()
    losses = []
    for batch in eval_batches:
        out = model(input_ids=batch, labels=batch.clone())
        losses.append(out.loss.item())
    if was_training:
        model.train()
    return sum(losses) / len(losses)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='HuggingFaceTB/SmolLM2-1.7B')
    parser.add_argument('--optimizer', type=str, required=True,
                        choices=['muon', 'dion2_uniform', 'sc_dion'])
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--seq_len', type=int, default=2048)
    parser.add_argument('--max_steps', type=int, default=200)
    parser.add_argument('--warmup_steps', type=int, default=20)
    parser.add_argument('--eval_interval', type=int, default=25)
    parser.add_argument('--eval_n_batches', type=int, default=4,
                        help='Eval당 batch 수 (4 × bs2 × seq2048 = 16k tokens)')
    parser.add_argument('--log_interval', type=int, default=25)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--lr_min', type=float, default=3e-5)
    parser.add_argument('--lr_adamw', type=float, default=1.5e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--subspace_rank', type=int, default=8)
    parser.add_argument('--cert_threshold', type=float, default=0.05)
    parser.add_argument('--refresh_period', type=int, default=20)
    parser.add_argument('--selector', type=str, default='topk',
                        choices=['greedy', 'block_greedy', 'topk'])
    parser.add_argument('--seed', type=int, default=42)
    # Eval seed는 학습 seed와 분리 (모든 옵티마이저가 같은 eval 사용)
    parser.add_argument('--eval_seed', type=int, default=99999)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--fail_fast_step_time_ms', type=float, default=30000)
    args = parser.parse_args()
    
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    torch.manual_seed(args.seed)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    # Model
    model, tokenizer = load_model_and_tokenizer(args.model_name)
    model = model.to(device)
    model.train()
    n_params = sum(p.numel() for p in model.parameters())
    log(f"  Total params: {n_params/1e6:.1f}M")
    
    # ===== Prebuild fixed eval batches (EVAL_SEED) =====
    log(f"\nPrebuilding {args.eval_n_batches} eval batches (eval_seed={args.eval_seed})...")
    eval_batches = collect_batches(
        tokenizer, args.eval_n_batches,
        args.batch_size, args.seq_len, device,
        seed=args.eval_seed, skip_first=0
    )
    log(f"  Got {len(eval_batches)} eval batches")
    
    # ===== Prebuild train batches =====
    # 충분한 train batches 미리 만들기 (eval seed랑 다른 seed)
    n_train_needed = args.max_steps + 20
    log(f"Prebuilding {n_train_needed} train batches (train_seed={args.seed})...")
    # Skip first N train batches that overlap with eval data
    train_batches = collect_batches(
        tokenizer, n_train_needed,
        args.batch_size, args.seq_len, device,
        seed=args.seed, skip_first=args.eval_n_batches
    )
    log(f"  Got {len(train_batches)} train batches")
    
    # Reset random for optimizer initialization (after data prep)
    torch.manual_seed(args.seed)
    
    # Optimizer
    optimizers, sc_dion_ref = build_optimizer(args.optimizer, model, args)
    
    # ===== Train =====
    log(f"\n{'='*70}")
    log(f"Training: {args.optimizer} (α={args.alpha}) for {args.max_steps} steps")
    log(f"Eval: every {args.eval_interval} steps on {args.eval_n_batches} fixed batches")
    log(f"{'='*70}\n")
    
    history = []
    step_times = []
    t_train_start = time.time()
    
    # Initial eval (step 0)
    eval_loss_init = evaluate(model, eval_batches)
    log(f"  step    0/{args.max_steps} | initial eval_loss = {eval_loss_init:.4f}")
    history.append({'step': 0, 'eval_loss': eval_loss_init, 'train_loss': None,
                    'elapsed_sec': 0, 'step_time_ms': 0})
    
    for step in range(args.max_steps):
        cur_lr = cosine_lr(step, args)
        for opt in optimizers:
            for g in opt.param_groups:
                if 'betas' not in g:
                    g['lr'] = cur_lr
        
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        
        batch = train_batches[step % len(train_batches)]
        
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        
        out = model(input_ids=batch, labels=batch.clone())
        loss = out.loss
        loss.backward()
        loss_val = loss.item()
        
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], args.grad_clip
        )
        for opt in optimizers:
            opt.step()
        
        torch.cuda.synchronize()
        step_time_ms = (time.perf_counter() - t0) * 1000
        step_times.append(step_time_ms)
        
        del out, loss
        
        if not (loss_val == loss_val):
            log(f"  ❌ NaN at step {step}. Aborting.")
            break
        
        if (step + 1) % args.log_interval == 0 or step < 3:
            cert_info = ""
            if sc_dion_ref is not None:
                cs = sc_dion_ref.get_cert_stats()
                cert_info = f" | cert_pass={cs['cert_pass_rate']:.2f}"
            log(f"  step {step+1:4d}/{args.max_steps} | "
                f"train {loss_val:.4f} | "
                f"lr {cur_lr:.2e} | {step_time_ms:6.1f} ms{cert_info}")
        
        # Eval at intervals
        if (step + 1) % args.eval_interval == 0 or step == args.max_steps - 1:
            eval_loss = evaluate(model, eval_batches)
            entry = {
                'step': step + 1,
                'eval_loss': eval_loss,
                'train_loss': loss_val,
                'elapsed_sec': time.time() - t_train_start,
                'step_time_ms': step_time_ms,
            }
            if sc_dion_ref is not None:
                entry['sc_dion_stats'] = sc_dion_ref.get_cert_stats()
            history.append(entry)
            log(f"        >>> eval_loss = {eval_loss:.4f}")
            
            with open(output_dir / 'history.json', 'w') as f:
                json.dump({'history': history, 'step_times': step_times,
                           'config': vars(args)}, f, indent=2)
    
    # ===== Final =====
    total_elapsed = time.time() - t_train_start
    import numpy as np
    step_times_arr = np.array(step_times)
    
    log(f"\n{'='*70}")
    log(f"FINAL: {args.optimizer} (α={args.alpha})")
    log(f"{'='*70}")
    log(f"Total wall-clock:  {total_elapsed:.1f} s")
    log(f"Step time median:  {np.median(step_times_arr):.1f} ms")
    
    # Loss trajectory
    log(f"\nEval loss trajectory:")
    for h in history:
        log(f"  step {h['step']:4d}: eval_loss = {h['eval_loss']:.4f}")
    
    initial_eval = history[0]['eval_loss']
    final_eval = history[-1]['eval_loss']
    log(f"\nLoss improvement: {initial_eval:.4f} → {final_eval:.4f} (Δ={final_eval-initial_eval:+.4f})")
    
    if sc_dion_ref is not None:
        cs = sc_dion_ref.get_cert_stats()
        log(f"\nSC-Dion: pass rate {cs['cert_pass_rate']:.3f}, "
            f"sc_frac {cs['sc_dion_fraction']:.3f}, "
            f"τ={cs.get('recent_tau_mean', 0):.3f}, "
            f"ω={cs.get('recent_omega_mean', 0):.3f}")
    
    final_data = {
        'history': history,
        'step_times': step_times,
        'step_time_median_ms': float(np.median(step_times_arr)),
        'step_time_trimmed_median_ms': float(np.median(
            step_times_arr[
                (step_times_arr >= np.percentile(step_times_arr, 10)) &
                (step_times_arr <= np.percentile(step_times_arr, 90))
            ])),
        'step_time_mean_ms': float(np.mean(step_times_arr)),
        'total_elapsed_sec': total_elapsed,
        'initial_eval_loss': initial_eval,
        'final_eval_loss': final_eval,
        'config': vars(args),
    }
    if sc_dion_ref is not None:
        final_data['sc_dion_final_stats'] = sc_dion_ref.get_cert_stats()
    with open(output_dir / 'history.json', 'w') as f:
        json.dump(final_data, f, indent=2)


if __name__ == '__main__':
    main()
