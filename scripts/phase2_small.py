"""
Phase 2 (small-scale): Quick wall-clock comparison.

목표:
- 360M 모델 + 500 step + 3 옵티마이저 (Muon, Dion2-uniform, SC-Dion)
- 시간: 각 ~15분 (총 ~1시간)
- 측정: step time, loss, cert pass rate, OOM/NaN 안정성

각 옵티마이저는 별도 run으로 실행. config/log/result는 모두 다른 디렉토리에.

사용:
    bash launch_phase2_small.sh muon
    bash launch_phase2_small.sh dion2_uniform
    bash launch_phase2_small.sh sc_dion
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

from optimizers import Muon, Dion2Uniform, SCDionGPU, is_2d_param


def log(msg):
    print(msg, flush=True)


def load_model_and_tokenizer(model_name: str, dtype: str = 'bfloat16',
                              use_gradient_checkpointing: bool = False):
    """360M은 작아서 gradient checkpointing 안 써도 메모리 충분."""
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
    if use_gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model.config, 'use_cache'):
            model.config.use_cache = False
    return model, tokenizer


def get_data_iter(tokenizer, batch_size: int, seq_len: int, device: str, seed: int):
    """FineWeb streaming with packed sequences. 같은 seed면 같은 데이터 순서."""
    from datasets import load_dataset
    ds = load_dataset('HuggingFaceFW/fineweb', name='sample-10BT',
                     split='train', streaming=True)
    ds = ds.shuffle(buffer_size=1000, seed=seed)
    
    def gen():
        buf = []
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
                t = torch.tensor(chunk, device=device).view(batch_size, seq_len)
                yield t
    
    return gen()


def build_optimizer(name: str, model, args):
    """옵티마이저 분리: 2D matrix → main, 나머지 → AdamW."""
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
    import math
    progress = (step - args.warmup_steps) / max(1, args.max_steps - args.warmup_steps)
    return args.lr_min + 0.5 * (args.lr - args.lr_min) * (1 + math.cos(math.pi * progress))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='HuggingFaceTB/SmolLM2-360M')
    parser.add_argument('--optimizer', type=str, required=True,
                        choices=['muon', 'dion2_uniform', 'sc_dion'])
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--seq_len', type=int, default=2048)
    parser.add_argument('--max_steps', type=int, default=500)
    parser.add_argument('--warmup_steps', type=int, default=50)
    parser.add_argument('--eval_interval', type=int, default=50)
    parser.add_argument('--log_interval', type=int, default=10)
    # LR
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--lr_min', type=float, default=5e-5)
    parser.add_argument('--lr_adamw', type=float, default=2e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    # Optimizer-specific
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--subspace_rank', type=int, default=8)
    parser.add_argument('--cert_threshold', type=float, default=0.05)
    parser.add_argument('--refresh_period', type=int, default=20)
    parser.add_argument('--selector', type=str, default='topk',
                        choices=['greedy', 'block_greedy', 'topk'],
                        help='SC-Dion row selector: topk (fastest), '
                             'block_greedy (balanced), greedy (slowest, most accurate)')
    # System
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--fail_fast_step_time_ms', type=float, default=10000,
                        help='Step time이 이 ms를 초과하면 abort (sanity check)')
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
    
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info(0)
        log(f"  GPU free after model: {free/1e9:.1f} / {total/1e9:.1f} GB")
    
    # Optimizer
    optimizers, sc_dion_ref = build_optimizer(args.optimizer, model, args)
    
    # Data (same seed across optimizers for fair comparison)
    log(f"Loading data (seed={args.seed})...")
    data_iter = get_data_iter(tokenizer, args.batch_size, args.seq_len, device, args.seed)
    
    # ========== Train ==========
    log(f"\n{'='*70}")
    log(f"Training: {args.optimizer} for {args.max_steps} steps")
    log(f"  batch_size={args.batch_size}, seq_len={args.seq_len}")
    log(f"  lr={args.lr}, alpha={args.alpha}")
    log(f"{'='*70}\n")
    
    history = []
    step_times = []
    t_train_start = time.time()
    train_loss_ema = None
    
    for step in range(args.max_steps):
        # LR
        cur_lr = cosine_lr(step, args)
        for opt in optimizers:
            for g in opt.param_groups:
                if 'betas' in g:  # AdamW
                    pass  # 별도 고정 lr 사용 (단순화)
                else:
                    g['lr'] = cur_lr
        
        # Forward / backward
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = get_data_iter(tokenizer, args.batch_size, args.seq_len, device, args.seed)
            batch = next(data_iter)
        
        # === Time the step (just forward+backward+optimizer) ===
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        
        out = model(input_ids=batch, labels=batch.clone())
        loss = out.loss
        loss.backward()
        loss_val = loss.item()
        
        # Grad clip
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], args.grad_clip
        )
        
        # Optimizer step
        for opt in optimizers:
            opt.step()
        
        torch.cuda.synchronize()
        step_time_ms = (time.perf_counter() - t0) * 1000
        step_times.append(step_time_ms)
        
        del out, loss, batch
        
        # Sanity: fail fast on insane step time
        if step_time_ms > args.fail_fast_step_time_ms:
            log(f"  ⚠️ Step {step}: {step_time_ms:.0f}ms exceeds fail-fast threshold "
                f"({args.fail_fast_step_time_ms:.0f}ms). Continuing but flagging.")
        
        # NaN check
        if not (loss_val == loss_val):  # NaN
            log(f"  ❌ NaN loss at step {step}. Aborting.")
            break
        
        # EMA
        if train_loss_ema is None:
            train_loss_ema = loss_val
        else:
            train_loss_ema = 0.9 * train_loss_ema + 0.1 * loss_val
        
        # Logging
        if (step + 1) % args.log_interval == 0 or step < 5:
            cert_info = ""
            if sc_dion_ref is not None:
                cs = sc_dion_ref.get_cert_stats()
                cert_info = (f" | cert_pass={cs['cert_pass_rate']:.2f} "
                            f"sc_frac={cs['sc_dion_fraction']:.2f}")
            log(f"  step {step+1:4d}/{args.max_steps} | "
                f"loss {loss_val:.4f} (ema {train_loss_ema:.4f}) | "
                f"lr {cur_lr:.2e} | "
                f"{step_time_ms:6.1f} ms"
                f"{cert_info}")
        
        # Periodic history snapshot
        if (step + 1) % args.eval_interval == 0:
            entry = {
                'step': step + 1,
                'loss': loss_val,
                'loss_ema': train_loss_ema,
                'step_time_ms': step_time_ms,
                'elapsed_sec': time.time() - t_train_start,
                'lr': cur_lr,
            }
            if sc_dion_ref is not None:
                entry['sc_dion_stats'] = sc_dion_ref.get_cert_stats()
            history.append(entry)
            
            # Save partial
            with open(output_dir / 'history.json', 'w') as f:
                json.dump({
                    'history': history,
                    'step_times': step_times,
                    'config': vars(args),
                }, f, indent=2)
    
    # ========== Final stats ==========
    total_elapsed = time.time() - t_train_start
    import numpy as np
    step_times_arr = np.array(step_times)
    
    # Trimmed median (drop top/bottom 10% for robustness)
    trim_lo, trim_hi = np.percentile(step_times_arr, [10, 90])
    trimmed = step_times_arr[(step_times_arr >= trim_lo) & (step_times_arr <= trim_hi)]
    
    log(f"\n{'='*70}")
    log(f"FINAL SUMMARY: {args.optimizer}")
    log(f"{'='*70}")
    log(f"Total wall-clock:  {total_elapsed:.1f} s ({total_elapsed/60:.1f} min)")
    log(f"Steps completed:   {len(step_times)} / {args.max_steps}")
    log(f"")
    log(f"Step time:")
    log(f"  median:          {np.median(step_times_arr):.1f} ms")
    log(f"  trimmed median:  {np.median(trimmed):.1f} ms")
    log(f"  mean:            {np.mean(step_times_arr):.1f} ms")
    log(f"  min/max:         {np.min(step_times_arr):.1f} / {np.max(step_times_arr):.1f} ms")
    log(f"  p10/p90:         {np.percentile(step_times_arr, 10):.1f} / "
        f"{np.percentile(step_times_arr, 90):.1f} ms")
    log(f"")
    log(f"Loss:")
    log(f"  initial:         {step_times[0] if False else (history[0]['loss'] if history else 0):.4f}"
        if history else "  (no eval)")
    log(f"  final:           {train_loss_ema:.4f} (EMA)")
    log(f"  final (raw):     {step_times_arr[-1] if False else (history[-1]['loss'] if history else 0):.4f}"
        if history else "")
    
    if sc_dion_ref is not None:
        cs = sc_dion_ref.get_cert_stats()
        log(f"")
        log(f"SC-Dion stats:")
        log(f"  Cert eval count:    {cs['cert_eval_count']}")
        log(f"  Pass rate:          {cs['cert_pass_rate']:.3f}")
        log(f"  SC-Dion mode steps: {cs['sc_dion_steps']}")
        log(f"  Fallback steps:     {cs['fallback_steps']}")
        log(f"  SC-Dion fraction:   {cs['sc_dion_fraction']:.3f}")
        if 'recent_tau_mean' in cs:
            log(f"  Recent mean τ:      {cs['recent_tau_mean']:.3f}")
        if 'recent_omega_mean' in cs:
            log(f"  Recent mean ω:      {cs['recent_omega_mean']:.3f}")
    
    # Save final
    final_data = {
        'history': history,
        'step_times': step_times,
        'step_time_median_ms': float(np.median(step_times_arr)),
        'step_time_trimmed_median_ms': float(np.median(trimmed)),
        'step_time_mean_ms': float(np.mean(step_times_arr)),
        'total_elapsed_sec': total_elapsed,
        'final_loss_ema': train_loss_ema,
        'config': vars(args),
    }
    if sc_dion_ref is not None:
        final_data['sc_dion_final_stats'] = sc_dion_ref.get_cert_stats()
    
    with open(output_dir / 'history.json', 'w') as f:
        json.dump(final_data, f, indent=2)
    
    log(f"\nResults saved to {output_dir / 'history.json'}")


if __name__ == '__main__':
    main()
