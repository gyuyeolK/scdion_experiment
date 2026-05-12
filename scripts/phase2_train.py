"""
Phase 2: Training comparison (Muon vs Dion2-uniform vs SC-Dion).

핵심 측정:
- 각 옵티마이저의 validation loss curve (T_D/T_M 측정)
- Per-step wall-clock breakdown (forward/backward/optimizer/comm)
- ρ, ρ_flop, ρ_byte share parameters
- 최종 시간/토큰 vs target loss

사용:
    torchrun --nproc_per_node=8 phase2_train.py \\
        --model_name HuggingFaceTB/SmolLM2-1.7B \\
        --optimizer sc_dion \\
        --alpha_u 0.5 \\
        --max_steps 2000 \\
        --output_dir runs/scdion_a0.5
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from contextlib import nullcontext

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    MixedPrecision, ShardingStrategy, BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
import functools

# Add parent for optimizers import
sys.path.insert(0, str(Path(__file__).parent.parent))
from optimizers import Muon, Dion2Uniform, SCDion, is_2d_param


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
        return True, local_rank, dist.get_world_size()
    return False, 0, 1


def get_transformer_wrap_policy(model):
    """모델 architecture에 맞춰 FSDP wrap policy를 자동 결정."""
    # Find a representative transformer block class
    candidates = []
    for n, m in model.named_modules():
        cls_name = m.__class__.__name__
        if 'DecoderLayer' in cls_name or 'Block' in cls_name or 'Layer' in cls_name:
            if hasattr(m, 'self_attn') or hasattr(m, 'attn') or hasattr(m, 'attention'):
                candidates.append(type(m))
                break
    if candidates:
        wrap_cls = set(candidates)
        log(f"FSDP wrap policy: {[c.__name__ for c in wrap_cls]}")
        return functools.partial(transformer_auto_wrap_policy,
                                 transformer_layer_cls=wrap_cls)
    log("WARN: could not auto-detect transformer block; FSDP wrap defaulting")
    return None


def build_optimizer(name: str, model, args):
    """
    옵티마이저 build. 2D 행렬 파라미터는 Muon/Dion 계열, 나머지는 AdamW.
    """
    matrix_params, other_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if is_2d_param(p):
            matrix_params.append(p)
        else:
            other_params.append(p)
    log(f"  Matrix params: {len(matrix_params)}, Other (AdamW): {len(other_params)}")
    
    # AdamW for non-matrix
    adamw = torch.optim.AdamW(other_params, lr=args.lr_adamw, weight_decay=0.0,
                               betas=(0.9, 0.95))
    
    if name == 'muon':
        main_opt = Muon(matrix_params, lr=args.lr, momentum=args.momentum,
                       ns_steps=args.ns_steps, weight_decay=args.weight_decay)
    elif name == 'dion2_uniform':
        main_opt = Dion2Uniform(matrix_params, lr=args.lr, alpha=args.alpha,
                                mu=args.momentum, ns_steps=args.ns_steps,
                                weight_decay=args.weight_decay)
    elif name == 'sc_dion':
        main_opt = SCDion(matrix_params, lr=args.lr,
                          alpha_u=args.alpha, alpha_d=args.alpha_d,
                          mu=args.momentum, ns_steps=args.ns_steps,
                          subspace_rank=args.subspace_rank,
                          cert_threshold=args.cert_threshold,
                          refresh_period=args.refresh_period,
                          weight_decay=args.weight_decay)
    elif name == 'adamw':
        # AdamW only (모든 파라미터). matrix_params도 AdamW로.
        adamw = torch.optim.AdamW(matrix_params + other_params,
                                   lr=args.lr_adamw, weight_decay=args.weight_decay,
                                   betas=(0.9, 0.95))
        return [adamw], None
    else:
        raise ValueError(f"Unknown optimizer: {name}")
    
    return [main_opt, adamw], main_opt


def get_lr(step: int, args) -> float:
    """Linear warmup + cosine decay."""
    if step < args.warmup_steps:
        return args.lr * (step + 1) / args.warmup_steps
    progress = (step - args.warmup_steps) / max(1, args.max_steps - args.warmup_steps)
    return args.lr_min + 0.5 * (args.lr - args.lr_min) * (1 + math.cos(math.pi * progress))


def get_data_iter(tokenizer, args, device):
    from datasets import load_dataset
    log(f"Loading dataset {args.dataset}...")
    ds = load_dataset(args.dataset, name=args.dataset_config,
                     split='train', streaming=True)
    ds = ds.shuffle(buffer_size=10000, seed=args.seed)
    
    def gen():
        buf_ids = []
        for ex in ds:
            text = ex.get('text', '') or ex.get('content', '')
            if not text:
                continue
            enc = tokenizer(text, truncation=False, add_special_tokens=False)
            buf_ids.extend(enc['input_ids'])
            buf_ids.append(tokenizer.eos_token_id or 0)
            while len(buf_ids) >= args.seq_len * args.batch_size:
                chunk = buf_ids[:args.seq_len * args.batch_size]
                buf_ids = buf_ids[args.seq_len * args.batch_size:]
                t = torch.tensor(chunk, device=device).view(args.batch_size, args.seq_len)
                yield t
    
    return gen()


class Timer:
    """GPU sync 포함 정밀 타이머."""
    def __init__(self):
        self.t = {}
        self.start_evt = None
    
    def reset(self):
        torch.cuda.synchronize()
        self.t = {k: 0.0 for k in self.t}
    
    def section(self, name):
        return _TimerCtx(self, name)


class _TimerCtx:
    def __init__(self, timer, name):
        self.timer = timer
        self.name = name
    
    def __enter__(self):
        torch.cuda.synchronize()
        self.t0 = time.perf_counter()
        return self
    
    def __exit__(self, *a):
        torch.cuda.synchronize()
        dt = time.perf_counter() - self.t0
        self.timer.t[self.name] = self.timer.t.get(self.name, 0.0) + dt


def main():
    parser = argparse.ArgumentParser()
    # Model
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--dtype', type=str, default='bfloat16')
    # Data
    parser.add_argument('--dataset', type=str, default='HuggingFaceFW/fineweb')
    parser.add_argument('--dataset_config', type=str, default='sample-10BT')
    parser.add_argument('--seq_len', type=int, default=2048)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--grad_accum_steps', type=int, default=1)
    # Optimizer
    parser.add_argument('--optimizer', type=str, required=True,
                        choices=['muon', 'dion2_uniform', 'sc_dion', 'adamw'])
    parser.add_argument('--lr', type=float, default=2e-3)
    parser.add_argument('--lr_min', type=float, default=2e-4)
    parser.add_argument('--lr_adamw', type=float, default=3e-4)
    parser.add_argument('--momentum', type=float, default=0.95)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--ns_steps', type=int, default=5)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--alpha_d', type=float, default=1.0)
    parser.add_argument('--subspace_rank', type=int, default=8)
    parser.add_argument('--cert_threshold', type=float, default=0.05)
    parser.add_argument('--refresh_period', type=int, default=10)
    # Schedule
    parser.add_argument('--max_steps', type=int, default=2000)
    parser.add_argument('--warmup_steps', type=int, default=100)
    parser.add_argument('--eval_interval', type=int, default=50)
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--eval_steps', type=int, default=20,
                        help='eval에서 사용할 batch 수')
    parser.add_argument('--grad_clip', type=float, default=1.0)
    # System
    parser.add_argument('--use_fsdp', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--target_losses', type=float, nargs='+',
                        default=[5.0, 4.5, 4.0, 3.5, 3.0])
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    is_dist, local_rank, world_size = setup_distributed()
    device = f'cuda:{local_rank}'
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if is_main():
        with open(output_dir / 'config.json', 'w') as f:
            json.dump(vars(args), f, indent=2)
    
    # === Model ===
    from transformers import AutoModelForCausalLM, AutoTokenizer
    log(f"Loading {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    torch_dtype = {'bfloat16': torch.bfloat16, 'float16': torch.float16,
                   'float32': torch.float32}[args.dtype]
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch_dtype)
    model = model.to(device)
    
    if is_dist and args.use_fsdp:
        wrap_policy = get_transformer_wrap_policy(model)
        mp = MixedPrecision(param_dtype=torch_dtype, reduce_dtype=torch.float32,
                            buffer_dtype=torch_dtype)
        model = FSDP(model, auto_wrap_policy=wrap_policy,
                     mixed_precision=mp,
                     sharding_strategy=ShardingStrategy.FULL_SHARD,
                     backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
                     device_id=local_rank)
        log(f"Wrapped in FSDP across {world_size} GPUs")
    elif is_dist:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[local_rank])
    
    # Param count
    n_params = sum(p.numel() for p in model.parameters())
    log(f"Total params: {n_params/1e9:.2f}B")
    
    # === Optimizer ===
    log(f"Building optimizer: {args.optimizer}")
    optimizers, sc_dion_ref = build_optimizer(args.optimizer, model, args)
    
    # === Data ===
    data_iter = get_data_iter(tokenizer, args, device)
    
    # === Train loop ===
    log("Starting training...")
    history = []
    target_hits = {t: None for t in args.target_losses}  # step at which target was first hit
    timer = Timer()
    
    model.train()
    train_loss_ema = None
    t_train_start = time.time()
    
    for step in range(args.max_steps):
        timer.t = {}  # reset
        
        # LR schedule (warmup + cosine).
        # 두 옵티마이저(main + AdamW for 1D)를 별도 스케일로 다룸.
        cur_lr_main = get_lr(step, args)
        # AdamW LR은 main과 같은 비율로 스케일링
        warmup_frac = min(1.0, (step + 1) / max(1, args.warmup_steps))
        if step < args.warmup_steps:
            cur_lr_adamw = args.lr_adamw * warmup_frac
        else:
            progress = (step - args.warmup_steps) / max(1, args.max_steps - args.warmup_steps)
            cur_lr_adamw = args.lr_adamw * 0.1 + 0.9 * args.lr_adamw * 0.5 * (1 + math.cos(math.pi * progress))
        
        for opt in optimizers:
            for g in opt.param_groups:
                # AdamW의 betas는 (0.9, 0.95), Muon/Dion은 momentum
                if 'betas' in g:
                    g['lr'] = cur_lr_adamw
                else:
                    g['lr'] = cur_lr_main
        
        # Forward/backward (accumulation)
        accum_loss = 0.0
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        
        with timer.section('forward_backward'):
            for micro in range(args.grad_accum_steps):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    log("Data exhausted, restarting")
                    data_iter = get_data_iter(tokenizer, args, device)
                    batch = next(data_iter)
                
                # CLM: shift internally via labels
                input_ids = batch
                labels = input_ids.clone()
                # No mask token for streaming text; ignore -100 for padded
                
                out = model(input_ids=input_ids, labels=labels)
                loss = out.loss / args.grad_accum_steps
                loss.backward()
                accum_loss += loss.item()
        
        # Grad clip
        with timer.section('grad_clip'):
            if args.grad_clip > 0:
                if isinstance(model, FSDP):
                    model.clip_grad_norm_(args.grad_clip)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        
        # Optimizer step (별도 측정)
        with timer.section('optimizer_step'):
            for opt in optimizers:
                opt.step()
        
        # Aggregate timing (논문의 ρ 분해)
        step_total = sum(timer.t.values())
        rho_fb = timer.t.get('forward_backward', 0) / max(step_total, 1e-9)
        rho_opt = timer.t.get('optimizer_step', 0) / max(step_total, 1e-9)
        rho_other = 1.0 - rho_fb - rho_opt
        
        # Loss EMA
        if train_loss_ema is None:
            train_loss_ema = accum_loss
        else:
            train_loss_ema = 0.9 * train_loss_ema + 0.1 * accum_loss
        
        if (step + 1) % args.log_interval == 0:
            cert_info = ""
            if sc_dion_ref is not None:
                cs = sc_dion_ref.get_cert_stats()
                cert_info = (f" | cert_pass={cs['cert_pass_rate']:.2f} "
                            f"τ={cs['recent_tau_mean']:.2f} ω={cs['recent_omega_mean']:.2f}")
            log(f"step {step+1:5d}/{args.max_steps} | "
                f"loss {accum_loss:.4f} (ema {train_loss_ema:.4f}) | "
                f"lr {cur_lr_main:.2e} | "
                f"step_time {step_total*1000:.1f}ms "
                f"(fb={rho_fb:.2f} opt={rho_opt:.2f} other={rho_other:.2f})"
                f"{cert_info}")
        
        # Eval
        if (step + 1) % args.eval_interval == 0:
            model.eval()
            with torch.no_grad():
                eval_losses = []
                for _ in range(args.eval_steps):
                    try:
                        eb = next(data_iter)
                    except StopIteration:
                        data_iter = get_data_iter(tokenizer, args, device)
                        eb = next(data_iter)
                    eo = model(input_ids=eb, labels=eb.clone())
                    eval_losses.append(eo.loss.item())
                eval_loss = sum(eval_losses) / len(eval_losses)
                if is_dist:
                    el = torch.tensor([eval_loss], device=device)
                    dist.all_reduce(el, op=dist.ReduceOp.AVG)
                    eval_loss = el.item()
            
            elapsed = time.time() - t_train_start
            log(f"  >>> EVAL @ step {step+1}: val_loss={eval_loss:.4f}  "
                f"(elapsed {elapsed/60:.1f} min)")
            
            entry = {
                'step': step + 1,
                'train_loss': accum_loss,
                'eval_loss': eval_loss,
                'elapsed_sec': elapsed,
                'step_time_ms': step_total * 1000,
                'rho_fb': rho_fb,
                'rho_opt': rho_opt,
                'rho_other': rho_other,
                'lr': cur_lr_main,
            }
            if sc_dion_ref is not None:
                entry['sc_dion_stats'] = sc_dion_ref.get_cert_stats()
            history.append(entry)
            
            # Target hit?
            for t in args.target_losses:
                if target_hits[t] is None and eval_loss <= t:
                    target_hits[t] = {'step': step + 1, 'elapsed_sec': elapsed}
                    log(f"  🎯 TARGET {t} hit at step {step+1} (elapsed {elapsed/60:.1f}m)")
            
            if is_main():
                with open(output_dir / 'history.json', 'w') as f:
                    json.dump({
                        'history': history,
                        'target_hits': target_hits,
                        'config': vars(args),
                    }, f, indent=2)
            
            model.train()
    
    # === Final summary ===
    total_elapsed = time.time() - t_train_start
    log(f"\n=== Final summary ===")
    log(f"Total wall-clock: {total_elapsed/60:.1f} min "
        f"({total_elapsed/args.max_steps*1000:.1f} ms/step avg)")
    log(f"Target hits:")
    for t, info in target_hits.items():
        if info is None:
            log(f"  loss {t}: NOT REACHED")
        else:
            log(f"  loss {t}: step {info['step']}, {info['elapsed_sec']/60:.1f} min")
    
    if sc_dion_ref is not None:
        cs = sc_dion_ref.get_cert_stats()
        log(f"\nSC-Dion certificate stats:")
        log(f"  pass rate: {cs['cert_pass_rate']:.3f} ({cs['cert_pass_count']}/{cs['cert_pass_count']+cs['cert_fail_count']})")
        log(f"  mean τ: {cs['recent_tau_mean']:.3f}")
        log(f"  mean ω: {cs['recent_omega_mean']:.3f}")
    
    if is_main():
        with open(output_dir / 'history.json', 'w') as f:
            json.dump({
                'history': history,
                'target_hits': target_hits,
                'total_elapsed_sec': total_elapsed,
                'final_eval_loss': history[-1]['eval_loss'] if history else None,
                'config': vars(args),
                'sc_dion_final_stats': sc_dion_ref.get_cert_stats() if sc_dion_ref else None,
            }, f, indent=2)
    
    if is_dist:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
