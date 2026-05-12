"""
Phase 1C: 모델 크기별 trend 진단.

여러 모델에 대해 같은 진단을 돌려 SC-Dion 통과율, τ, ω, spread의 모델 크기 trend 확인.

핵심 질문:
- 통과율이 모델 크기와 함께 증가하는가? (논문 가설)
- spread가 증가하는가? (row-norm-proxy 가능성)
- τ가 감소하는가? (더 dominant subspace)
- stable rank가 감소하는가? (더 저랭크)

각 모델은 단일 데이터 소스(FineWeb)로만 진단. 검증은 이전에 끝남.

사용법:
    bash launch_phase1_modelsweep.sh
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


def log(msg):
    print(msg, flush=True)


def load_model_and_tokenizer(model_name: str, dtype: str = 'bfloat16'):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch_dtype = {'bfloat16': torch.bfloat16}[dtype]
    log(f"  Loading {model_name}...")
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
        except Exception as e:
            log(f"  warn: could not enable gradient checkpointing: {e}")
    if hasattr(model.config, 'use_cache'):
        model.config.use_cache = False
    return model, tokenizer


def load_fineweb():
    from datasets import load_dataset
    ds = load_dataset('HuggingFaceFW/fineweb', name='sample-10BT',
                     split='train', streaming=True)
    return iter(ds)


def get_batch(it, tokenizer, batch_size: int, seq_len: int, device: str):
    texts = []
    while len(texts) < batch_size:
        try:
            ex = next(it)
            text = ex.get('text', '')
            if not text or len(text) < 50:
                continue
            texts.append(text)
        except StopIteration:
            break
    if len(texts) < batch_size:
        # Pad with random tokens if exhausted
        vocab_size = tokenizer.vocab_size
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        return {'input_ids': input_ids, 'labels': input_ids.clone()}
    enc = tokenizer(texts, max_length=seq_len, truncation=True,
                    padding='max_length', return_tensors='pt').to(device)
    labels = enc['input_ids'].clone()
    labels[enc['attention_mask'] == 0] = -100
    return {'input_ids': enc['input_ids'],
            'attention_mask': enc['attention_mask'],
            'labels': labels}


@torch.no_grad()
def diagnose_param(grad_cpu: torch.Tensor, ks: list, alphas: list,
                   c_min: float = 0.05) -> dict:
    from optimizers.sc_dion import (
        _randomized_subspace, _greedy_logdet_select, _certificate
    )
    S = grad_cpu
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
            out['certificates'][f"k{k}_a{alpha}"] = {
                'cert': float(cert), 'omega': float(omega), 'tau': float(tau),
                'k_effective': k_eff, 'num_selected': num_select,
                'would_pass': bool(cert >= c_min),
            }
        del U_k
    return out


def run_one_model(model_name: str, args, device: str) -> dict:
    """한 모델 진단."""
    log(f"\n{'='*70}")
    log(f"=== Model: {model_name} ===")
    log(f"{'='*70}")
    
    t_total = time.time()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        free, total = torch.cuda.mem_get_info(0)
        log(f"GPU free: {free/1e9:.1f} / {total/1e9:.1f} GB")
    
    try:
        model, tokenizer = load_model_and_tokenizer(model_name, args.dtype)
    except Exception as e:
        log(f"  ERROR loading model: {e}")
        return {'error': str(e), 'model_name': model_name}
    
    model = model.to(device)
    model.train()
    
    n_params_total = sum(p.numel() for p in model.parameters())
    log(f"  Total params: {n_params_total/1e9:.2f}B")
    
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info(0)
        log(f"  After load: {free/1e9:.1f} / {total/1e9:.1f} GB free")
    
    # Gradient
    log(f"  Computing {args.num_batches} batches (bs={args.batch_size}, seq={args.seq_len})...")
    try:
        fw_iter = load_fineweb()
    except Exception as e:
        log(f"  WARN: fineweb failed: {e}, using random")
        fw_iter = None
    
    losses = []
    for b in range(args.num_batches):
        if fw_iter is not None:
            batch = get_batch(fw_iter, tokenizer, args.batch_size, args.seq_len, device)
        else:
            vocab_size = tokenizer.vocab_size
            input_ids = torch.randint(0, vocab_size,
                                      (args.batch_size, args.seq_len), device=device)
            batch = {'input_ids': input_ids, 'labels': input_ids.clone()}
        out = model(**batch)
        loss = out.loss / args.num_batches
        loss.backward()
        losses.append(out.loss.item())
        del out, loss, batch
        torch.cuda.empty_cache()
    
    mean_loss = sum(losses) / len(losses)
    log(f"  Mean loss: {mean_loss:.3f}")
    
    # Param diagnostic
    param_names = [n for n, p in model.named_parameters()
                   if p.ndim == 2 and min(p.shape) >= 2 and p.requires_grad]
    total_2d = len(param_names)
    
    if total_2d > args.max_params:
        idx = [int(i * (total_2d - 1) / (args.max_params - 1))
               for i in range(args.max_params)]
        param_names = [param_names[i] for i in sorted(set(idx))]
    log(f"  Diagnosing {len(param_names)} / {total_2d} 2D params...")
    
    results_params = {}
    name_to_param = dict(model.named_parameters())
    t_diag = time.time()
    
    for i, name in enumerate(param_names):
        if (i + 1) % 30 == 0:
            log(f"    [{i+1}/{len(param_names)}] (elapsed {time.time()-t_diag:.0f}s)")
        p = name_to_param.get(name)
        if p is None or p.grad is None:
            continue
        grad_cpu = p.grad.detach().float().cpu()
        p.grad = None
        try:
            results_params[name] = diagnose_param(
                grad_cpu, ks=args.ks, alphas=args.alphas, c_min=args.cert_threshold
            )
        except Exception as e:
            log(f"    [{name}] failed: {e}")
        del grad_cpu
        if (i + 1) % 30 == 0:
            torch.cuda.empty_cache()
            gc.collect()
    log(f"  Diagnostic: {time.time()-t_diag:.0f}s")
    
    # Summary
    summary = {'mean_loss': mean_loss, 'n_params_total_billion': n_params_total/1e9,
               'n_2d_params': total_2d, 'n_diagnosed': len(results_params),
               'per_config': {}}
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
    
    spreads = [pd.get('row_norm_spread', 0) for pd in results_params.values()
               if 'row_norm_spread' in pd]
    if spreads:
        sps = sorted(spreads)
        summary['row_norm_spread'] = {
            'median': sps[len(sps)//2], 'max': max(spreads),
            'frac_gt_20': sum(1 for s in spreads if s>20) / len(spreads),
        }
    srs = [pd.get('stable_rank', 0) for pd in results_params.values()
           if pd.get('stable_rank', -1) > 0]
    if srs:
        srss = sorted(srs)
        summary['stable_rank'] = {'median': srss[len(srss)//2],
                                   'min': min(srs), 'max': max(srs)}
    
    log(f"  Total time for this model: {time.time()-t_total:.0f}s")
    
    # Free model
    del model, tokenizer, name_to_param, results_params
    torch.cuda.empty_cache()
    gc.collect()
    
    return {'summary': summary, 'model_name': model_name}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', type=str, nargs='+', required=True)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--seq_len', type=int, default=2048)
    parser.add_argument('--num_batches', type=int, default=4)
    parser.add_argument('--max_params', type=int, default=60)
    parser.add_argument('--ks', type=int, nargs='+', default=[8, 32])
    parser.add_argument('--alphas', type=float, nargs='+', default=[0.5, 0.25, 0.125])
    parser.add_argument('--cert_threshold', type=float, default=0.05)
    parser.add_argument('--dtype', type=str, default='bfloat16')
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()
    
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    all_results = {'config': vars(args), 'models': {}}
    
    t_start = time.time()
    for model_name in args.models:
        try:
            r = run_one_model(model_name, args, device)
            all_results['models'][model_name] = r
        except torch.cuda.OutOfMemoryError as e:
            log(f"\n!!! OOM on {model_name}: try reducing batch_size or seq_len")
            all_results['models'][model_name] = {'error': 'OOM', 'model_name': model_name}
            torch.cuda.empty_cache()
            gc.collect()
        except Exception as e:
            log(f"\n!!! Error on {model_name}: {e}")
            import traceback
            traceback.print_exc()
            all_results['models'][model_name] = {'error': str(e), 'model_name': model_name}
        
        # Save progressively (긴 진단이면 중간에 망해도 일부 보존)
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(all_results, f, indent=2)
    
    log(f"\nTotal time: {(time.time()-t_start)/60:.1f} min")
    
    # === Trend Analysis ===
    log("\n" + "="*90)
    log("MODEL SIZE TREND ANALYSIS")
    log("="*90)
    
    # Sort by model size
    valid = [(name, r['summary']) for name, r in all_results['models'].items()
             if 'error' not in r and 'summary' in r]
    valid.sort(key=lambda x: x[1].get('n_params_total_billion', 0))
    
    if not valid:
        log("No valid results")
        return
    
    # Table 1: model size, spread, stable rank
    log(f"\n{'Model':<35} {'Params':<10} {'Loss':<8} {'Spread':<10} {'SR med':<10}")
    log("-" * 75)
    for name, s in valid:
        params_b = s.get('n_params_total_billion', 0)
        loss = s.get('mean_loss', 0)
        spread = s.get('row_norm_spread', {}).get('median', 0)
        sr = s.get('stable_rank', {}).get('median', 0)
        log(f"{name[:34]:<35} {params_b:<10.2f} {loss:<8.3f} {spread:<10.1f} {sr:<10.1f}")
    
    # Table 2: pass rates per config across models
    log(f"\n{'='*90}")
    log("Pass rates by config (per model):")
    log(f"{'='*90}")
    
    for k in args.ks:
        for alpha in args.alphas:
            key = f"k{k}_a{alpha}"
            log(f"\n  Config {key}:")
            log(f"    {'Model':<35} {'pass%':<8} {'mean_τ':<10} {'mean_ω':<10} {'mean_cert':<10}")
            for name, s in valid:
                pc = s.get('per_config', {}).get(key)
                if pc is None:
                    log(f"    {name[:34]:<35} (no data)")
                    continue
                log(f"    {name[:34]:<35} {100*pc['pass_rate']:<8.1f} "
                    f"{pc['mean_tau']:<10.3f} {pc['mean_omega']:<10.3f} "
                    f"{pc['mean_cert']:+.3f}")
    
    # Trend conclusion
    log("\n" + "="*90)
    log("VERDICT - Does the model size trend support SC-Dion at scale?")
    log("="*90)
    
    # 가장 어려운 config (가장 작은 alpha)에서의 trend가 핵심
    hard_key = f"k{min(args.ks)}_a{min(args.alphas)}"
    log(f"\n[Hardest config: {hard_key}]")
    
    trend_rates = []
    trend_taus = []
    trend_spreads = []
    for name, s in valid:
        pc = s.get('per_config', {}).get(hard_key)
        params_b = s.get('n_params_total_billion', 0)
        spread = s.get('row_norm_spread', {}).get('median', 0)
        if pc:
            trend_rates.append((params_b, pc['pass_rate']))
            trend_taus.append((params_b, pc['mean_tau']))
            trend_spreads.append((params_b, spread))
            log(f"  {name[:30]:<32} ({params_b:.1f}B): "
                f"pass {100*pc['pass_rate']:.0f}%, τ={pc['mean_tau']:.3f}, spread={spread:.1f}x")
    
    if len(trend_rates) >= 2:
        # 단순 monotonic check
        rates_increase = all(trend_rates[i][1] <= trend_rates[i+1][1] + 0.05
                             for i in range(len(trend_rates)-1))
        taus_decrease = all(trend_taus[i][1] >= trend_taus[i+1][1] - 0.02
                            for i in range(len(trend_taus)-1))
        spreads_increase = all(trend_spreads[i][1] <= trend_spreads[i+1][1] + 2
                               for i in range(len(trend_spreads)-1))
        
        rate_range = max(r[1] for r in trend_rates) - min(r[1] for r in trend_rates)
        tau_range = max(t[1] for t in trend_taus) - min(t[1] for t in trend_taus)
        spread_range = max(s[1] for s in trend_spreads) - min(s[1] for s in trend_spreads)
        
        log(f"\nTrend summary (smallest → largest model):")
        log(f"  Pass rate range:  {100*min(r[1] for r in trend_rates):.0f}% → "
            f"{100*max(r[1] for r in trend_rates):.0f}% (Δ={100*rate_range:.1f}%)"
            f" {'monotonic' if rates_increase else 'non-monotonic'}")
        log(f"  Mean τ range:     {max(t[1] for t in trend_taus):.3f} → "
            f"{min(t[1] for t in trend_taus):.3f} (Δ={tau_range:.3f})"
            f" {'monotonic decrease' if taus_decrease else 'non-monotonic'}")
        log(f"  Spread range:     {min(s[1] for s in trend_spreads):.1f}x → "
            f"{max(s[1] for s in trend_spreads):.1f}x (Δ={spread_range:.1f}x)"
            f" {'monotonic increase' if spreads_increase else 'non-monotonic'}")
        
        # Final verdict
        signals = []
        if rates_increase and rate_range > 0.10:
            signals.append("✓ Pass rate INCREASES with model size")
        elif rate_range < 0.05:
            signals.append("~ Pass rate stable across sizes")
        else:
            signals.append("✗ Pass rate non-monotonic")
        
        if taus_decrease and tau_range > 0.03:
            signals.append("✓ τ DECREASES (more dominant subspace)")
        elif tau_range < 0.02:
            signals.append("~ τ stable")
        
        if spreads_increase and spread_range > 5:
            signals.append("✓ Spread INCREASES (more row-norm variation)")
        elif spread_range < 3:
            signals.append("~ Spread stable")
        
        log("\nSignal interpretation:")
        for s in signals:
            log(f"  {s}")
        
        positive_signals = sum(1 for s in signals if s.startswith("✓"))
        if positive_signals >= 2:
            log("\n🎯 STRONG SIGNAL: trend supports SC-Dion gains at larger scale")
        elif positive_signals == 1:
            log("\n📊 MIXED SIGNAL: some support but not definitive")
        else:
            log("\n🤷 NO CLEAR TREND: structural properties stable across scales")
            log("    (This is also informative — SC-Dion may work at any scale that has structure)")


if __name__ == '__main__':
    main()
