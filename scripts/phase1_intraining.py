"""
Phase 1D: 학습 중 진단.

Muon으로 짧은 학습 돌리면서 미리 정한 step에서 그래디언트 구조 측정.
- G_t (현재 그래디언트) 와 S_t = M_{t-1} + G_t (모멘텀 누적) 둘 다 측정
- 여러 c_min threshold로 통과율 평가
- Layer-wise 분포도 기록

이 진단의 목표: "사전학습된 모델 t=0에서 본 좋은 통과율이 학습 도중에도 유지되는가?"

사용:
    bash launch_phase1_intraining.sh
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

from optimizers import Muon, is_2d_param


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


def get_data_iter(tokenizer, batch_size: int, seq_len: int, device: str):
    """FineWeb streaming, packed into seq_len chunks."""
    from datasets import load_dataset
    ds = load_dataset('HuggingFaceFW/fineweb', name='sample-10BT',
                     split='train', streaming=True)
    ds = ds.shuffle(buffer_size=1000, seed=42)
    
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


@torch.no_grad()
def diagnose_tensor(S: torch.Tensor, ks: list, alphas: list,
                    cert_thresholds: list) -> dict:
    """
    한 텐서 진단. S는 진단 대상 (G_t 또는 모멘텀 누적 S_t).
    여러 c_min threshold에 대해 동시 평가.
    """
    from optimizers.sc_dion import (
        _randomized_subspace, _greedy_logdet_select, _certificate
    )
    
    transposed = S.size(0) > S.size(1)
    if transposed:
        S = S.t().contiguous()
    m, n = S.shape
    out = {'shape_oriented': (m, n)}
    
    row_norms = S.norm(dim=1)
    if row_norms.numel() > 0:
        mean_rn = row_norms.mean().item()
        max_rn = row_norms.max().item()
        out['row_norm_spread'] = max_rn / max(mean_rn, 1e-12)
    out['frob_norm'] = float(S.norm().item())
    
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
            # 여러 threshold에 대한 통과 여부 동시 기록
            entry = {
                'cert': float(cert), 'omega': float(omega), 'tau': float(tau),
            }
            for c_min in cert_thresholds:
                entry[f'pass_c{c_min}'] = bool(cert >= c_min)
            out['certificates'][f"k{k}_a{alpha}"] = entry
        del U_k
    return out


def diagnose_at_step(model, optimizer, step: int, args, device, ks, alphas,
                    cert_thresholds) -> dict:
    """
    현재 모델의 그래디언트와 모멘텀을 진단.
    optimizer의 state['momentum_buffer']에서 M_{t-1}을 가져옴.
    """
    log(f"\n  Diagnosing at step {step}...")
    t0 = time.time()
    
    # 2D 파라미터들 + 그들의 momentum buffer 매핑
    param_names = []
    name_to_p = {}
    for n, p in model.named_parameters():
        if p.ndim == 2 and min(p.shape) >= 2 and p.requires_grad and p.grad is not None:
            param_names.append(n)
            name_to_p[n] = p
    
    if len(param_names) > args.max_params:
        idx = [int(i * (len(param_names) - 1) / (args.max_params - 1))
               for i in range(args.max_params)]
        param_names = [param_names[i] for i in sorted(set(idx))]
    
    results_grad = {}
    results_momentum = {}
    
    for i, name in enumerate(param_names):
        if (i + 1) % 20 == 0:
            log(f"    [{i+1}/{len(param_names)}] (elapsed {time.time()-t0:.0f}s)")
        p = name_to_p[name]
        
        # G_t (그래디언트만)
        G = p.grad.detach().float().cpu()
        try:
            results_grad[name] = diagnose_tensor(G, ks, alphas, cert_thresholds)
        except Exception as e:
            log(f"    [{name}] G diagnostic failed: {e}")
        
        # S_t = M_{t-1} + G_t (모멘텀 누적 — 옵티마이저가 실제로 보는 것)
        state = optimizer.state.get(p, {})
        M = state.get('momentum_buffer')
        if M is not None:
            # Muon은 M <- mu*M + G; 그래서 새 S_t는 이미 M에 G가 누적됨.
            # 우리가 보고 싶은 건 M_t (이번 스텝 optimizer가 본 S_t)
            # → 정확한 S_t를 보려면 step 직전에 봐야 하지만 여기선 직후 측정.
            # 대신 mu*M + G를 새로 계산해서 진단 (학습 진행 전 시점의 S_t에 해당)
            mu = 0.95  # Muon default
            S_t = (mu * M + G.to(M.device, dtype=M.dtype)).float().cpu()
            try:
                results_momentum[name] = diagnose_tensor(S_t, ks, alphas, cert_thresholds)
            except Exception as e:
                log(f"    [{name}] S_t diagnostic failed: {e}")
            del S_t
        del G
        if (i + 1) % 20 == 0:
            gc.collect()
    
    log(f"  Diagnostic time: {time.time()-t0:.0f}s ({len(param_names)} params)")
    
    return {
        'step': step,
        'n_params_diagnosed': len(param_names),
        'grad': summarize(results_grad, ks, alphas, cert_thresholds),
        'momentum': summarize(results_momentum, ks, alphas, cert_thresholds),
    }


def summarize(per_param: dict, ks: list, alphas: list,
              cert_thresholds: list) -> dict:
    """파라미터별 결과를 요약."""
    if not per_param:
        return {}
    
    out = {'per_config': {}, 'n_params': len(per_param)}
    for k in ks:
        for alpha in alphas:
            key = f"k{k}_a{alpha}"
            tot = 0
            certs, taus, omegas = [], [], []
            pass_counts = {c: 0 for c in cert_thresholds}
            for pd in per_param.values():
                if key not in pd.get('certificates', {}):
                    continue
                cd = pd['certificates'][key]
                tot += 1
                certs.append(cd['cert']); taus.append(cd['tau']); omegas.append(cd['omega'])
                for c_min in cert_thresholds:
                    if cd.get(f'pass_c{c_min}', False):
                        pass_counts[c_min] += 1
            if tot > 0:
                out['per_config'][key] = {
                    'total': tot,
                    'mean_cert': sum(certs)/tot,
                    'mean_tau': sum(taus)/tot,
                    'mean_omega': sum(omegas)/tot,
                    'pass_rates': {str(c): pass_counts[c]/tot for c in cert_thresholds},
                }
    
    # Row-norm spread distribution
    spreads = [pd.get('row_norm_spread', 0) for pd in per_param.values()
               if 'row_norm_spread' in pd]
    if spreads:
        sps = sorted(spreads)
        out['spread_median'] = sps[len(sps)//2]
        out['spread_max'] = max(spreads)
    
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--seq_len', type=int, default=2048)
    parser.add_argument('--diagnose_steps', type=int, nargs='+',
                        default=[0, 10, 50, 200, 500, 1000])
    parser.add_argument('--max_steps', type=int, default=1000)
    parser.add_argument('--ks', type=int, nargs='+', default=[8, 32])
    parser.add_argument('--alphas', type=float, nargs='+', default=[0.5, 0.25, 0.125])
    parser.add_argument('--cert_thresholds', type=float, nargs='+',
                        default=[0.05, 0.1, 0.2, 0.3])
    parser.add_argument('--max_params', type=int, default=40,
                        help='시간 절약 위해 적게 (40개 × 6 step = 240회 진단)')
    parser.add_argument('--dtype', type=str, default='bfloat16')
    parser.add_argument('--lr', type=float, default=2e-4)  # 짧은 학습, conservative
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()
    
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info(0)
        log(f"GPU memory: {free/1e9:.1f} / {total/1e9:.1f} GB free")
    
    # 결과 진행 중 저장
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    model, tokenizer = load_model_and_tokenizer(args.model_name, args.dtype)
    model = model.to(device)
    model.train()
    
    # Optimizer: Muon만 (그래디언트 구조 측정용)
    matrix_params = [p for p in model.parameters() if is_2d_param(p) and p.requires_grad]
    other_params = [p for p in model.parameters() if not is_2d_param(p) and p.requires_grad]
    
    log(f"Matrix params: {len(matrix_params)}, Other: {len(other_params)}")
    
    muon = Muon(matrix_params, lr=args.lr, momentum=0.95, ns_steps=5)
    adamw = torch.optim.AdamW(other_params, lr=args.lr * 0.5, betas=(0.9, 0.95))
    optimizers = [muon, adamw]
    
    # Data
    log("Loading FineWeb data iterator...")
    data_iter = get_data_iter(tokenizer, args.batch_size, args.seq_len, device)
    
    # 진단 시점 (sorted)
    diagnose_steps = sorted(set(args.diagnose_steps))
    max_step = max(diagnose_steps[-1], args.max_steps)
    log(f"Will diagnose at steps: {diagnose_steps}")
    log(f"Total training steps: {max_step}")
    
    all_results = {
        'config': vars(args),
        'model_name': args.model_name,
        'diagnostics': [],  # list of per-step results
        'losses': [],  # full loss history
    }
    
    t_train_start = time.time()
    log(f"\n{'='*70}")
    log(f"=== Training with diagnostics ===")
    log(f"{'='*70}\n")
    
    diagnose_idx = 0
    for step in range(max_step + 1):  # +1 for final step
        # Forward / Backward
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = get_data_iter(tokenizer, args.batch_size, args.seq_len, device)
            batch = next(data_iter)
        
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        
        out = model(input_ids=batch, labels=batch.clone())
        loss = out.loss
        loss.backward()
        loss_val = loss.item()
        all_results['losses'].append({'step': step, 'loss': loss_val})
        
        del out, loss, batch
        
        # 진단 시점이면 (optimizer step 직전, 그래디언트 사용 가능)
        if diagnose_idx < len(diagnose_steps) and step == diagnose_steps[diagnose_idx]:
            elapsed_train = time.time() - t_train_start
            log(f"\n[step {step}] loss={loss_val:.4f}, train_elapsed={elapsed_train:.0f}s")
            
            diag = diagnose_at_step(model, muon, step, args, device,
                                    args.ks, args.alphas, args.cert_thresholds)
            diag['loss'] = loss_val
            all_results['diagnostics'].append(diag)
            
            # 중간 저장 (긴 실험 도중 망해도 보존)
            with open(output_path, 'w') as f:
                json.dump(all_results, f, indent=2)
            log(f"  Saved partial results to {output_path}")
            
            diagnose_idx += 1
        
        # Optimizer step
        if step < max_step:
            # grad clip
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            for opt in optimizers:
                opt.step()
        
        if (step + 1) % 50 == 0:
            log(f"  step {step+1}: loss={loss_val:.4f}")
        
        if diagnose_idx >= len(diagnose_steps):
            log(f"\nAll diagnostics done at step {step}, stopping training.")
            break
    
    log(f"\n{'='*70}")
    log(f"=== Final Comparison ===")
    log(f"{'='*70}")
    
    # 시점별 통과율 변화 표
    log(f"\n[Grad (G_t) — what selectors see if no momentum]:")
    log(f"  Config k8_a0.25, threshold 0.05:")
    log(f"    {'Step':<8} {'Loss':<8} {'Pass%':<8} {'τ':<8} {'ω':<8}")
    for d in all_results['diagnostics']:
        g = d.get('grad', {})
        pc = g.get('per_config', {}).get('k8_a0.25', {})
        if pc:
            log(f"    {d['step']:<8} {d['loss']:<8.3f} "
                f"{100*pc['pass_rates'].get('0.05', 0):<8.1f} "
                f"{pc['mean_tau']:<8.3f} {pc['mean_omega']:<8.3f}")
    
    log(f"\n[Momentum-accumulated (S_t = μM + G) — what SC-Dion actually selects on]:")
    log(f"  Config k8_a0.25, threshold 0.05:")
    log(f"    {'Step':<8} {'Loss':<8} {'Pass%':<8} {'τ':<8} {'ω':<8}")
    for d in all_results['diagnostics']:
        s = d.get('momentum', {})
        pc = s.get('per_config', {}).get('k8_a0.25', {})
        if pc:
            log(f"    {d['step']:<8} {d['loss']:<8.3f} "
                f"{100*pc['pass_rates'].get('0.05', 0):<8.1f} "
                f"{pc['mean_tau']:<8.3f} {pc['mean_omega']:<8.3f}")
    
    # Threshold sensitivity (마지막 step에 대해)
    if all_results['diagnostics']:
        last = all_results['diagnostics'][-1]
        log(f"\n[Threshold sensitivity at step {last['step']} (S_t)]:")
        for k in args.ks:
            for alpha in args.alphas:
                key = f"k{k}_a{alpha}"
                pc = last.get('momentum', {}).get('per_config', {}).get(key)
                if not pc:
                    continue
                rates = pc['pass_rates']
                rates_str = ", ".join(f"c={c}:{100*rates[str(c)]:.0f}%"
                                     for c in args.cert_thresholds)
                log(f"    {key}: {rates_str}")
    
    # Verdict
    log(f"\n{'='*70}")
    log(f"VERDICT")
    log(f"{'='*70}")
    
    if len(all_results['diagnostics']) < 2:
        log("Not enough diagnostic points for trend analysis")
        return
    
    # 핵심 질문: 학습 도중 통과율이 유지되는가?
    first_d = all_results['diagnostics'][0]
    last_d = all_results['diagnostics'][-1]
    
    for source_key, source_name in [('momentum', 'S_t (momentum-accumulated)'),
                                      ('grad', 'G_t (gradient only)')]:
        log(f"\n[{source_name}]")
        log(f"  {'Config':<15} {'Step ' + str(first_d['step']):<15} "
            f"{'Step ' + str(last_d['step']):<15} {'Δ':<10}")
        for k in args.ks:
            for alpha in args.alphas:
                key = f"k{k}_a{alpha}"
                pc_first = first_d.get(source_key, {}).get('per_config', {}).get(key)
                pc_last = last_d.get(source_key, {}).get('per_config', {}).get(key)
                if pc_first and pc_last:
                    r_first = pc_first['pass_rates'].get('0.05', 0)
                    r_last = pc_last['pass_rates'].get('0.05', 0)
                    delta = r_last - r_first
                    marker = "✅" if r_last >= 0.7 else ("⚠️" if r_last >= 0.3 else "❌")
                    log(f"  {key:<15} {100*r_first:>7.1f}%        "
                        f"{100*r_last:>7.1f}%        "
                        f"{100*delta:+7.1f}%  {marker}")
    
    # Most important diagnostic: did pass rate collapse during training?
    log("\nKey question: Does SC-Dion pass rate hold during training?")
    
    # 가장 어려운 config에서의 trajectory
    hard_key = f"k{min(args.ks)}_a{min(args.alphas)}"
    s_traj = [(d['step'], d.get('momentum', {}).get('per_config', {}).get(hard_key, {}).get('pass_rates', {}).get('0.05', None))
              for d in all_results['diagnostics']]
    s_traj = [(s, r) for s, r in s_traj if r is not None]
    
    if len(s_traj) >= 2:
        first_rate = s_traj[0][1]
        last_rate = s_traj[-1][1]
        min_rate = min(r for _, r in s_traj)
        
        log(f"\n  [Hardest config {hard_key}, S_t, c_min=0.05]:")
        log(f"    Initial:  {100*first_rate:.0f}%")
        log(f"    Minimum:  {100*min_rate:.0f}%")
        log(f"    Final:    {100*last_rate:.0f}%")
        
        if min_rate >= 0.7:
            log("\n  ✅ SUSTAINED: Pass rate stays high throughout training")
            log("     → SC-Dion should work in real training")
            log("     → 다음 단계: GPU SC-Dion 구현 + Phase 2 학습 비교")
        elif min_rate >= 0.3:
            log("\n  ⚠️ DEGRADED: Pass rate dropped but not collapsed")
            log("     → SC-Dion will partially fallback to Muon during training")
            log("     → Wall-clock gains will be smaller than t=0 suggests")
        else:
            log("\n  ❌ COLLAPSED: Pass rate dropped significantly during training")
            log("     → t=0 diagnostic was misleading; SC-Dion mostly falls back")
            log("     → Wall-clock gains unlikely in pretraining setting")
            log("     → 대안: fine-tuning 시나리오나 다른 architecture 고려")


if __name__ == '__main__':
    main()
