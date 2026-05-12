"""
Phase 1B: 진단 검증 (Validation).

이전 진단 결과가 너무 좋게 나와서, 데이터 로딩이 진짜였는지 확인.

핵심 변경:
1. 데이터 로딩을 EXPLICITLY 검증 (성공/실패 명확히 보고)
2. 데이터 샘플을 출력 (사용자가 직접 확인 가능)
3. 같은 모델에 대해 여러 데이터 소스로 진단:
   - random_tokens: 의도적 비교군 (의미 없는 입력)
   - fineweb: 진짜 사전학습 데이터
   - wikipedia: 다른 진짜 데이터
4. 결과 비교표 출력 — 데이터 소스에 따라 다르면 이전 결과 의심 가능

사용:
    bash launch_phase1_validate.sh
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


def load_dataset_explicit(source: str):
    """데이터셋 로딩 시도. 성공 여부와 첫 샘플 명확히 보고."""
    log(f"\n--- Loading dataset: {source} ---")
    
    if source == 'random_tokens':
        log("[random_tokens] OK (no dataset needed)")
        return 'random'
    
    from datasets import load_dataset
    
    try:
        if source == 'fineweb':
            ds = load_dataset('HuggingFaceFW/fineweb', name='sample-10BT',
                             split='train', streaming=True)
            it = iter(ds)
        elif source == 'wikipedia':
            ds = load_dataset('wikimedia/wikipedia', '20231101.en',
                             split='train', streaming=True)
            it = iter(ds)
        elif source == 'c4':
            ds = load_dataset('allenai/c4', 'en',
                             split='train', streaming=True)
            it = iter(ds)
        else:
            log(f"[{source}] Unknown source")
            return None
        
        # 첫 샘플 시도
        sample = next(it)
        text_key = 'text' if 'text' in sample else 'content'
        sample_text = sample.get(text_key, '')[:300]
        log(f"[{source}] LOADED. First sample preview:")
        log(f"  >>> {sample_text!r}")
        return it
        
    except Exception as e:
        log(f"[{source}] FAILED: {type(e).__name__}: {e}")
        log(f"[{source}] (would have fallen back to random tokens)")
        return None


def get_batch(source_iter, tokenizer, batch_size: int, seq_len: int, device: str):
    """진짜 데이터로 batch 만들기. fallback이 발생하면 명시적으로 알림."""
    if source_iter == 'random' or source_iter is None:
        vocab_size = tokenizer.vocab_size
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        return {'input_ids': input_ids, 'labels': input_ids.clone()}, 'random'
    
    texts = []
    sample_for_log = None
    try:
        while len(texts) < batch_size:
            ex = next(source_iter)
            text = ex.get('text', '') or ex.get('content', '')
            if not text or len(text) < 50:  # 너무 짧은 건 skip
                continue
            texts.append(text)
            if sample_for_log is None:
                sample_for_log = text[:200]
    except StopIteration:
        if not texts:
            log("  WARN: dataset exhausted, falling back to random")
            return get_batch('random', tokenizer, batch_size, seq_len, device)
    
    enc = tokenizer(texts, max_length=seq_len, truncation=True,
                    padding='max_length', return_tensors='pt').to(device)
    labels = enc['input_ids'].clone()
    labels[enc['attention_mask'] == 0] = -100
    return ({'input_ids': enc['input_ids'],
             'attention_mask': enc['attention_mask'],
             'labels': labels}, sample_for_log)


@torch.no_grad()
def diagnose_param(grad_cpu: torch.Tensor, ks: list, alphas: list,
                   c_min: float = 0.05) -> dict:
    """한 파라미터 진단. CPU float32 grad 입력."""
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
            out['certificates'][f"k{k}_a{alpha}"] = {
                'cert': float(cert), 'omega': float(omega), 'tau': float(tau),
                'k_effective': k_eff, 'num_selected': num_select,
                'would_pass': bool(cert >= c_min),
            }
        del U_k
    return out


def run_one_source(model, tokenizer, source: str, args, device) -> dict:
    """한 데이터 소스에 대해 진단."""
    log(f"\n{'='*70}")
    log(f"=== Source: {source} ===")
    log(f"{'='*70}")
    
    source_iter = load_dataset_explicit(source)
    
    # 모든 grad 초기화
    for p in model.parameters():
        if p.grad is not None:
            p.grad = None
    
    log(f"\nComputing gradients on {args.num_batches} batches "
        f"(bs={args.batch_size}, seq={args.seq_len})...")
    
    actual_data_type = None  # 실제 사용된 데이터 종류 추적
    losses = []
    for b in range(args.num_batches):
        batch, actual = get_batch(source_iter, tokenizer,
                                   args.batch_size, args.seq_len, device)
        actual_data_type = actual or actual_data_type
        out = model(**batch)
        loss = out.loss / args.num_batches
        loss.backward()
        losses.append(out.loss.item())
        log(f"  batch {b+1}/{args.num_batches}: loss={out.loss.item():.4f}")
        del out, loss, batch
        torch.cuda.empty_cache()
    
    log(f"  Mean loss: {sum(losses)/len(losses):.3f}")
    log(f"  Actual data used: {'random tokens' if actual_data_type == 'random' else 'real text'}")
    if actual_data_type != 'random' and actual_data_type is not None:
        log(f"  Sample text preview: {actual_data_type[:150]!r}")
    
    # Per-parameter 진단
    log(f"\nRunning diagnostics on params...")
    param_names = [n for n, p in model.named_parameters()
                   if p.ndim == 2 and min(p.shape) >= 2 and p.requires_grad]
    
    # Sample if too many
    if len(param_names) > args.max_params:
        idx = [int(i * (len(param_names) - 1) / (args.max_params - 1))
               for i in range(args.max_params)]
        param_names = [param_names[i] for i in sorted(set(idx))]
    
    results_params = {}
    name_to_param = dict(model.named_parameters())
    t0 = time.time()
    
    for i, name in enumerate(param_names):
        if (i + 1) % 30 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i+1) * (len(param_names) - i - 1)
            log(f"    [{i+1}/{len(param_names)}] (elapsed {elapsed:.0f}s, ETA {eta:.0f}s)")
        
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
    
    elapsed = time.time() - t0
    log(f"  Diagnostic time: {elapsed:.0f}s ({elapsed/len(param_names):.1f}s/param)")
    
    # Summary for this source
    summary = {'losses': losses, 'mean_loss': sum(losses)/len(losses),
               'actual_data_type': 'random' if actual_data_type == 'random' else 'real',
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
                    cv.append(cd['cert'])
                    tv.append(cd['tau'])
                    ov.append(cd['omega'])
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
    
    return {'summary': summary, 'params': results_params}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--seq_len', type=int, default=2048)
    parser.add_argument('--num_batches', type=int, default=8,
                        help='많을수록 신뢰성 높음 (이전엔 4, 이번엔 8 권장)')
    parser.add_argument('--sources', type=str, nargs='+',
                        default=['random_tokens', 'fineweb', 'wikipedia'],
                        help='비교할 데이터 소스들')
    parser.add_argument('--ks', type=int, nargs='+', default=[8, 32])
    parser.add_argument('--alphas', type=float, nargs='+',
                        default=[0.5, 0.25, 0.125])
    parser.add_argument('--cert_threshold', type=float, default=0.05)
    parser.add_argument('--dtype', type=str, default='bfloat16')
    parser.add_argument('--max_params', type=int, default=60,
                        help='진단할 파라미터 수 (적게: 빠르지만 통계 약함)')
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()
    
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info(0)
        log(f"GPU memory: {free/1e9:.1f} / {total/1e9:.1f} GB free")
    
    model, tokenizer = load_model_and_tokenizer(args.model_name, args.dtype)
    model = model.to(device)
    model.train()
    
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info(0)
        log(f"After model: {free/1e9:.1f} / {total/1e9:.1f} GB free")
    
    log(f"\nTotal 2D params: "
        f"{sum(1 for _, p in model.named_parameters() if p.ndim==2 and min(p.shape)>=2)}")
    log(f"Will diagnose at most {args.max_params} params per source\n")
    
    # 각 소스별로 진단
    all_results = {'config': vars(args), 'model_name': args.model_name,
                   'sources': {}}
    
    for source in args.sources:
        try:
            result = run_one_source(model, tokenizer, source, args, device)
            all_results['sources'][source] = result
        except Exception as e:
            log(f"\n!!! Source {source} failed entirely: {e}")
            import traceback
            traceback.print_exc()
            all_results['sources'][source] = {'error': str(e)}
        
        # 다음 소스 전에 grad 완전 초기화
        for p in model.parameters():
            if p.grad is not None:
                p.grad = None
        torch.cuda.empty_cache()
        gc.collect()
    
    # === Comparative analysis ===
    log("\n" + "=" * 80)
    log("COMPARATIVE ANALYSIS (the moment of truth)")
    log("=" * 80)
    
    log(f"\n{'Source':<20} {'Data':<10} {'Loss':<8} {'Spread':<10} {'SR med':<10}")
    log("-" * 60)
    for source, result in all_results['sources'].items():
        if 'error' in result:
            log(f"{source:<20} FAILED")
            continue
        s = result['summary']
        actual = s.get('actual_data_type', '?')
        loss = s.get('mean_loss', 0)
        spread = s.get('row_norm_spread', {}).get('median', 0)
        sr = s.get('stable_rank', {}).get('median', 0)
        log(f"{source:<20} {actual:<10} {loss:<8.3f} {spread:<10.1f} {sr:<10.1f}")
    
    log(f"\n{'='*80}")
    log("Pass rates across sources (the key comparison):")
    log(f"{'='*80}")
    
    # Pass rate 테이블
    for k in args.ks:
        for alpha in args.alphas:
            key = f"k{k}_a{alpha}"
            log(f"\n  Config {key}:")
            log(f"    {'Source':<20} {'pass_rate':<12} {'mean_τ':<10} {'mean_ω':<10} {'mean_cert':<12}")
            for source, result in all_results['sources'].items():
                if 'error' in result:
                    continue
                pc = result['summary']['per_config'].get(key)
                if pc is None:
                    log(f"    {source:<20} (no data)")
                    continue
                log(f"    {source:<20} {100*pc['pass_rate']:>6.1f}%      "
                    f"{pc['mean_tau']:<10.3f} {pc['mean_omega']:<10.3f} "
                    f"{pc['mean_cert']:+.3f}")
    
    # === Verdict ===
    log("\n" + "=" * 80)
    log("VERDICT")
    log("=" * 80)
    
    # Cross-source consistency 체크
    sources_with_data = [s for s, r in all_results['sources'].items()
                         if 'error' not in r and r['summary'].get('per_config')]
    
    if len(sources_with_data) < 2:
        log("\n⚠️ 충분한 비교 데이터 없음.")
    else:
        # 같은 (k, α)에 대해 다른 소스의 pass rate 차이 보기
        max_diff = 0
        biggest_diff_key = None
        for k in args.ks:
            for alpha in args.alphas:
                key = f"k{k}_a{alpha}"
                rates = []
                for s in sources_with_data:
                    pc = all_results['sources'][s]['summary']['per_config'].get(key)
                    if pc is not None:
                        rates.append((s, pc['pass_rate']))
                if len(rates) >= 2:
                    diff = max(r[1] for r in rates) - min(r[1] for r in rates)
                    if diff > max_diff:
                        max_diff = diff
                        biggest_diff_key = key
        
        log(f"\n소스 간 통과율 최대 차이: {100*max_diff:.1f}% (at {biggest_diff_key})")
        
        if max_diff < 0.15:
            log("\n✅ ROBUST: 데이터 소스에 관계없이 일관된 결과")
            log("    → 진단 결과를 신뢰할 수 있음")
            log("    → Phase 2 학습 비교 진행 가능")
        elif max_diff < 0.30:
            log("\n⚠️ MODERATE: 데이터 소스별로 어느 정도 차이 있음")
            log("    → 결과를 부분적으로 신뢰. real data만 사용 권장")
        else:
            log("\n❌ INCONSISTENT: 데이터에 따라 결과가 크게 다름")
            log("    → 이전 진단이 random tokens였을 가능성 큼")
            log("    → 사전학습된 모델에 random input 흘리면 인공적 저랭크가 됨")
            log("    → Phase 2는 real-data 결과만 가지고 결정해야 함")
        
        # Random vs Real 직접 비교
        if 'random_tokens' in sources_with_data:
            real_sources = [s for s in sources_with_data if s != 'random_tokens']
            if real_sources:
                log(f"\n[Random tokens vs Real data 직접 비교]")
                for k in args.ks:
                    for alpha in args.alphas:
                        key = f"k{k}_a{alpha}"
                        rnd = all_results['sources']['random_tokens']['summary']['per_config'].get(key)
                        real_avgs = []
                        for rs in real_sources:
                            pc = all_results['sources'][rs]['summary']['per_config'].get(key)
                            if pc:
                                real_avgs.append(pc['pass_rate'])
                        if rnd and real_avgs:
                            real_avg = sum(real_avgs) / len(real_avgs)
                            log(f"    {key}: random={100*rnd['pass_rate']:.1f}%, "
                                f"real_avg={100*real_avg:.1f}%, "
                                f"diff={100*(rnd['pass_rate']-real_avg):+.1f}%")
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # 큰 'params' 부분은 별도 파일로 저장 (메인 결과는 summary만)
    summary_only = {
        'config': all_results['config'],
        'model_name': all_results['model_name'],
        'sources': {s: {'summary': r['summary']} if 'error' not in r else r
                    for s, r in all_results['sources'].items()},
    }
    with open(output_path, 'w') as f:
        json.dump(summary_only, f, indent=2)
    log(f"\nSaved summary to {output_path}")


if __name__ == '__main__':
    main()
