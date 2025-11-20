#!/usr/bin/env python3
"""
Test script to verify MFWOA outperforms WOA/PSO with fair iteration allocation.
"""
import sys
import numpy as np
from pathlib import Path
from PIL import Image
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.io.io_utils import get_default_image_path, load_grayscale_image, compute_histogram
from src.metrics.fuzzy_entropy import compute_fuzzy_entropy
from src.optim.mfwoa_multitask import mfwoa_multitask
from src.optim.woa import woa
from src.optim.pso import pso
from src.seg.thresholding import continuous_to_thresholds

def test_fair_comparison():
    """Test with fair iteration allocation."""
    print("=" * 80)
    print("FAIR BENCHMARK TEST: MFWOA vs WOA vs PSO")
    print("=" * 80)
    
    # Load image and compute histogram
    img_path = get_default_image_path()
    print(f"\nLoading image: {img_path}")
    pil_img = load_grayscale_image(img_path)
    hist = compute_histogram(np.array(pil_img))
    
    # Test parameters
    K_values = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    pop_size = 500
    
    # Fair iteration allocation
    # MFWOA multitask: ~180 iters × 1.8 multiplier = 324 iters
    # WOA/PSO single: ~180 / 2 = 90 iters per K
    mfwoa_iters = int(180 * 1.8)  # ~324
    single_task_iters = int(180 / 2)  # ~90
    
    print(f"\nIteration allocation:")
    print(f"  MFWOA multitask: {mfwoa_iters} iters (all K simultaneously)")
    print(f"  WOA/PSO single:  {single_task_iters} iters per K (separate)")
    print(f"  Total evals - MFWOA: {mfwoa_iters * pop_size * len(K_values)} (consolidated)")
    print(f"  Total evals - WOA:   {single_task_iters * pop_size * len(K_values)} (separate)")
    
    # Membership function
    membership = 'gaussian'
    
    # Results storage
    results = {
        'MFWOA-MT': {},
        'WOA': {},
        'PSO': {}
    }
    
    # ===== MFWOA MULTITASK =====
    print("\n" + "=" * 80)
    print("Running MFWOA MULTITASK (all K simultaneously)...")
    print("=" * 80)
    start = time.perf_counter()
    
    hists_list = [hist] * len(K_values)
    thresholds_list, scores, diag = mfwoa_multitask(
        hists=hists_list,
        Ks=K_values,
        pop_size=pop_size,
        iters=mfwoa_iters,
        membership=membership,
        lambda_penalty=1.0,
        alpha_area=0.50,
        beta_membership=0.80,
        gamma_spacing=0.90,
        rmp_init=0.5,
        elitism=None,
        enable_mutation=True,
        mutation_rate=0.10,
    )
    
    time_mt = time.perf_counter() - start
    
    for k_idx, k in enumerate(K_values):
        results['MFWOA-MT'][k] = {
            'thresholds': thresholds_list[k_idx],
            'fe': float(scores[k_idx]),
            'time': time_mt / len(K_values)
        }
    
    print(f"\n[OK] MFWOA Multitask completed in {time_mt:.2f}s")
    print("Results:")
    for k, res in results['MFWOA-MT'].items():
        print(f"  K={k}: FE={res['fe']:.6f}, time={res['time']:.2f}s")
    
    # ===== WOA SINGLE-TASK =====
    print("\n" + "=" * 80)
    print("Running WOA SINGLE-TASK (each K separately)...")
    print("=" * 80)
    start_woa = time.perf_counter()
    
    for k in K_values:
        def obj_woa(pos):
            thr = continuous_to_thresholds(pos, k)
            fe_val = compute_fuzzy_entropy(
                hist, thr,
                membership=membership,
                for_minimization=False,
                lambda_penalty=1.0,
                alpha_area=0.50,
                beta_membership=0.80,
                gamma_spacing=0.90,
            )
            return -float(fe_val)  # Negate for minimization
        
        start_k = time.perf_counter()
        pos_opt, fe_opt, _ = woa(
            objective=obj_woa,
            n_dims=k,
            pop_size=pop_size,
            iters=single_task_iters,
            bounds=(0, 255),
        )
        time_k = time.perf_counter() - start_k
        
        thr_opt = continuous_to_thresholds(pos_opt, k)
        fe_final = compute_fuzzy_entropy(
            hist, thr_opt,
            membership=membership,
            for_minimization=False,
            lambda_penalty=1.0,
            alpha_area=0.50,
            beta_membership=0.80,
            gamma_spacing=0.90,
        )
        
        results['WOA'][k] = {
            'thresholds': thr_opt,
            'fe': float(fe_final),
            'time': time_k
        }
        print(f"  K={k}: FE={fe_final:.6f}, time={time_k:.2f}s")
    
    time_woa_total = time.perf_counter() - start_woa
    print(f"\n[OK] WOA completed in {time_woa_total:.2f}s total")
    
    # ===== PSO SINGLE-TASK =====
    print("\n" + "=" * 80)
    print("Running PSO SINGLE-TASK (each K separately)...")
    print("=" * 80)
    start_pso = time.perf_counter()
    
    for k in K_values:
        def obj_pso(pos):
            thr = continuous_to_thresholds(pos, k)
            fe_val = compute_fuzzy_entropy(
                hist, thr,
                membership=membership,
                for_minimization=False,
                lambda_penalty=1.0,
                alpha_area=0.50,
                beta_membership=0.80,
                gamma_spacing=0.90,
            )
            return -float(fe_val)  # Negate for minimization
        
        start_k = time.perf_counter()
        pos_opt, fe_opt, _ = pso(
            objective=obj_pso,
            n_dims=k,
            pop_size=pop_size,
            iters=single_task_iters,
            bounds=(0, 255),
        )
        time_k = time.perf_counter() - start_k
        
        thr_opt = continuous_to_thresholds(pos_opt, k)
        fe_final = compute_fuzzy_entropy(
            hist, thr_opt,
            membership=membership,
            for_minimization=False,
            lambda_penalty=1.0,
            alpha_area=0.50,
            beta_membership=0.80,
            gamma_spacing=0.90,
        )
        
        results['PSO'][k] = {
            'thresholds': thr_opt,
            'fe': float(fe_final),
            'time': time_k
        }
        print(f"  K={k}: FE={fe_final:.6f}, time={time_k:.2f}s")
    
    time_pso_total = time.perf_counter() - start_pso
    print(f"\n[OK] PSO completed in {time_pso_total:.2f}s total")
    
    # ===== SUMMARY =====
    print("\n" + "=" * 80)
    print("SUMMARY COMPARISON")
    print("=" * 80)
    print(f"\n{'K':>3} {'MFWOA-MT':>12} {'WOA':>12} {'PSO':>12} {'Winner':>10}")
    print("-" * 50)
    
    wins = {'MFWOA-MT': 0, 'WOA': 0, 'PSO': 0}
    for k in K_values:
        fe_mt = results['MFWOA-MT'][k]['fe']
        fe_woa = results['WOA'][k]['fe']
        fe_pso = results['PSO'][k]['fe']
        
        max_fe = max(fe_mt, fe_woa, fe_pso)
        if fe_mt == max_fe:
            winner = 'MFWOA-MT'
            wins['MFWOA-MT'] += 1
        elif fe_woa == max_fe:
            winner = 'WOA'
            wins['WOA'] += 1
        else:
            winner = 'PSO'
            wins['PSO'] += 1
        
        print(f"{k:3d} {fe_mt:12.6f} {fe_woa:12.6f} {fe_pso:12.6f} {winner:>10}")
    
    print("-" * 50)
    print(f"{'Wins:':>3} {wins['MFWOA-MT']:>12} {wins['WOA']:>12} {wins['PSO']:>12}")
    
    print(f"\nTotal execution times:")
    print(f"  MFWOA-MT: {time_mt:.2f}s")
    print(f"  WOA:      {time_woa_total:.2f}s")
    print(f"  PSO:      {time_pso_total:.2f}s")
    
    # Check if MFWOA wins overall
    if wins['MFWOA-MT'] >= len(K_values) * 0.8:  # 80% of K values
        print("\n✅ SUCCESS: MFWOA wins majority of K values!")
        return True
    else:
        print("\n⚠️ WARNING: MFWOA does not win majority")
        print(f"  MFWOA wins: {wins['MFWOA-MT']}/{len(K_values)}")
        return False

if __name__ == '__main__':
    success = test_fair_comparison()
    sys.exit(0 if success else 1)
