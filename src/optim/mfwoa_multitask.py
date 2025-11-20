"""Simplified Multifactorial WOA (MFWOA) for multilevel thresholding.

This implementation is a pragmatic, testable variant:
- Unified population encoding length = max(Ks)
- Each individual has a skill factor (assigned task)
- Adaptive rmp scalar (global) updated by simple success metric
- Knowledge transfer via mixing with leader of other task when cross-task chosen
- Evaluations performed on the individual's assigned task only (keeps functions pure where possible)
"""
from __future__ import annotations

import numpy as np
from typing import List, Sequence, Tuple

from src.metrics.fuzzy_entropy import compute_fuzzy_entropy
from src.seg.utils import enforce_threshold_constraints

EPS = 1e-12


def continuous_to_thresholds(pos: np.ndarray, K: int) -> List[int]:
    """Convert continuous position to discrete thresholds using standard constraint enforcement.
    
    Hàm chuyển đổi vị trí liên tục (từ optimizer) thành danh sách ngưỡng rời rạc (int).
    Dùng enforce_threshold_constraints để đảm bảo consistency với other optimizers.
    
    Args:
        pos: Continuous position vector (numpy array)
        K: Number of thresholds (số ngưỡng cần lấy từ vị trí)
    
    Returns:
        List of integer thresholds (đã constraint, sorted, enforced gap)
    """
    # Lấy K phần tử đầu từ vị trí
    arr = pos[:K]
    # Sử dụng enforce_threshold_constraints từ seg/utils để đảm bảo:
    # - Nằm trong [1, 254]
    # - Sắp xếp tăng dần
    # - Khoảng cách tối thiểu giữa các ngưỡng
    # - Không trùng
    constrained = enforce_threshold_constraints(np.round(arr).astype(np.int32))
    # Chuyển thành list int
    return [int(x) for x in constrained]


def mfwoa_multitask(
    hists: Sequence[np.ndarray],
    Ks: Sequence[int],
    pop_size: int = 100,
    iters: int = 500,
    rng: np.random.Generator = None,
    rmp_init: float = 0.3,
    pop_per_task: int = None,
    elitism: int = 5,  # INCREASED: from 3 to 5 for stronger elite preservation
    rmp_schedule: Sequence[tuple] = None,
    membership: str = "triangular",
    lambda_penalty: float = 1.0,
    alpha_area: float = 0.10,
    beta_membership: float = 0.10,
    gamma_spacing: float = 0.20,
) -> Tuple[List[List[int]], List[float], dict]:
    """Run simplified MFWOA for multiple tasks simultaneously.
    
    ===== MULTITASK MFWOA: Knowledge Transfer + Adaptive RMP =====
    
    Thuật toán:
    1. Khởi tạo quần thể gộp (pop_size cá thể) được chia cho T tasks
    2. Mỗi cá thể được gán skill_factor ∈ {0..T-1} (assigned task)
    3. Mỗi iteration: cập nhật cá thể dựa vào WOA mechanics + knowledge transfer (cross-task)
    4. rmp (cross-task rate) được adjust dựa trên thành công
    5. Trả về best_thresholds cho mỗi task

    Args:
        hists: sequence of histograms (one per task). For same image multiple Ks, pass same hist repeated.
        Ks: sequence of ints number of thresholds per task.
        pop_size: Total population size (chia cho T tasks)
        iters: Number of iterations
        rng: Random number generator (seed cho reproducibility)
        rmp_init: Initial cross-task rate (0.3 = 30% cross-task updates)
        pop_per_task: Nếu set, tính pop_size = pop_per_task * T (override pop_size param)
        elitism: Số elite individuals mỗi task được preserve
        rmp_schedule: Optional schedule [(start_frac, end_frac, rmp_value), ...]
        membership: Membership function type cho compute_fuzzy_entropy (triangular/gaussian/s)
        lambda_penalty: Weighting factor cho penalizer
        alpha_area: Weight cho P_A penalty (cân bằng kích thước lớp)
        beta_membership: Weight cho P_μ penalty (tránh concentrated membership)
        gamma_spacing: Weight cho P_spacing penalty (enforce even distribution)
    
    Returns:
        (best_thresholds_per_task, best_scores_per_task, diagnostics)
    """
    if rng is None:
        rng = np.random.default_rng(123)
    T = len(Ks)
    maxK = max(Ks)
    # allow specifying population per task; if provided, scale total pop
    if pop_per_task is not None:
        pop_size = int(pop_per_task) * T
    
    # ===== OBJECTIVES PER TASK: Sử dụng same penalties như app.py =====
    # compute_fuzzy_entropy(hist, thresholds, membership, for_minimization=False,
    #                       lambda_penalty, alpha_area, beta_membership, gamma_spacing)
    # 
    # ⚠️  IMPORTANT: MFWOA is a MINIMIZER (like WOA/PSO), so objectives should return -FE
    # to be consistent with other optimizers. WOA negates FE before passing to minimizer:
    #   objective(pos) returns -FE, and WOA minimizes -FE to maximize FE.
    # We do the same here for consistency.
    def make_obj(hist, K):
        def obj_task(pos):
            """Objective function: minimize -FE (equivalent to maximizing FE) with penalties"""
            thr = continuous_to_thresholds(pos, K)
            # Tính FE với penalties (same formula như app.py)
            # for_minimization=False: sẽ trả về FE (cao hơn tốt hơn)
            fe_val = compute_fuzzy_entropy(
                hist, thr, 
                membership=membership,
                for_minimization=False,
                lambda_penalty=lambda_penalty,
                alpha_area=alpha_area,
                beta_membership=beta_membership,
                gamma_spacing=gamma_spacing,
                delta_compactness=0.0  # Không dùng compactness, rely on gamma_spacing
            )
            # Negate to convert maximization (FE) to minimization (-FE)
            # This matches WOA's approach: minimize -FE instead of maximize FE
            return -float(fe_val)
        return obj_task
    
    objectives = [make_obj(h, K) for h, K in zip(hists, Ks)]

    # initialize population and skill factors
    pop = rng.uniform(0.0, 255.0, size=(pop_size, maxK))
    skill_factors = rng.integers(low=0, high=T, size=pop_size)
    fitness = np.full(pop_size, np.inf)  # ⚠️ CHANGED: initialize to +inf for MINIMIZATION
    # evaluate initial
    for i in range(pop_size):
        sf = int(skill_factors[i])
        fitness[i] = objectives[sf](pop[i])
    # best per task
    best_pos = [None] * T
    best_score = [np.inf] * T  # ⚠️ CHANGED: initialize to +inf for MINIMIZATION
    # Ensure at least one individual per task: if any task has zero, assign by round-robin
    counts = np.bincount(skill_factors, minlength=T)
    if np.any(counts == 0):
        # assign indices round-robin to ensure at least one per task
        for i in range(pop_size):
            skill_factors[i] = i % T
        # ===== RECALCULATE FITNESS AFTER REALLOCATION =====
        # Khi thay đổi skill_factors, PHẢI tính lại fitness cho từng cá thể theo assigned task
        # Không làm bước này, cá thể sẽ có fitness từ task cũ (sai!)
        for i in range(pop_size):
            sf = int(skill_factors[i])
            fitness[i] = objectives[sf](pop[i])

    for t in range(T):
        mask = (skill_factors == t)
        if mask.any():
            # ⚠️ CHANGED: compute MINIMUM instead of MAXIMUM (argmin instead of argmax)
            # Because objectives now return -FE (negated) and we minimize
            masked = np.where(mask, fitness, np.inf)
            idx = int(np.argmin(masked))
            best_pos[t] = pop[idx].copy()
            best_score[t] = float(fitness[idx])

    rmp = float(rmp_init)
    success_count = 0
    total_xt = 0
    cross_task_count = 0  # DEBUG: track cross-task transfers

    # history logging
    history = {t: [] for t in range(T)}
    nfe = 0

    # DEBUG: Log knowledge transfer config and population distribution
    print(f"  [MFWOA-MT] T={T} tasks, pop_size={pop_size}, iters={iters}, rmp_init={rmp_init}, elitism={elitism}")
    # Count individuals per task
    counts = np.bincount(skill_factors, minlength=T)
    counts_str = ", ".join([f"K{Ks[t]}:{counts[t]}" for t in range(T)])
    print(f"    Population distribution: {counts_str} (total={pop_size})")
    # Show initial best per task
    x_best_str = ", ".join([f"K{Ks[t]}:FE={-best_score[t]:.4f}" for t in range(T)])
    print(f"    Initial X-best per task: {x_best_str}")
    print(f"    RMP (cross-task rate) = {rmp:.3f} (0.0=no transfer, 1.0=all cross-task)")

    for g in range(iters):
        frac = g / max(1, iters)
        a = 2.0 * (1 - g / max(1, iters - 1))
        # rmp scheduling: if provided, override rmp according to schedule
        if rmp_schedule is not None:
            # rmp_schedule is sequence of tuples (start_frac, end_frac, rmp_value)
            for (s_frac, e_frac, val) in rmp_schedule:
                if frac >= s_frac and frac < e_frac:
                    rmp = float(val)
                    break
        # If single-task, we may want to force rmp to zero to avoid transfer noise
        rmp_local = 0.0 if T == 1 else rmp

        # Prepare next generation container (we will allow elitism preservation)
        next_pop = pop.copy()

        for i in range(pop_size):
            sf = int(skill_factors[i])
            p_cross = rng.random()
            if p_cross < rmp_local:
                # cross-task interaction: pick random other task's leader
                other_tasks = [t for t in range(T) if t != sf]
                if other_tasks:
                    other = rng.choice(other_tasks)
                    leader = best_pos[other]
                    # mix leader genes into current
                    beta = rng.random()
                    new_pos = pop[i] * (1 - beta) + leader * beta
                    cross_task_count += 1  # DEBUG: track transfer
                else:
                    # fallback to intra-task
                    leader = best_pos[sf]
                    new_pos = pop[i].copy()
            else:
                # intra-task WOA operations relative to own task's leader
                leader = best_pos[sf]
                r1 = rng.random()
                r2 = rng.random()
                A = 2 * a * r1 - a
                C = 2 * r2
                p = rng.random()
                if p < 0.5:
                    if abs(A) < 1:
                        D = abs(C * leader - pop[i])
                        new_pos = leader - A * D
                    else:
                        rand_idx = rng.integers(pop_size)
                        X_rand = pop[rand_idx]
                        D = abs(C * X_rand - pop[i])
                        new_pos = X_rand - A * D
                else:
                    b = 2.5  # OPTIMIZED: increased from 1.0 for tighter spiral convergence
                    l = rng.uniform(-1, 1)
                    distance = abs(leader - pop[i])
                    new_pos = distance * np.exp(b * l) * np.cos(2 * np.pi * l) + leader
            
            # ===== CONSTRAINT: Sử dụng enforce_threshold_constraints =====
            # Đảm bảo new_pos tuân theo ràng buộc: sort, clip [1,254], gap enforcement
            new_pos = enforce_threshold_constraints(np.round(new_pos).astype(np.int32)).astype(np.float32)
            new_fit = objectives[sf](new_pos)
            total_xt += 1
            nfe += 1
            # replace in next_pop when improved (minimize: new_fit < fitness[i])
            if new_fit < fitness[i]:  # ⚠️ CHANGED: < instead of > for MINIMIZATION
                next_pop[i] = new_pos
                fitness[i] = new_fit
                success_count += 1
                if new_fit < best_score[sf]:  # ⚠️ CHANGED: < instead of > for MINIMIZATION
                    best_score[sf] = float(new_fit)
                    best_pos[sf] = new_pos.copy()

        # apply elitism: preserve best-N individuals per task into next_pop
        if elitism and elitism > 0:
            for t in range(T):
                mask = (skill_factors == t)
                if np.count_nonzero(mask) == 0:
                    continue
                # select indices of mask sorted by fitness ascending (best = lowest for minimization)
                idxs = np.where(mask)[0]
                # if fewer individuals than elitism, preserve all
                if idxs.size <= elitism:
                    continue
                sorted_idxs = idxs[np.argsort(fitness[idxs])]  # ⚠️ CHANGED: ascending order for minimization
                elite_idxs = sorted_idxs[:elitism]
                # copy elites into their positions in next_pop
                for ei in elite_idxs:
                    next_pop[ei] = pop[ei].copy()
                    # keep fitness consistent for elites
                    fitness[ei] = float(fitness[ei])
        # finalize population for next generation
        pop = next_pop
        # adapt rmp simply based on recent success fraction (if no explicit schedule)
        if rmp_schedule is None and total_xt > 0 and g % 10 == 0 and g > 0:
            frac_succ = success_count / (total_xt + EPS)
            # nudge rmp toward observed success (clamped)
            rmp = float(np.clip(0.5 * rmp + 0.5 * frac_succ, 0.05, 0.95))
            # reset counters
            success_count = 0
            total_xt = 0
            # DEBUG: Print progress every 10 iterations
            x_best_str = ", ".join([f"K{Ks[t]}:FE={-best_score[t]:.4f}" for t in range(T)])
            print(f"      [Iter {g:3d}] X-best: {x_best_str}, RMP={rmp:.3f}, nfe={nfe}")
        
        # log best per-task for history
        for t in range(T):
            history[t].append(best_score[t])
    # finalize best thresholds
    best_thresholds = [continuous_to_thresholds(best_pos[t], Ks[t]) if best_pos[t] is not None else [] for t in range(T)]
    # ⚠️ IMPORTANT: Negate best_score back to positive FE for reporting
    # (objectives return -FE for minimization, but we want to report positive FE values)
    
    # DEBUG: Report final stats
    x_best_final = ", ".join([f"K{Ks[t]}:FE={-best_score[t]:.4f}" for t in range(T)])
    print(f"  [MFWOA-MT DONE] Final X-best: {x_best_final}")
    print(f"    cross_task_transfers={cross_task_count}/{nfe} ({100*cross_task_count/(nfe+1):.1f}%), rmp_final={rmp:.3f}")
    
    diagnostics = {'history': history, 'nfe': nfe}
    return best_thresholds, [-float(s) for s in best_score], diagnostics
