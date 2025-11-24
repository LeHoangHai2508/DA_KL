"""
MFWOA with optimized single-task mode
======================================
Detect T=1 and skip all multitask overhead
"""
from __future__ import annotations
import numpy as np
from typing import List, Sequence, Tuple

from src.metrics.fuzzy_entropy import compute_fuzzy_entropy
from src.seg.utils import enforce_threshold_constraints

EPS = 1e-12


def continuous_to_thresholds(pos: np.ndarray, K: int) -> List[int]:
    """Convert continuous position to discrete thresholds."""
    arr = pos[:K]
    constrained = enforce_threshold_constraints(np.round(arr).astype(np.int32))
    return [int(x) for x in constrained]


def sbx_crossover(p1: np.ndarray, p2: np.ndarray, rng: np.random.Generator, 
                  eta: float = 2.0, pc: float = 0.9) -> Tuple[np.ndarray, np.ndarray]:
    """Simulated Binary Crossover (SBX)."""
    c1, c2 = p1.copy(), p2.copy()
    for i in range(len(p1)):
        if rng.random() < pc:
            if abs(p1[i] - p2[i]) > EPS:
                u = rng.random()
                if u <= 0.5:
                    beta = (2 * u) ** (1 / (eta + 1))
                else:
                    beta = (1 / (2 * (1 - u))) ** (1 / (eta + 1))
                c1[i] = 0.5 * ((1 + beta) * p1[i] + (1 - beta) * p2[i])
                c2[i] = 0.5 * ((1 - beta) * p1[i] + (1 + beta) * p2[i])
    return c1, c2


def gaussian_mutation(ind: np.ndarray, rng: np.random.Generator, 
                      pm: float = 0.15, sigma: float = 10.0,
                      lb: float = 1.0, ub: float = 254.0) -> np.ndarray:
    """Gaussian mutation."""
    mutant = ind.copy()
    for i in range(len(ind)):
        if rng.random() < pm:
            mutant[i] += sigma * rng.standard_normal()
    return np.clip(mutant, lb, ub)


def update_factorial_ranks(factorial_costs: np.ndarray, T: int) -> np.ndarray:
    """Update factorial ranks (only for T > 1)."""
    pop_size = factorial_costs.shape[0]
    factorial_ranks = np.zeros_like(factorial_costs, dtype=int)
    
    for t in range(T):
        valid_mask = factorial_costs[:, t] < np.inf
        if np.any(valid_mask):
            valid_costs = factorial_costs[valid_mask, t]
            order = np.argsort(valid_costs)
            ranks = np.full(pop_size, pop_size + 1, dtype=int)
            valid_indices = np.where(valid_mask)[0]
            ranks[valid_indices[order]] = np.arange(1, len(order) + 1)
            factorial_ranks[:, t] = ranks
        else:
            factorial_ranks[:, t] = pop_size + 1
    
    return factorial_ranks


# def mfwoa_single_task_fast(
#     hist: np.ndarray,
#     K: int,
#     pop_size: int,
#     iters: int,
#     rng: np.random.Generator,
#     membership: str,
#     lambda_penalty: float,
#     alpha_area: float,
#     beta_membership: float,
#     gamma_spacing: float,
#     verbose: bool,
# ) -> Tuple[List[int], float]:
#     """
#     Fast single-task WOA (no multitask overhead).
#     Pure WOA with greedy selection.
#     """
#     # Objective function
#     def objective(pos: np.ndarray) -> float:
#         try:
#             thr = continuous_to_thresholds(pos, K)
#             fe_val = compute_fuzzy_entropy(
#                 hist, thr, membership=membership, for_minimization=False,
#                 lambda_penalty=lambda_penalty, alpha_area=alpha_area,
#                 beta_membership=beta_membership, gamma_spacing=gamma_spacing,
#             )
#             return -float(fe_val)
#         except:
#             return 1e9
    
#     # Initialize population
#     pop = rng.uniform(1.0, 254.0, size=(pop_size, K))
#     fitness = np.full(pop_size, np.inf)
    
#     # Initial evaluation
#     for i in range(pop_size):
#         fitness[i] = objective(pop[i])
    
#     best_idx = int(np.argmin(fitness))
#     best_pos = pop[best_idx].copy()
#     best_score = fitness[best_idx]
    
#     if verbose:
#         fe_val = -best_score
#         print(f"[MFWOA-FAST] Initial FE: {fe_val:.6f}")
    
#     # Main loop (pure WOA with greedy)
#     for g in range(iters):
#         a = 2.0 - 2.0 * g / max(iters - 1, 1)
        
#         for i in range(pop_size):
#             Xi = pop[i].copy()
            
#             r1, r2 = rng.random(), rng.random()
#             A = 2.0 * a * r1 - a
#             C = 2.0 * r2
#             p = rng.random()
#             l = rng.uniform(-1.0, 1.0)
#             b = 1.0
            
#             # WOA position update
#             if p < 0.5:
#                 if abs(A) < 1.0:
#                     D = np.abs(C * best_pos - Xi)
#                     new_pos = best_pos - A * D
#                 else:
#                     rand_idx = rng.integers(pop_size)
#                     X_rand = pop[rand_idx]
#                     D = np.abs(C * X_rand - Xi)
#                     new_pos = X_rand - A * D
#             else:
#                 D_prime = np.abs(best_pos - Xi)
#                 new_pos = D_prime * np.exp(b * l) * np.cos(2 * np.pi * l) + best_pos
            
#             new_pos = np.clip(new_pos, 1.0, 254.0)
            
#             # Greedy selection
#             new_score = objective(new_pos)
#             if new_score < fitness[i]:
#                 pop[i] = new_pos
#                 fitness[i] = new_score
                
#                 if new_score < best_score:
#                     best_score = new_score
#                     best_pos = new_pos.copy()
    
#     if verbose:
#         fe_val = -best_score
#         print(f"[MFWOA-FAST] Final FE: {fe_val:.6f}")
    
#     thr = continuous_to_thresholds(best_pos, K)
#     fe = -float(best_score)
#     return thr, fe


def mfwoa_multitask(
    hists: Sequence[np.ndarray],
    Ks: Sequence[int],
    pop_size :int = None,
    iters: int = None,
    rng: np.random.Generator = None,
    rmp_init: float = 0.3,
    pop_per_task: int = None,
    membership: str = "triangular",
    lambda_penalty: float = 1.0,
    alpha_area: float = 0.10,
    beta_membership: float = 0.10,
    gamma_spacing: float = 0.20,
    pc: float = 0.9,
    pm: float = 0.15,
    elitism: int = None,
    enable_mutation: bool = True,
    mutation_rate: float = 0.2,
    rmp_schedule: Sequence[tuple] = None,
    verbose: bool = True,
    log_interval: int = 50,
    greedy_selection: bool = False,
    acceptance_threshold: float = 0.05,
) -> Tuple[List[List[int]], List[float], dict]:
    """
    MFWOA with optimized single-task detection.
    
    âœ… NEW: Auto-detect T=1 and use fast single-task mode
    """
    if rng is None:
        rng = np.random.default_rng(42)

    T = len(Ks)
    if T == 0:
        print("[MFWOA] ERROR: No tasks provided (T=0)")
        return [], [], {"history": {}, "nfe": 0, "cross_task_count": 0}

    # âœ… NEW: Fast path for single-task
    if T == 0:
        print("[MFWOA] ERROR: No tasks provided (T=0)")
        return [], [], {"history": {}, "nfe": 0, "cross_task_count": 0}
    
    if T == 1:
        print("[MFWOA] WARNING: T=1 detected, but running full multitask pipeline")

    # ===== MULTITASK MODE (T > 1) =====
    maxK = int(max(Ks))
    if pop_per_task is not None and pop_per_task > 0:
        pop_size = int(pop_per_task) * T
    pop_size = max(int(pop_size), T)

    if elitism is None:
        elitism = max(1, int(0.05 * pop_size))

    if verbose == True:
        print("=" * 60)
        print("[MFWOA-MULTITASK] INITIALIZATION")
        print("=" * 60)
        print(f"  Tasks: {T}, Dims: {list(Ks)}, Pop: {pop_size}, Iters: {iters}")
        print(f"  RMP: {rmp_init}, Greedy: {greedy_selection}")
        print("=" * 60)

    # Objective functions
    def make_obj(hist, K, task_id):
        def obj_task(pos: np.ndarray) -> float:
            try:
                thr = continuous_to_thresholds(pos, K)
                fe_val = compute_fuzzy_entropy(
                    hist, thr, membership=membership, for_minimization=False,
                    lambda_penalty=lambda_penalty, alpha_area=alpha_area,
                    beta_membership=beta_membership, gamma_spacing=gamma_spacing,
                )
                return -float(fe_val)
            except:
                return 1e9
        return obj_task

    objectives = [make_obj(h, K, t) for t, (h, K) in enumerate(zip(hists, Ks))]

    # Initialize population
    pop = rng.uniform(1.0, 254.0, size=(pop_size, maxK))
    
    # Initial evaluation
    factorial_costs = np.full((pop_size, T), np.inf, dtype=float)
    nfe = 0
    
    for i in range(pop_size):
        for t in range(T):
            factorial_costs[i, t] = objectives[t](pop[i])
            nfe += 1

    # Initial ranks and skill factors
    factorial_ranks = update_factorial_ranks(factorial_costs, T)
    skill_factors = np.argmin(factorial_ranks, axis=1)

    # Initialize best solutions
    best_pos: List[np.ndarray] = [None] * T
    best_score = np.full(T, np.inf, dtype=float)
    
    for t in range(T):
        valid_mask = factorial_costs[:, t] < np.inf
        if np.any(valid_mask):
            valid_costs = factorial_costs[valid_mask, t]
            best_idx = np.where(valid_mask)[0][int(np.argmin(valid_costs))]
            best_score[t] = float(factorial_costs[best_idx, t])
            best_pos[t] = pop[best_idx].copy()

    # Tracking
    fitness = factorial_costs[np.arange(pop_size), skill_factors].copy()
    rmp = float(rmp_init)
    cross_task_count = 0
    mutation_count = 0
    best_score_prev = best_score.copy()
    history = {t: [best_score[t]] for t in range(T)}
    improvements_per_iter = []
    stagnation_counter = 0

    # ===== MAIN MULTITASK LOOP =====
    for g in range(iters):
        a = 2.0 - 2.0 * g / max(iters - 1, 1)
        
        iter_improvements = 0
        
        # Adaptive RMP
        if g > 0:
            # Check if ALL tasks are stagnant
            all_stagnant = np.all(best_score >= best_score_prev - EPS)
            
            if all_stagnant:
                # Add Gaussian noise (paper: δ=0.1)
                delta = 0.1
                noise = delta * rng.standard_normal()
                rmp = np.clip(rmp + noise, 0.0, 1.0)
                
                if verbose == True:
                # In ra g (vòng lặp) và rmp hiện tại bất kể có bị stagnant hay không
                    print(f"  [Iter {g}/{iters}] RMP: {rmp:.4f}")
                    # Lặp qua từng task (từng K) để in chi tiết
                    for t in range(T):
                        k_val = Ks[t]
                        
                        # 1. Lấy FE (Lưu ý: thuật toán đang tìm Min(-FE), nên cần đổi dấu lại thành dương)
                        if best_score[t] < 1e9:
                            current_fe = -best_score[t] 
                        else:
                            current_fe = 0.0
                        
                        # 2. Lấy Thresholds (Ngưỡng) hiện tại
                        if best_pos[t] is not None:
                            # Gọi hàm helper có sẵn đầu file để chuyển sang số nguyên
                            curr_thr = continuous_to_thresholds(best_pos[t], k_val)
                        else:
                            curr_thr = []
                        
                        # 3. Xuất ra màn hình (Mỗi K một dòng để dễ nhìn như bạn muốn)
                        print(f"   [K={k_val}] FE: {current_fe:.6f} | Thr: {curr_thr}")
            
            # Update prev scores for next iteration
            best_score_prev = best_score.copy()

        for i in range(pop_size):
            k = int(skill_factors[i])
            Xi = pop[i].copy()

            rand1, rand2, rand3, rand4 = rng.random(4)
            r1, r2 = rng.random(), rng.random()
            A = 2.0 * a * r1 - a
            C = 2.0 * r2
            p = rng.random()
            l = rng.uniform(-1.0, 1.0)
            b = 1.0

            new_pos = Xi.copy()
            new_skill = k

            # ===== INTER-TASK OR INNER-TASK =====
            if rand1 < rmp:
                # ===== INTER-TASK KNOWLEDGE TRANSFER =====
                cross_task_count += 1
                
                # Random select other task (NO similarity filtering)
                other_tasks = [tt for tt in range(T) if tt != k]
                j = int(rng.choice(other_tasks))
                
                # Select individual from task j
                idx_j_cands = np.where(skill_factors == j)[0]
                X_j = pop[int(rng.choice(idx_j_cands))] if len(idx_j_cands) > 0 else \
                      (best_pos[j] if best_pos[j] is not None else Xi)
                
                leader_k = best_pos[k] if best_pos[k] is not None else Xi
                leader_j = best_pos[j] if best_pos[j] is not None else X_j
                
                # Method 1 (20%) or Method 2 (80%)
                if rand2 < 0.2:
                    # ===== METHOD 1: Add distance term (Eq. 11-13) =====
                    if p < 0.5:
                        if abs(A) < 1.0:  # Encircling prey
                            D1 = np.abs(C * leader_k - Xi)
                            D_other = np.abs(C * leader_j - X_j)
                            D_avg = (D1 + D_other) / 2.0
                            new_pos = leader_k - A * D_avg
                        else:  # Search for prey
                            idx_k = np.where(skill_factors == k)[0]
                            X_rand_k = pop[int(rng.choice(idx_k))] if len(idx_k) > 0 else Xi
                            D2 = np.abs(C * X_rand_k - Xi)
                            D_other = np.abs(C * leader_j - X_j)
                            D_avg = (D2 + D_other) / 2.0
                            new_pos = X_rand_k - A * D_avg
                    else:  # Bubble-net attack
                        D_prime = np.abs(leader_k - Xi)
                        D_other = np.abs(leader_j - X_j)
                        D_avg = (D_prime + D_other) / 2.0
                        new_pos = D_avg * np.exp(b * l) * np.cos(2 * np.pi * l) + leader_k
                
                else:
                    # ===== METHOD 2: Crossover + Mutation (lines 14-19) =====
                    idx_k = np.where(skill_factors == k)[0]
                    X_rand_k = pop[int(rng.choice(idx_k))].copy() if len(idx_k) > 0 else Xi.copy()
                    
                    # Crossover
                    X_rand_k, _ = sbx_crossover(X_rand_k, X_j, rng, pc=pc)
                    leader_k_new, _ = sbx_crossover(leader_k.copy(), leader_j.copy(), rng, pc=pc)
                    
                    # Mutation (paper uses 0.1, not 0.2)
                    if rand3 < 0.1:
                        X_rand_k = gaussian_mutation(X_rand_k, rng, pm=pm)
                        leader_k_new = gaussian_mutation(leader_k_new, rng, pm=pm)
                        mutation_count += 1
                    
                    # Use new leaders in WOA update
                    if p < 0.5:
                        if abs(A) < 1.0:
                            D1 = np.abs(C * leader_k_new - Xi)
                            new_pos = leader_k_new - A * D1
                        else:
                            D2 = np.abs(C * X_rand_k - Xi)
                            new_pos = X_rand_k - A * D2
                    else:
                        dist = np.abs(leader_k_new - Xi)
                        new_pos = dist * np.exp(b * l) * np.cos(2 * np.pi * l) + leader_k_new
                
                # Horizontal cultural transmission (line 20-21, paper uses 0.5)
                if rand4 < 0.5:
                    new_skill = j
            
            else:
                # ===== INNER-TASK: Traditional WOA =====
                leader_k = best_pos[k] if best_pos[k] is not None else Xi
                
                if p < 0.5:
                    if abs(A) < 1.0:  # Encircling prey
                        D = np.abs(C * leader_k - Xi)
                        new_pos = leader_k - A * D
                    else:  # Search for prey
                        idx_k = np.where(skill_factors == k)[0]
                        X_rand = pop[int(rng.choice(idx_k))] if len(idx_k) > 0 else pop[rng.integers(pop_size)]
                        D = np.abs(C * X_rand - Xi)
                        new_pos = X_rand - A * D
                else:  # Bubble-net attack
                    dist = np.abs(leader_k - Xi)
                    new_pos = dist * np.exp(b * l) * np.cos(2 * np.pi * l) + leader_k

            # Clip to bounds
            new_pos = np.clip(new_pos, 1.0, 254.0)
            
            # Evaluate new position
            cost_new = objectives[new_skill](new_pos)
            nfe += 1
            
            # ===== ACCEPTANCE STRATEGY =====
            accept = False
            if greedy_selection:
                if cost_new < fitness[i] - EPS:
                    accept = True
                    iter_improvements += 1
            else:
                # Balanced acceptance
                if cost_new < fitness[i] - EPS:
                    accept = True
                    iter_improvements += 1
                elif cost_new < fitness[i] * (1 + acceptance_threshold):
                    prob = 0.3 * (1 - g / iters)
                    if rng.random() < prob:
                        accept = True
            
            if accept:
                pop[i] = new_pos
                skill_factors[i] = new_skill
                fitness[i] = cost_new
                factorial_costs[i, new_skill] = cost_new

        # Update ranks
        factorial_ranks = update_factorial_ranks(factorial_costs, T)

        # Update best
        for t in range(T):
            valid_mask = factorial_costs[:, t] < np.inf
            if np.any(valid_mask):
                valid_costs = factorial_costs[valid_mask, t]
                best_idx = np.where(valid_mask)[0][int(np.argmin(valid_costs))]
                best_val = float(factorial_costs[best_idx, t])
                
                if best_val < best_score[t] - EPS:
                    best_score[t] = best_val
                    best_pos[t] = pop[best_idx].copy()

        for t in range(T):
            history[t].append(best_score[t])
        
        improvements_per_iter.append(iter_improvements)

    # Final results
    best_thresholds = []
    best_FEs = []
    for t in range(T):
        if best_pos[t] is None:
            best_thresholds.append([])
            best_FEs.append(float("nan"))
        else:
            thr = continuous_to_thresholds(best_pos[t], Ks[t])
            fe = -float(best_score[t])
            best_thresholds.append(thr)
            best_FEs.append(fe)
    
    return best_thresholds, best_FEs, {
        "history": history, 
        "nfe": nfe,
        "cross_task_count": cross_task_count,
        "mutation_count": mutation_count,
        "improvements_per_iter": improvements_per_iter,
    }