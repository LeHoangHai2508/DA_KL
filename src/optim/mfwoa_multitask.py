
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
    elitism: int = None,  #  Make adaptive
    rmp_schedule: Sequence[tuple] = None,
    membership: str = "triangular",
    lambda_penalty: float = 1.0,
    alpha_area: float = 0.10,
    beta_membership: float = 0.10,
    gamma_spacing: float = 0.20,
    enable_mutation: bool = True,  # Enable mutation by default
    mutation_rate: float = 0.10,  #  10% mutation chance
) -> Tuple[List[List[int]], List[float], dict]:
    """Run simplified MFWOA for multiple tasks simultaneously.
    
    ===== MULTITASK MFWOA: Knowledge Transfer + Adaptive RMP + CRITICAL=====
    
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
        elitism: Number of elite individuals per task to preserve (None = adaptive)
        rmp_schedule: Optional schedule [(start_frac, end_frac, rmp_value), ...]
        membership: Membership function type cho compute_fuzzy_entropy (triangular/gaussian/s)
        lambda_penalty: Weighting factor cho penalizer
        alpha_area: Weight cho P_A penalty (cân bằng kích thước lớp)
        beta_membership: Weight cho P_μ penalty (tránh concentrated membership)
        gamma_spacing: Weight cho P_spacing penalty (enforce even distribution)
        enable_mutation: Enable mutation operator for diversity (default True)
        mutation_rate: Probability of mutation per individual (default 0.10)
    
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
    
    # ===== ADAPTIVE ELITISM =====
    if elitism is None:
        # Scale elitism with max(K):
        # - K=2-4: 5-6 elites
        # - K=5-7: 8-10 elites
        # - K=8-10: 12-15 elites
        # Formula: min(15% of pop, max(5, 1.5*maxK))
        elitism = min(int(pop_size * 0.15), max(5, int(maxK * 1.5)))
    
    print(f"  [MFWOA-MT] T={T} tasks, pop_size={pop_size}, iters={iters}, rmp_init={rmp_init}")
    print(f"  Adaptive elitism={elitism} (scaled with maxK={maxK})")
    print(f"  Mutation={'ENABLED' if enable_mutation else 'DISABLED'} (rate={mutation_rate})")
    
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
            """Objective function: minimize -FE (equivalent to maximizing FE) with penalties
        
            """
            thr = continuous_to_thresholds(pos, K)
            fe_val = compute_fuzzy_entropy(
                hist, thr, 
                membership=membership,
                for_minimization=False,
                lambda_penalty=lambda_penalty,
                alpha_area=alpha_area,
                beta_membership=beta_membership,
                gamma_spacing=gamma_spacing,
            )
            # Negate to convert maximization (FE) to minimization (-FE)
            # This matches WOA's approach: minimize -FE instead of maximize FE
            return -float(fe_val)
        return obj_task
    
    objectives = [make_obj(h, K) for h, K in zip(hists, Ks)]

    # ===== INITIALIZE POPULATION =====
    #  Initialize as CONTINUOUS float (not rounded)
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
    cross_task_count = 0  # DEBUG: track cross-task transfers
    mutation_count = 0  

    # history logging
    history = {t: [] for t in range(T)}
    nfe = 0

    # Track best_score per task from PREVIOUS iteration (for Algorithm 2 RMP update)
    best_score_prev = best_score.copy()

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
        
        # ===== ALGORITHM 2: RMP UPDATE (Lines 12-15) =====
        # Check if ANY task improved since previous iteration
        # If NO task improved: rmp = rmp + δN(0,1) with δ=0.1
        # If ANY task improved: rmp stays same
        if g > 0:
            task_improved = any(best_score[t] < best_score_prev[t] for t in range(T))
            if not task_improved:
                # No task improved -> update RMP by Gaussian noise
                delta = 0.1
                gaussian_noise = rng.normal(0, 1)  # N(0,1)
                rmp = float(np.clip(rmp + delta * gaussian_noise, 0.0, 1.0))
        
        # If custom schedule provided, override
        if rmp_schedule is not None:
            for (s_frac, e_frac, val) in rmp_schedule:
                if frac >= s_frac and frac < e_frac:
                    rmp = float(val)
                    break
        
        # If single-task, force rmp to zero
        rmp_local = 0.0 if T == 1 else rmp

        # Prepare next generation container (we will allow elitism preservation)
        next_pop = pop.copy()

        for i in range(pop_size):
            sf = int(skill_factors[i])
            
            # ===== ALGORITHM 3: KNOWLEDGE TRANSFER =====
            rand_i1 = rng.random()  # For cross-task decision
            rand_i2 = rng.random()  # For WOA method selection
            rand_i3 = rng.random()  # For mutation
            rand_i4 = rng.random()  # For skill factor change
            
            if rand_i1 < rmp_local:  # Inter-task knowledge transfer (Algorithm 3, Step 2-20)
                # Step 3: Randomly select one other task
                other_tasks = [t for t in range(T) if t != sf]
                if other_tasks:
                    other_task = rng.choice(other_tasks)
                    
                    # Step 4: Randomly select a learned individual from other task
                    other_mask = (skill_factors == other_task)
                    if np.count_nonzero(other_mask) > 0:
                        other_indices = np.where(other_mask)[0]
                        j_idx = rng.choice(other_indices)
                        X_j = pop[j_idx]
                    else:
                        X_j = best_pos[other_task]
                    
                    # Step 5-18: Choose method (first way or second way)
                    if rand_i2 < 0.2:  # First way: traditional WOA
                        leader = best_pos[sf]
                        r1 = rng.random()
                        r2 = rng.random()
                        A = 2 * a * r1 - a
                        C = 2 * r2
                        p = rng.random()
                        
                        if p < 0.5:
                            if abs(A) < 1:  # Encircling prey
                                D1 = abs(C * leader - pop[i])
                                D_other = abs(C * best_pos[other_task] - X_j)
                                new_pos = leader - A * (D1 + D_other) / 2.0
                            else:  # Search for prey
                                X_rand = pop[rng.integers(pop_size)]
                                D2 = abs(C * X_rand - pop[i])
                                other_pop_idxs = np.where(skill_factors == other_task)[0]
                                if len(other_pop_idxs) >= 2:
                                    X_j1 = pop[rng.choice(other_pop_idxs)]
                                    X_j2 = pop[rng.choice(other_pop_idxs)]
                                elif len(other_pop_idxs) == 1:
                                    X_j1 = pop[other_pop_idxs[0]]
                                    X_j2 = best_pos[other_task]
                                else:
                                    X_j1 = best_pos[other_task]
                                    X_j2 = best_pos[other_task]
                                D_other = abs(C * X_j1 - X_j2)
                                new_pos = X_rand - A * (D2 + D_other) / 2.0
                        else:  # Bubble-net attack
                            b = 2.5
                            l = rng.uniform(-1, 1)
                            D_prime = abs(leader - pop[i])
                            D_other = abs(C * best_pos[other_task] - X_j)
                            new_pos = ((D_prime + D_other) / 2.0) * np.exp(b * l) * np.cos(2 * np.pi * l) + leader
                    else:  # Second way: crossover + mutation
                        # Step 14-15: Crossover
                        X_rand_k = pop[rng.integers(pop_size)]
                        X_rand_new = (X_rand_k + X_j) / 2.0  # Crossover
                        X_best_new = (best_pos[sf] + best_pos[other_task]) / 2.0
                        
                        # Step 16-18: Mutation
                        if rand_i3 < 0.1:
                            n_mut = rng.integers(1, min(3, maxK + 1))
                            mut_dims = rng.choice(maxK, n_mut, replace=False)
                            for d in mut_dims:
                                X_rand_new[d] = rng.uniform(1.0, 254.0)
                                X_best_new[d] = rng.uniform(1.0, 254.0)
                            mutation_count += 1
                        
                        # Step 19: Use best as new position
                        new_pos = X_best_new
                    
                    # Step 20-21: Potentially change skill factor
                    if rand_i4 < 0.5:
                        skill_factors[i] = other_task
                        sf = other_task
                    
                    cross_task_count += 1
                else:
                    # Fallback: standard WOA
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
                            X_rand = pop[rng.integers(pop_size)]
                            D = abs(C * X_rand - pop[i])
                            new_pos = X_rand - A * D
                    else:
                        b = 2.5
                        l = rng.uniform(-1, 1)
                        distance = abs(leader - pop[i])
                        new_pos = distance * np.exp(b * l) * np.cos(2 * np.pi * l) + leader
            else:  # Intra-task: traditional WOA (Algorithm 3, Step 22)
                # ===== INTRA-TASK WOA OPERATIONS =====
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
                        X_rand = pop[rng.integers(pop_size)]
                        D = abs(C * X_rand - pop[i])
                        new_pos = X_rand - A * D
                else:
                    b = 2.5
                    l = rng.uniform(-1, 1)
                    distance = abs(leader - pop[i])
                    new_pos = distance * np.exp(b * l) * np.cos(2 * np.pi * l) + leader
            
            # =====  LAZY CONSTRAINT ENFORCEMENT =====
            # Only clip to range [1, 254], NO ROUNDING during search
            # Rounding happens only in continuous_to_thresholds (when evaluating fitness)
            new_pos = np.clip(new_pos, 1.0, 254.0)  # ← Keep CONTINUOUS
            
            # =====  MUTATION OPERATOR =====
            if enable_mutation and rng.random() < mutation_rate:
                # Mutate 1-2 dimensions randomly to maintain diversity
                n_mutate = rng.integers(1, min(3, maxK + 1))
                mut_dims = rng.choice(maxK, n_mutate, replace=False)
                for d in mut_dims:
                    new_pos[d] = rng.uniform(1.0, 254.0)
                mutation_count += 1
            
            # ===== EVALUATE NEW POSITION =====
            new_fit = objectives[sf](new_pos)  # ← Constraints enforced INSIDE objective
            nfe += 1
            
            # replace in next_pop when improved (minimize: new_fit < fitness[i])
            if new_fit < fitness[i]:  # ⚠️ CHANGED: < instead of > for MINIMIZATION
                next_pop[i] = new_pos
                fitness[i] = new_fit
                if new_fit < best_score[sf]:  # ⚠️ CHANGED: < instead of > for MINIMIZATION
                    best_score[sf] = float(new_fit)
                    best_pos[sf] = new_pos.copy()

        # ===== APPLY ELITISM =====
        # preserve best-N individuals per task into next_pop
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
        
        # ===== END OF ITERATION: Update best_score_prev for next RMP check =====
        best_score_prev = best_score.copy()
        
        # ===== ENHANCED LOGGING =====
        if g % 10 == 0 and g > 0:
            x_best_str = ", ".join([f"K{Ks[t]}:FE={-best_score[t]:.4f}" for t in range(T)])
            print(f"      [Iter {g:3d}] X-best: {x_best_str}, RMP={rmp:.3f}, nfe={nfe}, mutations={mutation_count}")
        
        # log best per-task for history
        for t in range(T):
            history[t].append(best_score[t])
    
    # ===== FINALIZE BEST THRESHOLDS =====
    best_thresholds = [continuous_to_thresholds(best_pos[t], Ks[t]) if best_pos[t] is not None else [] for t in range(T)]
    
    # ⚠️ IMPORTANT: Negate best_score back to positive FE for reporting
    # (objectives return -FE for minimization, but we want to report positive FE values)
    
    # DEBUG: Report final stats
    x_best_final = ", ".join([f"K{Ks[t]}:FE={-best_score[t]:.4f}" for t in range(T)])
    print(f"  [MFWOA-MT DONE] Final X-best: {x_best_final}")
    print(f"    cross_task_transfers={cross_task_count}/{nfe} ({100*cross_task_count/(nfe+1):.1f}%), rmp_final={rmp:.3f}")
    print(f"    ✅ mutations={mutation_count}/{nfe} ({100*mutation_count/(nfe+1):.1f}%)")
    
    diagnostics = {'history': history, 'nfe': nfe, 'cross_task_count': cross_task_count, 'mutation_count': mutation_count}
    return best_thresholds, [-float(s) for s in best_score], diagnostics