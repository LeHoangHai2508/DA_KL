"""PSO optimization for image thresholding - CORRECTED VERSION

Particle Swarm Optimization (PSO) - thuật toán dựa trên hành vi của đàn chim.

CORRECTIONS:
1. Remove redundant constraint enforcement in evaluation
2. Properly constraint pbest and gbest
3. Evaluate only once per iteration
4. Fix position update to avoid double constraint
"""
from typing import Tuple, Callable
import numpy as np


def pso_optimize(hist: np.ndarray, K: int, pop_size: int, iters: int,
                objective: Callable[[np.ndarray], float]) -> Tuple[list[int], float]:
    """PSO optimization for thresholding.
    
    Particle Swarm Optimization để tối ưu ngưỡng phân đoạn ảnh.
    
    Args:
        hist: Image histogram (256-bin)
        K: Number of thresholds
        pop_size: Population size (số lượng particles, thường 20*K)
        iters: Number of iterations
        objective: Objective function to minimize (fitness function, thấp hơn càng tốt)
    
    Returns:
        Tuple: (Best thresholds list of int, best score float)
    """
    from src.seg.utils import enforce_threshold_constraints
    
    # ===== INITIALIZE PARTICLES =====
    particles = np.zeros((pop_size, K), dtype=np.float32)
    rng = np.random.default_rng()
    
    for i in range(pop_size):
        thresholds = np.sort(rng.uniform(1, 254, K))
        particles[i] = thresholds
    
    # ===== VELOCITY INITIALIZATION =====
    Vmax = (254 - 1) / 10.0  # ≈ 25.3
    velocities = np.zeros((pop_size, K), dtype=np.float32)
    for i in range(pop_size):
        velocities[i] = rng.uniform(-Vmax, Vmax, K)
    
    # ===== PSO PARAMETERS =====
    w_max = 0.9  # Initial inertia weight
    w_min = 0.4  # Final inertia weight
    c1 = 2.0     # Cognitive parameter
    c2 = 2.0     # Social parameter
    
    # ===== PERSONAL BEST (FIX 1: Constraint pbest properly) =====
    pbest = np.zeros_like(particles)
    pbest_scores = np.zeros(pop_size)
    
    for i in range(pop_size):
        # ✅ Constraint particle and save constrained version
        constrained = enforce_threshold_constraints(particles[i])
        pbest[i] = constrained
        pbest_scores[i] = objective(constrained)
    
    # ===== GLOBAL BEST (FIX 2: Constraint gbest properly) =====
    valid_scores = [(i, s) for i, s in enumerate(pbest_scores) if s is not None]
    if not valid_scores:
        return [], None
    
    gbest_idx = min(valid_scores, key=lambda x: x[1])[0]
    gbest = pbest[gbest_idx].copy()  # ✅ Now pbest is already constrained
    gbest_score = pbest_scores[gbest_idx]
    
    # ===== MAIN LOOP =====
    for iter in range(iters):
        # Linear decreasing inertia weight
        w = w_max - (w_max - w_min) * iter / max(1, iters - 1)
        
        # Update velocities (vectorized)
        r1 = rng.random(particles.shape)
        r2 = rng.random(particles.shape)
        
        velocities = (
            w * velocities
            + c1 * r1 * (pbest - particles)
            + c2 * r2 * (gbest - particles)
        )
        
        # Velocity clamping
        velocities = np.clip(velocities, -Vmax, Vmax)
        
        # Update positions
        particles = particles + velocities
        
        # ===== CONSTRAINT AND EVALUATE (FIX 3: Only once per particle) =====
        for i in range(pop_size):
            # Clip to bounds
            particles[i] = np.clip(particles[i], 1, 254)
            # Sort
            particles[i] = np.sort(particles[i])
            # Enforce all constraints (unique, min_gap, etc.)
            particles[i] = enforce_threshold_constraints(particles[i])
            
            # ✅ Evaluate only once (particles[i] is already constrained)
            score = objective(particles[i])
            
            # Update personal best
            if score is not None and pbest_scores[i] is not None:
                if score < pbest_scores[i]:
                    pbest[i] = particles[i].copy()  # ✅ Save constrained version
                    pbest_scores[i] = score
            elif score is not None and pbest_scores[i] is None:
                pbest[i] = particles[i].copy()
                pbest_scores[i] = score
            
            # Update global best
            if score is not None:
                if gbest_score is None or score < gbest_score:
                    gbest = particles[i].copy()  # ✅ Save constrained version
                    gbest_score = score
    
    # ===== RETURN RESULTS =====
    if gbest_score is None:
        return [], None
    
    # ✅ gbest is already constrained, just convert to int
    final_thresholds = [int(t) for t in gbest]
    return final_thresholds, float(gbest_score)