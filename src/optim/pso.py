"""PSO implementation for image thresholding.

Particle Swarm Optimization (PSO) - thuật toán dựa trên hành vi của đàn chim.
Mỗi particle (hạt) lưu trữ position (vị trí), velocity (vận tốc), và personal best.
"""
from typing import Tuple, Callable
import numpy as np


def pso_optimize(hist: np.ndarray, K: int, pop_size: int, iters: int,
                objective: Callable[[np.ndarray], float]) -> Tuple[list[int], float]:
    """PSO optimization for thresholding.
    
    Particle Swarm Optimization để tối ưu ngưỡng phân đoạn ảnh.
    Mỗi particle theo dõi: vị trí (position), vận tốc (velocity), personal best.
    
    Args:
        hist: Image histogram (256-bin, dùng để quyết định dimension)
        K: Number of thresholds (số ngưỡng = dimension của bài toán)
        pop_size: Population size (số lượng particles, thường 20*K)
        iters: Number of iterations (số vòng lặp)
        objective: Objective function to minimize (fitness function, thấp hơn càng tốt)
    
    Returns:
        Tuple: (Best thresholds list of int, best score float)
    """
    # Import constraint function
    from src.seg.utils import enforce_threshold_constraints
    
    # ===== KHỞI TẠO PARTICLES (QUẦN THỂ) =====
    # Mảng lưu position của tất cả particles
    # Shape: (pop_size, K) - mỗi hàng là một particle (K thresholds)
    particles = np.zeros((pop_size, K), dtype=np.float32)
    # Random number generator không cố định seed
    rng = np.random.default_rng()
    
    # Khởi tạo mỗi particle bằng K ngưỡng ngẫu nhiên
    for i in range(pop_size):
        # ===== RANDOM INITIALIZATION =====
        # Sinh K số ngẫu nhiên từ Uniform(1, 254)
        thresholds = np.sort(rng.uniform(1, 254, K))  # Sort để đảm bảo thứ tự
        # Gán vào particle i
        particles[i] = thresholds
    
    # ===== KHỞI TẠO VELOCITIES (VẬN TỐC) =====
    # Theo Algorithm 1, Line 4: v0_i ← random vector within [LB, UB]^D
    # Vận tốc khởi tạo random từ [1, 254]
    velocities = np.zeros((pop_size, K), dtype=np.float32)
    for i in range(pop_size):
        velocities[i] = rng.uniform(1, 254, K)
    
    # ===== PSO PARAMETERS =====
    # w: Inertia weight - điều chỉnh tác động của vận tốc cũ lên vận tốc mới
    #    w cao: inertia lớn, particles di chuyển nhiều (exploration)
    #    w thấp: inertia nhỏ, particles di chuyển ít (exploitation)
    w = 0.7  # Giá trị chuẩn PSO (thường 0.6-0.8)
    # c1: Cognitive parameter - trọng số hướng về personal best của particle
    #     c1 cao: particle tập trung vào kinh nghiệm của nó
    c1 = 2.0
    # c2: Social parameter - trọng số hướng về global best của toàn swarm
    #     c2 cao: particle tập trung vào thông tin toàn swarm
    c2 = 2.0
    
    # ===== PERSONAL BEST =====
    # Theo Algorithm 1, Line 6: p0_besti ← x0_i
    # pbest khởi tạo = position ban đầu (KHÔNG constraint)
    pbest = particles.copy()
    # pbest_scores: lưu score (fitness) của pbest
    # Tính fitness cho tất cả particles (dengan constraint khi eval)
    pbest_scores = np.array([objective(enforce_threshold_constraints(p)) for p in pbest])
    
    # ===== GLOBAL BEST =====
    # Theo Algorithm 1, Line 8: Apply Eq. (2) to find g0_best
    # gbest = position của best particle (pbest với minimum fitness)
    # gbest_idx: chỉ số particle có best score toàn swarm
    valid_scores = [(i, s) for i, s in enumerate(pbest_scores) if s is not None]
    if not valid_scores:
        # Nếu không có valid score, return empty
        return [], None
    gbest_idx = min(valid_scores, key=lambda x: x[1])[0]
    # gbest: position của best particle (Eq 2)
    gbest = pbest[gbest_idx].copy()
    # gbest_score: score của best particle
    gbest_score = pbest_scores[gbest_idx]
    
    # ===== VÒNG LẶP CHÍNH: CẬP NHẬT PARTICLES =====
    for iter in range(iters):
        # ===== CẬP NHẬT INERTIA WEIGHT =====
        # Linearly decrease inertia: w giảm từ 0.7 -> 0.7*0 = 0 qua các iteration
        # Tác dụng: từ từ chuyển từ exploration (early) sang exploitation (late)
        w_iter = w * (1 - iter / max(1, iters - 1))
        
        # ===== CẬP NHẬT VELOCITY (VECTƠ HÓA) =====
        # Sinh random numbers r1, r2 cho mỗi dimension (vectorized)
        # Shape: (pop_size, K) - mỗi phần tử random từ [0, 1)
        r1 = rng.random(particles.shape)  # random matrix
        r2 = rng.random(particles.shape)  # random matrix
        # Công thức velocity update PSO:
        # v_new = w*v_old + c1*r1*(pbest - x) + c2*r2*(gbest - x)
        # Ba thành phần:
        # 1. w_iter * velocities: inertia (vận tốc cũ)
        # 2. c1 * r1 * (pbest - particles): cognitive (về personal best)
        # 3. c2 * r2 * (gbest - particles): social (về global best)
        velocities = (
            w_iter * velocities  # Inertia term
            + c1 * r1 * (pbest - particles)  # Cognitive term
            + c2 * r2 * (gbest - particles)  # Social term
        )
        # Tất cả phép toán này là element-wise (vectorized, không vòng lặp)
        
        # ===== CẬP NHẬT POSITION (VỊ TRÍ) =====
        # Công thức PSO: x_new = x_old + v_new (Eq. 4)
        particles = particles + velocities
        
        # ===== ENFORCE CONSTRAINTS TRÊN TỪNG PARTICLE =====
        # Vòng lặp từng particle để constraint
        for i in range(pop_size):
            # Clip về range [1, 254]
            particles[i] = np.clip(particles[i], 1, 254)
            # Sort để đảm bảo thứ tự tăng dần
            particles[i] = np.sort(particles[i])  # Ensure increasing order
            # Enforce tất cả constraints (unique, min_gap, etc)
            particles[i] = enforce_threshold_constraints(particles[i])
        
        # ===== ĐÁNH GIÁ TOÀN PARTICLES =====
        # Tính fitness (objective score) cho tất cả particles sau khi cập nhật
        scores = np.array([objective(enforce_threshold_constraints(p)) for p in particles], dtype=object)
        
        # ===== CẬP NHẬT PERSONAL BEST =====
        # Theo Algorithm 1, Line 15-17
        # if f(x(t)_i) < f(pbest(t-1)_i) then pbest(t)_i = x(t)_i (Eq. 1)
        for i in range(pop_size):
            if scores[i] is not None and pbest_scores[i] is not None:
                if scores[i] < pbest_scores[i]:
                    pbest[i] = particles[i].copy()
                    pbest_scores[i] = scores[i]
            elif scores[i] is not None and pbest_scores[i] is None:
                pbest[i] = particles[i].copy()
                pbest_scores[i] = scores[i]
        
        # ===== CẬP NHẬT GLOBAL BEST =====
        # Theo Algorithm 1, Line 19: Apply Eq. (2) to find g(t)_best
        # g(t)_best = x* | f(x*) = min_{i,k}(f(x(k)_i))
        # Tìm best particle trong current pbest scores
        valid_pbest = [(i, s) for i, s in enumerate(pbest_scores) if s is not None]
        if valid_pbest:
            min_idx = min(valid_pbest, key=lambda x: x[1])[0]
            # Nếu best score hiện tại tốt hơn global best
            if gbest_score is None or pbest_scores[min_idx] < gbest_score:
                # Cập nhật global best (Eq. 2)
                gbest = pbest[min_idx].copy()
                gbest_score = pbest_scores[min_idx]
    # end iteration loop
    
    # ===== TRẢ VỀ KẾT QUẢ =====
    # Constraint global best một lần nữa để đảm bảo
    if gbest_score is None:
        # Không tìm được valid threshold, return empty
        return [], None
    final_thresholds = enforce_threshold_constraints(gbest)
    # Chuyển từ float về int (làm tròn)
    return [int(t) for t in final_thresholds], float(gbest_score)