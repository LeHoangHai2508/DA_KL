"""Simple MFWOA implementation (pure functions where possible).

This is a compact MFWOA variant supporting multiple tasks (different K). It keeps positions
in continuous [0,255] and enforces sorted unique thresholds.
"""
from __future__ import annotations

import numpy as np
from typing import Callable, List, Sequence, Tuple


def _enforce_mfwoa_constraints(vec: np.ndarray, min_gap: int = 1) -> np.ndarray:
    """Enforce constraints on thresholds: sorting, clipping, and minimum gap.
    
    Hàm này đảm bảo tất cả threshold tuân theo các ràng buộc:
    - Nằm trong khoảng [1, 254] (không được 0 hoặc 255)
    - Sắp xếp tăng dần (sorted)
    - Khoảng cách tối thiểu giữa các ngưỡng liên tiếp >= min_gap
    
    Args:
        vec: Array of threshold values (có thể unsorted, nằm ngoài phạm vi)
        min_gap: Minimum gap between consecutive thresholds (default 1, để tránh trùng)
    
    Returns:
        Constrained threshold array (sorted, clipped, enforced gap)
    """
    # Round and clip to [1, 254] - dùng numpy.clip để giới hạn giá trị
    # Từ phạm vi liên tục [0, 255] xuống [1, 254]
    vals = np.clip(vec, 1.0, 254.0)
    # Sort tăng dần: numpy.sort sắp xếp từ nhỏ đến lớn
    vals = np.sort(vals)
    
    # Enforce minimum gap between consecutive thresholds
    # Vòng lặp kiểm tra từng cặp ngưỡng kế tiếp và đảm bảo khoảng cách >= min_gap
    for i in range(1, len(vals)):
        # Nếu khoảng cách nhỏ hơn min_gap, tăng ngưỡng tiếp theo
        if vals[i] - vals[i - 1] < min_gap:
            # vals[i] = vals[i-1] + min_gap: đẩy vals[i] về phía trên
            vals[i] = vals[i - 1] + min_gap
    
    # Clip again in case enforcing gaps pushed values out of range
    # Sau khi enforce gap, có thể một vài giá trị vượt quá 254, nên clip lại
    vals = np.clip(vals, 1.0, 254.0)
    
    return vals


def continuous_to_thresholds(pos: np.ndarray, min_gap: int = 1) -> List[int]:
    """Convert continuous positions to discrete integer thresholds.
    
    Hàm chuyển đổi vị trí liên tục (từ optimizer) thành danh sách ngưỡng rời rạc (int).
    
    Args:
        pos: Continuous position vector (numpy array, có thể chứa số thực)
        min_gap: Minimum gap between consecutive thresholds (để enforce constraints)
    
    Returns:
        List of integer thresholds (đã constraint, sorted, và là kiểu int)
    """
    # Gọi _enforce_mfwoa_constraints để áp dụng tất cả constraints
    # Đảm bảo sorted, clipped, và gap enforcement
    arr = _enforce_mfwoa_constraints(pos, min_gap=min_gap)
    # Chuyển từ float -> int: int(round(x)) làm tròn x về số nguyên gần nhất
    # Ví dụ: 100.4 -> 100, 100.6 -> 101
    return [int(round(x)) for x in arr]


def mfwoa_optimize(
    hist: np.ndarray,
    K: int,
    pop_size: int = 100,
    iters: int = 500,
    objective: Callable[[np.ndarray], float] = None,
    rng: np.random.Generator = None,
) -> Tuple[List[int], float]:
    """Simplified single-task WOA (not full multifactorial) but structured so it can be extended.

    Hàm MFWOA chính: sử dụng Whale Optimization Algorithm để tối ưu vị trí ngưỡng (maximize).
    
    Thuật toán:
    1. Khởi tạo quần thể ngẫu nhiên (pop_size cá thể, mỗi cá thể là K ngưỡng)
    2. Vòng lặp iters lần: cập nhật vị trí dựa vào WOA mechanics (encircle, spiral)
    3. Theo dõi best solution và cập nhật
    4. Trả về best thresholds tìm được

    Args:
        hist: 256-bin histogram (từ ảnh input, không dùng trực tiếp nhưng để thống nhất signature)
        K: number of thresholds (số ngưỡng cần tìm)
        pop_size: population size (số lượng cá thể trong quần thể, thường 20-200)
        iters: number of iterations (số vòng lặp, thường 100-500)
        objective: function mapping position vector (len K) to score (higher better)
                   hàm này tính fitness cho một vị trí (objective function từ bên ngoài)
        rng: Random number generator (numpy.random.Generator, để random reproducible)
    
    Returns:
        Tuple: (best_thresholds_list_int, best_score_float)
    """
    # Kiểm tra xem rng có được truyền vào không
    # Nếu không, dùng default_rng() sử dụng system entropy (ngẫu nhiên không cố định)
    if rng is None:
        # Use system entropy by default (align behavior with other optimizers)
        rng = np.random.default_rng()
    # Kiểm tra objective function có được truyền không
    if objective is None:
        raise ValueError("objective must be provided")
    # Dimension của vị trí = số ngưỡng K
    dim = K
    # ===== KHỞI TẠO QUẦN THỂ =====
    # Sinh pop_size cá thể, mỗi cá thể là K vị trí ngẫu nhiên trong [1, 254]
    # rng.uniform(1.0, 254.0, size=(pop_size, dim)): sinh ma trận (pop_size, K) từ Uniform[1, 254)
    pop = rng.uniform(1.0, 254.0, size=(pop_size, dim))
    
    # ===== ĐÁNH GIÁ SƠ BỘ QUẦN THỂ =====
    # Tính fitness (objective score) cho tất cả cá thể
    # Mỗi cá thể được constraint trước khi đánh giá (sorted, clipped, gap enforce)
    scores = np.array([objective(_enforce_mfwoa_constraints(ind, min_gap=1)) for ind in pop])
    # Tìm cá thể tốt nhất (highest score) trong quần thể ban đầu
    best_idx = int(np.argmax(scores))  # argmax: vị trí của phần tử lớn nhất
    # Lưu lại best position và score
    best_pos = pop[best_idx].copy()  # copy() để tránh reference
    best_score = float(scores[best_idx])

    # ===== VÒNG LẶP CHÍNH: CẬP NHẬT QUẦN THỂ =====
    for t in range(iters):
        # WOA parameter 'a': giảm tuyến tính từ 2 -> 0 qua các iteration
        # Điều chỉnh mức độ exploration (high a) vs exploitation (low a)
        a = 2.0 * (1 - t / max(1, iters - 1))  # t=0: a=2, t=iters-1: a=0
        
        # Cập nhật từng cá thể trong quần thể
        for i in range(pop_size):
            # ===== SINH RANDOM PARAMETERS =====
            # WOA uses random numbers r1, r2, p để quyết định hành động
            r1 = rng.random()  # random in [0, 1)
            r2 = rng.random()  # random in [0, 1)
            # Tính các hệ số A, C dùng trong WOA update
            # A: controls how far the whale should move towards the prey (-a to a)
            A = 2 * a * r1 - a  # A ∈ [-a, a]
            # C: random weight cho lực kéo về best solution
            C = 2 * r2  # C ∈ [0, 2]
            # Random probability quyết định kiểu update (encircle vs spiral)
            p = rng.random()  # p ∈ [0, 1)
            
            # ===== QUYẾT ĐỊNH HÀNH ĐỘNG (Encircle vs Spiral) =====
            if p < 0.5:
                # ENCIRCLE PREY: di chuyển theo hướng best solution
                if abs(A) < 1:
                    # ===== EXPLOITATION: Encircle best prey =====
                    # Di chuyển gần về best solution (|A| < 1: exploitation phase)
                    # D = |C * best_pos - pop[i]|: khoảng cách cần di chuyển
                    D = abs(C * best_pos - pop[i])
                    # new_pos = best_pos - A * D: công thức WOA encircling
                    new_pos = best_pos - A * D
                else:
                    # ===== EXPLORATION: Search via random agent =====
                    # Chọn một cá thể ngẫu nhiên (không phải best) để follow
                    # Giúp tăng tính explore và tránh cực trị cục bộ
                    rand_idx = rng.integers(pop_size)  # random index from 0 to pop_size-1
                    X_rand = pop[rand_idx]
                    D = abs(C * X_rand - pop[i])
                    new_pos = X_rand - A * D
            else:
                # SPIRAL UPDATE: xoắn ốc di chuyển về best solution
                # Cơ chế này giúp di chuyển mượt mà, không bước nhảy
                # Sử dụng exponential spiral: x = r*e^(b*l)*cos(2πl) + center
                b = 1.0  # Shape parameter của spiral (logarithmic spiral)
                # Tính khoảng cách từ whale hiện tại đến best solution
                distance = abs(best_pos - pop[i])
                # Random random number trong [-1, 1] điều chỉnh độ xoắn
                l = rng.uniform(-1, 1)  # l ∈ [-1, 1]
                # Công thức spiral: moving in spiral pattern around best_pos
                # np.exp(b*l): exponential decay
                # np.cos(2*π*l): oscillation
                new_pos = distance * np.exp(b * l) * np.cos(2 * np.pi * l) + best_pos
            
            # ===== CONSTRAINT VÀ ĐÁNH GIÁ =====
            # Sửa lại new_pos để tuân theo constraints (sort, clip, gap)
            new_pos = _enforce_mfwoa_constraints(new_pos, min_gap=1)
            # Tính fitness của new position
            new_score = objective(new_pos)
            
            # ===== CẬP NHẬT GREEDY =====
            # Nếu new solution tốt hơn current individual, thay thế
            if new_score > scores[i]:
                pop[i] = new_pos  # Cập nhật position của cá thể i
                scores[i] = new_score  # Cập nhật score của cá thể i
            
            # Kiểm tra xem new solution có tốt hơn global best không
            if new_score > best_score:
                best_score = new_score  # Cập nhật best score
                best_pos = new_pos.copy()  # Cập nhật best position (lưu copy)
        # end population loop (for i in range(pop_size))
    # end iteration loop (for t in range(iters))
    
    # ===== TRẢ VỀ KẾT QUẢ =====
    # Chuyển best_pos (liên tục) thành danh sách int thresholds
    # continuous_to_thresholds: làm tròn, constraint, và chuyển kiểu
    return continuous_to_thresholds(best_pos, min_gap=1), best_score
