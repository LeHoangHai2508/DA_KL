"""Standard Whale Optimization Algorithm implementation.

Whale Optimization Algorithm (WOA) - một thuật toán meta-heuristic
dựa trên hành động kiếm ăn của cá voi Humpback.
"""
from typing import Tuple, Callable
import numpy as np


def woa_optimize(hist: np.ndarray, K: int, pop_size: int, iters: int,
                objective: Callable[[np.ndarray], float]) -> Tuple[list[int], float]:
    """Standard WOA optimization for thresholding.
    
    Whale Optimization Algorithm để tối ưu ngưỡng phân đoạn ảnh.
    Khác với MFWOA: WOA là minimizer (minimize loss), có adaptive step size.
    
    Args:
        hist: Image histogram (256-bin, dùng để quyết định dimension)
        K: Number of thresholds (số ngưỡng cần tìm, = dimension của bài toán)
        pop_size: Population size (số lượng cá thể, thường 20*K trở lên)
        iters: Number of iterations (số vòng lặp tối ưu)
        objective: Objective function to minimize (hàm fitness, thấp hơn càng tốt)
    
    Returns:
        Tuple: (Best thresholds list of int, best score float)
    """
    # Import constraint function để enforce threshold rules
    from src.seg.utils import enforce_threshold_constraints
    
    # ===== KHỞI TẠO QUẦN THỂ =====
    # Tạo ma trận pop_size x K để lưu vị trí của tất cả cá thể
    # dtype=np.float32: dùng float32 để tiết kiệm memory
    whales = np.zeros((pop_size, K), dtype=np.float32)
    # Tạo random number generator không có seed cố định (mỗi lần chạy khác nhau)
    rng = np.random.default_rng()
    
    # Khởi tạo mỗi cá thể (whale) bằng K ngưỡng ngẫu nhiên
    for i in range(pop_size):
        # ===== RANDOM INITIALIZATION =====
        # Sinh K số ngẫu nhiên từ Uniform(1, 254)
        # rng.uniform(1, 254, K): trả về mảng 1D có K phần tử
        thresholds = np.sort(rng.uniform(1, 254, K))  # Sort để đảm bảo thứ tự tăng
        # Gán vào hàng i của ma trận whales
        whales[i] = thresholds
    
    # ===== ĐÁNH GIÁ SƠ BỘ QUẦN THỂ =====
    # Tính fitness (objective score) cho tất cả cá thể ban đầu
    best_whale = None  # Biến lưu best solution (thresholds tốt nhất)
    best_score = float('inf')  # Biến lưu best score (khởi tạo = infinity, minimize)
    for i in range(pop_size):
        # Constraint whale[i] để đảm bảo tuân theo rules (sort, unique, range)
        current = enforce_threshold_constraints(whales[i])
        # Tính objective score cho constraint whale[i]
        score = objective(current)
        # Nếu score tốt hơn best_score (thấp hơn), cập nhật best
        if score < best_score:
            best_score = score
            best_whale = current.copy()  # copy() tránh reference
    
    # ===== WOA PARAMETERS =====
    # b: spiral parameter (logarithmic spiral shape parameter)
    b = 1  # Giá trị chuẩn WOA, điều chỉnh hình dáng xoắn ốc
    
    # ===== VÒNG LẶP CHÍNH: CẬP NHẬT QUẦN THỂ =====
    for t in range(iters):
        # ===== CẬP NHẬT THAM SỐ WOA =====
        # Update a linearly from 2 to 0 qua các iteration (exploration -> exploitation)
        # a = 2 - (2/iters) * t: giảm từ 2 xuống 0
        # t=0: a=2.0 (exploration), t=iters-1: a≈0 (exploitation)
        a = 2 * (1 - t / max(1, iters - 1))
        
        # ===== CẬP NHẬT TỪNG CÁ THỂ =====
        # For each whale (cá thể) trong quần thể
        new_whales = np.copy(whales)  # Store new positions
        
        for i in range(pop_size):
            # ===== SINH RANDOM PARAMETERS (theo Algorithm) =====
            # Các tham số ngẫu nhiên điều chỉnh hành động di chuyển
            r1 = rng.random()  # r ∈ [0, 1): dùng cho A
            r2 = rng.random()  # r ∈ [0, 1): dùng cho C
            # A: Coefficient vector (Eq 2.3): A = 2a*r1 - a, A ∈ [-a, a]
            A = 2 * a * r1 - a
            # C: Coefficient vector (Eq 2.4): C = 2*r2, C ∈ [0, 2]
            C = 2 * r2
            # l: random parameter cho spiral shape
            l = rng.uniform(-1, 1)  # l ∈ [-1, 1]
            # p: probability quyết định strategy (encircle vs spiral)
            p = rng.random()  # p ∈ [0, 1)
            
            # Get current whale position
            current = whales[i]
            
            # ===== QUYẾT ĐỊNH STRATEGY (theo Algorithm) =====
            # Eq 2.6: X(t+1) phụ thuộc vào p và |A|
            if p < 0.5:
                # ENCIRCLING PREY (50% probability)
                # Eq 2.1-2.2: D = |C * X* - X|, X(t+1) = X* - A*D
                if abs(A) < 1:
                    # ===== EXPLOITATION: Encircling best prey (Eq 2.1-2.2) =====
                    # Khi |A| < 1: khai thác best solution (tập trung)
                    # D = |C * X* - X| (Eq 2.1)
                    D = np.abs(C * best_whale - current)
                    # X(t+1) = X* - A*D (Eq 2.2)
                    new_pos = best_whale - A * D
                else:
                    # ===== EXPLORATION: Search via random whale (Eq 2.7-2.8) =====
                    # Khi |A| >= 1: khám phá không gian (từ từ)
                    # Chọn một cá thể ngẫu nhiên để follow (không phải best)
                    random_idx = rng.integers(pop_size)
                    random_whale = whales[random_idx]
                    # D = |C * X_rand - X| (Eq 2.7)
                    D = np.abs(C * random_whale - current)
                    # X(t+1) = X_rand - A*D (Eq 2.8)
                    new_pos = random_whale - A * D
            else:
                # SPIRAL UPDATING (50% probability)
                # Eq 2.5: X(t+1) = D' * e^(b*l) * cos(2π*l) + X*
                # Xoắn ốc di chuyển về best solution
                D_prime = np.abs(best_whale - current)
                # Spiral equation (Eq 2.5)
                new_pos = D_prime * np.exp(b * l) * np.cos(2 * np.pi * l) + best_whale
            
            # ===== CONSTRAINT VÀ ENFORCE RULES =====
            # Mandatory move: cập nhật vị trí bất kể tốt hay xấu (WOA pure strategy)
            # Clip vào range [1, 254]
            new_pos = np.clip(new_pos, 1, 254)
            # Enforce tất cả constraints (sort, unique, min_gap, etc)
            new_pos = enforce_threshold_constraints(new_pos)
            # Cập nhật vị trí mới
            new_whales[i] = new_pos
        
        # ===== UPDATE POPULATION AFTER ALL MOVES =====
        whales = new_whales
        
        # ===== EVALUATE ALL WHALES & UPDATE BEST =====
        # Tính fitness cho tất cả cá thể APRÈS khi cập nhật hết
        for i in range(pop_size):
            current = whales[i]
            score = objective(current)
            
            # Cập nhật global best nếu tốt hơn
            if score < best_score:
                best_score = score
                best_whale = current.copy()
    # end iteration loop
    
    # ===== TRẢ VỀ KẾT QUẢ =====
    # Constraint best_whale một lần nữa để đảm bảo
    final_thresholds = enforce_threshold_constraints(best_whale)
    # Chuyển từ float về int (làm tròn)
    return [int(t) for t in final_thresholds], float(best_score)