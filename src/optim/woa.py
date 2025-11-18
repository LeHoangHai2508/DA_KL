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
        # t=0: a=2.0 (exploration), t=iters-1: a≈0 (exploitation)
        a = 2 * (1 - t / max(1, iters - 1))
        
        # ===== CẬP NHẬT TỪNG CÁ THỂ =====
        # For each whale (cá thể) trong quần thể
        for i in range(pop_size):
            # Constraint whale[i] và tính score
            current = enforce_threshold_constraints(whales[i])
            score = objective(current)
            
            # ===== CẬP NHẬT GLOBAL BEST =====
            # Kiểm tra nếu current solution tốt hơn global best
            if score < best_score:
                best_score = score
                best_whale = current.copy()
            
            # ===== SINH RANDOM PARAMETERS =====
            # Các tham số ngẫu nhiên điều chỉnh hành động di chuyển
            r = rng.random()  # r ∈ [0, 1): dùng cho A
            # A: Coefficient vector điều chỉnh bước nhảy về/từ prey
            A = 2 * a * r - a  # A ∈ [-a, a]
            # C: Coefficient vector random weight
            C = 2 * r  # C ∈ [0, 2]
            # l: random parameter cho spiral shape
            l = rng.uniform(-1, 1)  # l ∈ [-1, 1]
            # p: probability quyết định strategy (encircle vs spiral)
            p = rng.random()  # p ∈ [0, 1)
            
            # ===== QUYẾT ĐỊNH STRATEGY =====
            if p < 0.5:
                # ENCIRCLING PREY (50% probability)
                # Kỹ thuật đốn mồi: di chuyển xung quanh best solution
                if abs(A) < 1:
                    # ===== EXPLOITATION: Encircling best prey =====
                    # Khi |A| < 1: exploitation phase (khai thác best solution)
                    # Tính khoảng cách từ current đến best_whale
                    D = np.abs(C * best_whale - current)
                    # new_pos = best_whale - A * D: công thức encircling
                    new_pos = best_whale - A * D
                else:
                    # ===== EXPLORATION: Search via random whale =====
                    # Khi |A| >= 1: exploration phase (khám phá không gian)
                    # Chọn một cá thể ngẫu nhiên để follow (không phải best)
                    random_idx = rng.integers(pop_size)  # random index [0, pop_size)
                    # Constraint random whale trước use
                    random_whale = enforce_threshold_constraints(whales[random_idx])
                    # Tính khoảng cách đến random whale
                    D = np.abs(C * random_whale - current)
                    # Cập nhật vị trí
                    new_pos = random_whale - A * D
            else:
                # SPIRAL UPDATING (50% probability)
                # Xoắn ốc di chuyển về best solution (smooth, không bước nhảy)
                # Tính khoảng cách từ current đến best
                D = np.abs(best_whale - current)
                # Tính spiral component: D * e^(b*l) * cos(2π*l)
                # Exponential decay * oscillation = xoắn ốc logarit
                spiral = D * np.exp(b * l) * np.cos(2 * np.pi * l)
                # new_pos = best_whale + spiral: cộng spiral vào best position
                new_pos = best_whale + spiral
            
            # ===== ADAPTIVE STEP SIZE REDUCTION =====
            # Giảm bước di chuyển qua các iteration (từ từ hội tụ)
            # step_scale giảm từ 1.0 (iteration đầu) đến 0.1 (iteration cuối)
            step_scale = 1.0 - 0.9 * (t / max(1, iters - 1))
            # Áp dụng step scale: blending giữa current và new_pos
            # Công thức: current + step_scale * (new_pos - current)
            # Tác dụng: giảm bước nhảy, tăng ổn định
            new_pos = current + step_scale * (new_pos - current)
            
            # ===== CONSTRAINT VÀ ENFORCE RULES =====
            # Clip vào range [1, 254]
            new_pos = np.clip(new_pos, 1, 254)
            # Sort để đảm bảo thứ tự tăng dần
            new_pos = np.sort(new_pos)  # Ensure increasing order
            # Enforce tất cả constraints (unique, min_gap, etc)
            new_pos = enforce_threshold_constraints(new_pos)
            # Cập nhật whale[i] với constrained new_pos
            whales[i] = new_pos
    # end iteration loop
    
    # ===== TRẢ VỀ KẾT QUẢ =====
    # Constraint best_whale một lần nữa để đảm bảo
    final_thresholds = enforce_threshold_constraints(best_whale)
    # Chuyển từ float về int (làm tròn)
    return [int(t) for t in final_thresholds], float(best_score)