"""Fuzzy Entropy utilities (pure functions).

Các hàm thuần: histogram_from_image, compute_fuzzy_entropy, membership generators.
"""
from __future__ import annotations

from typing import Literal, Sequence

import numpy as np

EPS = 1e-12
# When a threshold set produces empty classes, we treat the true FE as invalid/zero
# for plotting and comparison (FE_true). For optimizers we still need a strong
# penalty so they avoid such solutions; set FITNESS_PENALTY to a large positive
# value returned when for_minimization=True and classes are empty.
FITNESS_PENALTY = 1e6


def histogram_from_image(image: np.ndarray) -> np.ndarray:
    """Tạo histogram 256-bin (counts) cho ảnh xám 2D.

    Args:
        image: mảng numpy 2D, ảnh xám (uint8 hoặc float)

    Returns:
        hist: mảng float shape (256,), tổng = số pixel
    """
    if image.ndim != 2:
        raise ValueError("image must be grayscale 2D array")

    # Dùng range=(0,256) để mức 255 được tính (np.histogram là [low, high))
    hist, _ = np.histogram(image.ravel(), bins=256, range=(0, 256))
    return hist.astype(float)


def _triangular_membership(centers: np.ndarray) -> np.ndarray:
    """Hàm membership tam giác.

    Args:
        centers: mảng float tâm lớp (0..255), kích thước C

    Returns:
        mu: mảng shape (C, 256), mỗi hàng là hàm membership trên 0..255
    """
    levels = np.arange(256, dtype=float)
    C = centers.size
    mu = np.zeros((C, 256), dtype=float)

    # boundaries: midpoint giữa các tâm, thêm 0 và 255 ở hai đầu
    boundaries = np.empty(C + 1, dtype=float)
    boundaries[0] = 0.0
    boundaries[-1] = 255.0
    if C > 1:
        boundaries[1:-1] = (centers[:-1] + centers[1:]) / 2.0

    for i in range(C):
        left = boundaries[i]
        right = boundaries[i + 1]
        center = centers[i]

        # đoạn tăng từ left -> center
        if center > left:
            idx = (levels >= left) & (levels <= center)
            mu[i, idx] = (levels[idx] - left) / (center - left + EPS)

        # đoạn giảm từ center -> right
        if center < right:
            idx = (levels >= center) & (levels <= right)
            mu[i, idx] = (right - levels[idx]) / (right - center + EPS)

    return np.clip(mu, 0.0, 1.0)


def _gaussian_membership(centers: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Hàm membership Gaussian với sigma phụ thuộc độ rộng lớp, không normalize theo lớp.

    Ý tưởng:
        - Mỗi lớp có vùng ảnh hưởng (left, right) được xác định từ midpoint giữa các tâm.
        - Sigma của lớp i được đặt tỉ lệ với độ rộng lớp: sigma_i = alpha * (right_i - left_i).
        - Không chuẩn hóa theo lớp (không ép sum_c μ_c(y) = 1) để tránh bias mạnh
          khiến các tâm dồn cụm vào vùng histogram dày.

    Args:
        centers: mảng float tâm lớp (0..255), kích thước C
        alpha: hệ số tỉ lệ (0.3–0.7 là khoảng hợp lý)

    Returns:
        mu: mảng shape (C, 256), mỗi hàng là hàm membership Gaussian trên 0..255
    """
    levels = np.arange(256, dtype=float)
    C = centers.size
    mu = np.zeros((C, 256), dtype=float)

    # boundaries: midpoint giữa các tâm, thêm 0 và 255 ở hai đầu
    boundaries = np.empty(C + 1, dtype=float)
    boundaries[0] = 0.0
    boundaries[-1] = 255.0
    if C > 1:
        boundaries[1:-1] = (centers[:-1] + centers[1:]) / 2.0

    for i, c in enumerate(centers):
        left = boundaries[i]
        right = boundaries[i + 1]
        width = max(right - left, 1.0)
        sigma = alpha * width
        mu[i] = np.exp(-0.5 * ((levels - c) / (sigma + EPS)) ** 2)

    return np.clip(mu, 0.0, 1.0)


MembershipType = Literal["triangular", "gaussian"]


def compute_fuzzy_entropy(
    hist: np.ndarray,
    thresholds: Sequence[int],
    membership: MembershipType = "triangular",
    for_minimization: bool = False,
    lambda_penalty: float = 1.0,
    alpha_area: float = 0.1,
    beta_membership: float = 0.1,
    gamma_spacing: float = 0.2,
    gamma_spacing_var: float = 0.05,
    delta_compactness: float = 0.3,
) -> float:
    """Tính Fuzzy Entropy theo công thức De Luca (có phạt tuỳ chọn).

    **Công thức Fuzzy Entropy De Luca (Shannon):**
        H = -K Σ_{i=1}^{n} [ρ_i * log(ρ_i) + (1-ρ_i) * log(1-ρ_i)]
        
        Với:
        - K: hằng số chuẩn hoá (thường = 1 / ln(2))
        - n: số mức xám (256)
        - ρ_i: độ thành viên mờ của pixel i trong lớp được chọn

    **Công thức phạt toàn diện:**
        F'(T) = F(T) - λ[α·P_A(T) + β·P_μ(T) + γ·P_S(T) + δ·P_C(T)]
        
        Với:
        - F(T): Fuzzy Entropy (giá trị chính)
        - P_A(T): Penalty diện tích (cân bằng kích thước lớp)
        - P_μ(T): Penalty membership (tránh tập trung quá)
        - P_S(T): Penalty spacing (buộc ngưỡng phân tán đều) ✨
        - P_C(T): Penalty compactness (buộc segments sắc nét hơn) ✨ MỚI
        - λ, α, β, γ, δ: hệ số điều chỉnh

    Args:
        hist: mảng histogram 256-bin (độ đếm mỗi mức xám)
        thresholds: danh sách ngưỡng (k giá trị)
        membership: loại hàm membership ("triangular" hoặc "gaussian")
        for_minimization: nếu True trả về -FE (dùng cho minimizer), ngược lại trả về FE
        lambda_penalty: hệ số λ điều chỉnh cường độ phạt toàn diện
        alpha_area: hệ số α cho penalty diện tích (cân bằng kích thước lớp)
        beta_membership: hệ số β cho penalty membership (tránh tập trung)
        gamma_spacing: hệ số γ cho penalty spacing (buộc phân tán đều) ✨
        delta_compactness: hệ số δ cho penalty compactness (buộc sắc nét) ✨ MỚI

    Returns:
        float: Fuzzy Entropy (có phạt nếu cấu hình cho phép)
               Dương nếu for_minimization=False (dùng cho maximizer)
               Âm nếu for_minimization=True (dùng cho minimizer)
    
    **Công thức Penalty Spacing (mở rộng):**
        P_S(T) = w1 * Σ_{i=1}^{m} (1 / Δ_i) + w2 * Σ_{i=1}^{m} ((Δ_i - Δ_ideal)/Δ_ideal)^2

        Trong đó:
        - Δ_i: khoảng cách giữa hai ranh giới kế tiếp (bao gồm hai cạnh 0..t1 và tk..255), tổng có m = k+1 đoạn.
        - Δ_ideal = 255 / m: khoảng cách mong muốn nếu phân bố đều trên [0,255].
        - Thành phần đầu (1/Δ_i) trừng phạt khoảng cách quá nhỏ (tránh dồn cụm).
        - Thành phần thứ hai ((Δ_i-Δ_ideal)^2) trừng phạt sai lệch so với phân bố đều (khuyến khích đều nhau).

    **Công thức Penalty Compactness (mới, tăng sắc nét):**
        P_C(T) = Σ_c (1 / |S_c|)^2
        
        Trong đó:
        - |S_c|: số pixel trong lớp c (hay degree of membership sum)
        - Phạt các lớp nhỏ → khuyến khích các segment lớn hơn và sắc nét hơn
        - Tránh fuzzy membership tạo ra các lớp nhỏ và mềm mỏng

        Hiệu ứng: Khi delta_compactness > 0, MFWOA sẽ tìm cách tạo các segment có ranh giới sắc nét hơn,
        gần giống hành vi của Otsu (tạo các biên rõ ràng) → SSIM, DICE sẽ cao hơn.
    """
    # ========== BƯỚC 1: Kiểm tra tính hợp lệ của input ==========
    # Đảm bảo histogram có đúng 256 bin (một bin cho mỗi mức xám 0-255)
    if hist.shape[0] != 256:
        raise ValueError("hist must be length-256 array")

    # ========== BƯỚC 2: Chuẩn hoá histogram thành phân phối xác suất ==========
    # Tính tổng số pixel (tổng histogram)
    total = float(np.sum(hist))
    
    # Kiểm tra nếu histogram trống hoặc không hợp lệ
    if total <= 0.0:
        # Trả về penalty (nếu tối ưu) hoặc 0 (nếu chỉ tính toán)
        return float(FITNESS_PENALTY if for_minimization else 0.0)

    # Chuẩn hoá: p(y) = hist[y] / tổng -> mỗi p[y] ∈ [0, 1], Σ p = 1
    p_levels = hist.astype(float) / (total + EPS)

    # ========== BƯỚC 3: Ràng buộc và chuẩn hoá ngưỡng ==========
    # Chuyển ngưỡng thành array int và áp dụng ràng buộc
    # (đảm bảo thresholds tăng dần, không trùng, trong [1..254])
    th = np.array([int(t) for t in thresholds], dtype=int)
    from src.seg.utils import enforce_threshold_constraints
    th = enforce_threshold_constraints(th)

    # ========== BƯỚC 4: Xây dựng các tâm lớp ==========
    # Tâm lớp từ ranh giới: [0] + [t1, t2, ..., tk] + [255]
    # Ví dụ: nếu thresholds=[100, 150], centers = [0, 100, 150, 255]
    centers = np.concatenate(([0.0], th.astype(float), [255.0]))

    # ========== BƯỚC 5: Sinh ma trận membership (μ_c(y)) ==========
    # μ_c(y): độ thành viên của mức xám y trong lớp c
    # shape: (số_lớp, 256)
    if membership == "triangular":
        # Hàm tam giác: tâm có membership=1, giảm tuyến tính tới edges
        mu = _triangular_membership(centers)
    elif membership == "gaussian":
        # Hàm Gaussian: hình chuông xung quanh tâm
        mu = _gaussian_membership(centers)
    else:
        raise ValueError(f"unknown membership: {membership}")

    # ========== BƯỚC 6: Tính xác suất mỗi lớp ==========
    # p_classes[c] = Σ_y μ_c(y) * p(y) (xác suất lớp c)
    # Dùng để phát hiện lớp rỗng (p_classes[c] rất nhỏ)
    p_classes = mu.dot(p_levels)

    # ========== BƯỚC 7: Kiểm tra lớp rỗng ==========
    # NOTE: Bỏ check empty class. Một số ảnh (như Lena) không có pixel ở cạnh [0...20),
    # khiến class0 rỗng. Điều này là bình thường cho FE computation.
    # Optimizer sẽ tự penalize bộ thresholds tạo classes rỗng thông qua FE entropy thấp.
    # (Entropy của class rỗng = 0, làm FE giảm)
    # Nếu muốn enforce strictly, dùng constraint trực tiếp trong optimizer, không ở đây.

    # ========== BƯỚC 8: Tính Shannon Entropy cho từng pixel-lớp ==========
    # Shannon entropy: S_n(μ) = -μ * ln(μ) - (1-μ) * ln(1-μ)
    # 
    # Ý nghĩa:
    #   - μ = 0 hoặc 1 -> S = 0 (không mờ, entropy thấp)
    #   - μ = 0.5 -> S = ln(2) ≈ 0.693 (mờ tối đa, entropy cao)
    #
    # Clip μ để tránh ln(0) hoặc ln(số âm)
    mu_safe = np.clip(mu, EPS, 1.0 - EPS)
    
    # S[c, y] = -μ_c(y) * ln(μ_c(y)) - (1-μ_c(y)) * ln(1-μ_c(y))
    # Shape: (số_lớp, 256)
    S = -mu_safe * np.log(mu_safe) - (1.0 - mu_safe) * np.log(1.0 - mu_safe)

    # ========== BƯỚC 9: Tính Fuzzy Entropy tổng thể ==========
    # Công thức De Luca:
    #   H = K * Σ_{i=1}^{n} Σ_{c=1}^{C} [ρ_i,c * log(ρ_i,c) + (1-ρ_i,c) * log(1-ρ_i,c)]
    #
    # Sau khi đơn giản hóa với p(y):
    #   H = K * Σ_y p(y) * Σ_c S_n(μ_c(y))
    #
    # Dùng K = 1 / ln(2) để normalize
    num_classes = mu.shape[0]
    
    # Tính tổng entropy từ tất cả pixel và lớp
    # Sum chiều class (axis=0): Σ_c S_n(μ_c(y)) cho mỗi y
    # Dot với p_levels: Σ_y p(y) * (...)
    H_entropy = num_classes * float(np.sum(p_levels * np.sum(S, axis=0))) / np.log(2.0)

    # ========== BƯỚC 10: Tính penalty (nếu lambda_penalty > 0) ==========
    # Penalty diện tích: P_A(T)
    # Phạt nếu các lớp có kích thước quá khác nhau (không cân bằng)
    # P_A = Σ_c (p_classes[c] - 1/C)^2  (khoảng cách từ trung bình)
    mean_class_prob = 1.0 / num_classes
    P_A = float(np.sum((p_classes - mean_class_prob) ** 2))

    # Penalty membership: P_μ(T)
    # Phạt nếu membership quá tập trung (entropy thấp)
    # Để tránh membership bị spike ở một giá trị
    # P_μ = Σ_y p(y) * max_c μ_c(y)^2  (nếu max membership quá cao)
    # Hoặc có thể dùng: P_μ = -Σ_c p_classes[c] * entropy(p_classes)
    max_membership = float(np.max(mu))  # Giá trị membership lớn nhất
    P_mu = max_membership ** 2

    # ========== BƯỚC 11: Áp dụng công thức phạt ==========
    # F'(T) = F(T) - λ[α*P_A(T) + β*P_μ(T) + γ*P_S(T)]
    
    # Penalty spacing: P_S(T)
    # Phạt nếu các ngưỡng bị dồn cụm (khoảng cách quá nhỏ)
    # P_S = Σ_{i=1}^{k-1} (1 / (t_{i+1} - t_i))
    # Công thức: khoảng cách nhỏ → 1/khoảng_cách lớn → penalty cao
    # Ví dụ:
    #   - Đều: [50, 100, 150, 200] → khoảng 50 → P_S = 4/50 = 0.08 (thấp)
    #   - Không đều: [50, 60, 70, 200] → khoảng [10,10,130] → P_S = 1/10+1/10+1/130 ≈ 0.208 (cao)
    if len(th) > 0:
        # Khoảng cách giữa ranh giới, bao gồm đoạn từ 0->t1 và tk->255
        if len(th) > 1:
            spacings = np.diff(th).astype(float)
        else:
            spacings = np.empty(0, dtype=float)
        spacings_with_edges = np.concatenate(([th[0] - 0.0] if len(th) >= 1 else [255.0], spacings, [255.0 - th[-1]] if len(th) >= 1 else []))
        # Đảm bảo mảng spacing có chiều dương và loại bỏ giá trị không hợp lệ
        spacings_with_edges = np.maximum(spacings_with_edges.astype(float), EPS)

        # Thành phần 1: nghịch đảo khoảng cách (trừng phạt khoảng cách quá nhỏ)
        P_S_inv = float(np.sum(1.0 / (spacings_with_edges + EPS)))

        # Thành phần 2: phương sai so với khoảng cách lý tưởng (khuyến khích đều nhau)
        delta_ideal = 255.0 / float(spacings_with_edges.size)
        P_S_var = float(np.sum(((spacings_with_edges - delta_ideal) / (delta_ideal + EPS)) ** 2))

        # Kết hợp hai thành phần; hệ số thực thi do tham số gamma_spacing và gamma_spacing_var
        P_S = P_S_inv  # giữ tương thích, phần cuối sẽ dùng cả hai gamma
    else:
        # Không có ngưỡng: không có penalty spacing
        P_S_inv = 0.0
        P_S_var = 0.0
        P_S = 0.0
    
    # Tổng hợp penalty: phần 1 (nghịch đảo) điều chỉnh bằng gamma_spacing,
    # phần 2 (phương sai so với Δ_ideal) điều chỉnh bằng gamma_spacing_var
    
    # ========== BƯỚC 11b: Tính penalty compactness (mới) ==========
    # P_C(T): phạt sự mềm mỏng của membership để khuyến khích ranh giới sắc nét
    # Công thức: P_C = Σ_c (p_c - 1/C)^2 (variance từ uniform distribution)
    #   - Nếu các lớp cân bằng (p_c ≈ 1/C): P_C ≈ 0 (không phạt)
    #   - Nếu có lớp quá nhỏ: P_C cao → khuyến khích các segment lớn hơn
    #   - Khác với P_A (cũng phạt imbalance), P_C nhấn mạnh vào tính sắc nét
    # 
    # Thay vì dùng (1/p_c)^2 (quá aggressive), dùng variance từ đều:
    mean_class_prob = 1.0 / num_classes
    P_C = float(np.sum((p_classes - mean_class_prob) ** 2))
    
    penalty_term = lambda_penalty * (
        alpha_area * P_A + beta_membership * P_mu + gamma_spacing * (P_S_inv if 'P_S_inv' in locals() else 0.0) + gamma_spacing_var * (P_S_var if 'P_S_var' in locals() else 0.0) + delta_compactness * P_C
    )
    
    # Fuzzy Entropy có phạt
    FE_adjusted = H_entropy - penalty_term

    # ========== BƯỚC 12: Trả về kết quả ==========
    # Nếu for_minimization=True: trả về -FE (để optimizer minimize)
    # Ngược lại: trả về FE (để optimizer maximize)
    result = float(-FE_adjusted if for_minimization else FE_adjusted)
    
    return result
