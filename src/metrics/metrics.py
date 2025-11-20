"""Image quality metrics: PSNR, SSIM, FSIM (pure functions).

Simple, dependency-light implementations suitable for grayscale images.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, sobel


def psnr(original: np.ndarray, compared: np.ndarray, data_range: float = 255.0) -> float:
    """Compute Peak Signal-to-Noise Ratio between two images.
    
    Args:
        original: Original image array
        compared: Compared image array
        data_range: Dynamic range of the images (default 255 for uint8)
    
    Returns:
        PSNR value in dB
    """
    if original.shape != compared.shape:
        raise ValueError("shapes must match")
    mse = np.mean((original.astype(np.float64) - compared.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    return 10.0 * np.log10((data_range ** 2) / mse)


def ssim(original: np.ndarray, compared: np.ndarray, data_range: float = 255.0, gaussian_sigma: float = 1.5) -> float:
    """Compute a single-scale SSIM index for two grayscale images.

    Reference: Wang et al., 2004. Simplified implementation using gaussian filter.
    
    Args:
        original: Original image array
        compared: Compared image array
        data_range: Dynamic range of the images (default 255)
        gaussian_sigma: Sigma for Gaussian weighting
    
    Returns:
        SSIM value in [0, 1]
    """
    if original.shape != compared.shape:
        raise ValueError("shapes must match")
    orig = original.astype(np.float64)
    comp = compared.astype(np.float64)

    K1 = 0.01
    K2 = 0.03
    L = data_range
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2

    # gaussian weighted means
    mu1 = gaussian_filter(orig, gaussian_sigma)
    mu2 = gaussian_filter(comp, gaussian_sigma)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = gaussian_filter(orig * orig, gaussian_sigma) - mu1_sq
    sigma2_sq = gaussian_filter(comp * comp, gaussian_sigma) - mu2_sq
    sigma12 = gaussian_filter(orig * comp, gaussian_sigma) - mu1_mu2

    sigma1_sq = sigma1_sq.clip(min=0)
    sigma2_sq = sigma2_sq.clip(min=0)

    num = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    den = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    ssim_map = num / (den + 1e-12)
    return float(np.mean(ssim_map))


def fsim(original: np.ndarray, compared: np.ndarray, T1: float = 0.85, T2: float = 160.0) -> float:
    """Compute Feature Similarity Index Measure (FSIM) for two grayscale images.
    
    FSIM measures structural similarity by comparing:
    1. Phase Congruency (PC) - local phase alignment
    2. Gradient Magnitude (G) - local contrast/edge strength
    
    Args:
        original: Original image array (reference)
        compared: Compared image array (distorted)
        T1: Constant to avoid division by zero in phase similarity (default 0.85)
        T2: Constant to avoid division by zero in gradient similarity (default 160.0)
    
    Returns:
        FSIM value in [0, 1], higher is better
        
    Reference: L. Zhang et al., "FSIM: A Feature Similarity Index for Image Quality Assessment"
    """
    if original.shape != compared.shape:
        raise ValueError("shapes must match")
    
    orig = original.astype(np.float64)
    comp = compared.astype(np.float64)
    
    # Compute gradients using Sobel operator
    # Returns gradient magnitude in x and y directions
    gx1 = sobel(orig, axis=1)  # Gradient in x direction (reference)
    gy1 = sobel(orig, axis=0)  # Gradient in y direction (reference)
    G1 = np.sqrt(gx1**2 + gy1**2)  # Gradient magnitude (reference)
    
    gx2 = sobel(comp, axis=1)  # Gradient in x direction (compared)
    gy2 = sobel(comp, axis=0)  # Gradient in y direction (compared)
    G2 = np.sqrt(gx2**2 + gy2**2)  # Gradient magnitude (compared)
    
    # Compute phase congruency (simplified as normalized gradients)
    # In full FSIM, PC is computed from phase information via Fourier-based analysis
    # For practical grayscale implementation, we use gradient orientation alignment
    # PC represents how well the local phase/orientation aligns between images
    
    # Compute phase congruency as dot product of normalized gradients
    # Avoid division by zero
    eps = 1e-10
    mag1 = np.sqrt(gx1**2 + gy1**2) + eps
    mag2 = np.sqrt(gx2**2 + gy2**2) + eps
    
    # Normalized gradients (direction vectors)
    gx1_norm = gx1 / mag1
    gy1_norm = gy1 / mag1
    gx2_norm = gx2 / mag2
    gy2_norm = gy2 / mag2
    
    # Phase congruency: dot product of normalized gradients
    # Value in [-1, 1]; 1 means perfect alignment
    PC1 = gx1_norm * gx2_norm + gy1_norm * gy2_norm
    PC1 = (PC1 + 1.0) / 2.0  # Map from [-1, 1] to [0, 1]
    
    # Simplified version: use normalized gradient magnitudes directly
    PC1_alt = (G1 / (G1.max() + eps)) if G1.max() > 0 else np.zeros_like(G1)
    PC2_alt = (G2 / (G2.max() + eps)) if G2.max() > 0 else np.zeros_like(G2)
    
    # Phase similarity map: measures how similar the local structures are
    S_PC = (2.0 * PC1_alt * PC2_alt + T1) / (PC1_alt**2 + PC2_alt**2 + T1)
    
    # Gradient similarity: measures edge strength similarity
    G1_norm = G1 / (G1.max() + eps) if G1.max() > 0 else G1
    G2_norm = G2 / (G2.max() + eps) if G2.max() > 0 else G2
    S_G = (2.0 * G1_norm * G2_norm + T2) / (G1_norm**2 + G2_norm**2 + T2)
    
    # Importance map: use max of normalized gradients
    P_C = np.maximum(PC1_alt, PC2_alt)
    
    # Global FSIM: weighted average of similarities
    numerator = np.sum(S_PC * S_G * P_C)
    denominator = np.sum(P_C) + eps
    
    fsim_value = numerator / denominator
    return float(np.clip(fsim_value, 0.0, 1.0))


def dice_coefficient(y_true: np.ndarray, y_pred: np.ndarray, smooth: float = 1e-6) -> float:
    """Tính Dice Similarity Coefficient cho ảnh nhị phân.
    
    Args:
        y_true: Ground truth binary mask (0 hoặc 1)
        y_pred: Predicted binary mask (0 hoặc 1)
        smooth: Hệ số làm mịn để tránh chia cho 0
    
    Returns:
        Dice score trong khoảng [0, 1]
    """
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true và y_pred phải cùng kích thước")
    
    # Đảm bảo ảnh là kiểu boolean
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)
    
    # Tính phần giao (intersection)
    intersection = np.sum(y_true & y_pred)
    
    # Tính tổng số pixel '1' của mỗi ảnh
    total_pixels = np.sum(y_true) + np.sum(y_pred)
    
    # Áp dụng công thức Dice: (2 * Giao) / (Tổng)
    dice = (2.0 * intersection + smooth) / (total_pixels + smooth)
    
    return float(dice)


def dice_macro(pred_labels: np.ndarray, gt_labels: np.ndarray, labels=None, eps: float = 1e-9) -> float:
    """Tính Dice Similarity Coefficient cho phân đoạn đa lớp (macro-average).
    
    Phương pháp: Tính Dice cho từng lớp riêng biệt, sau đó lấy trung bình.
    
    Args:
        pred_labels: Ảnh nhãn dự đoán (HxW), kiểu int, mỗi pixel là ID lớp
        gt_labels: Ảnh nhãn ground truth (HxW), kiểu int
        labels: Danh sách các lớp cần tính; None = tự động lấy từ union của pred và gt
        eps: Epsilon để tránh chia cho 0
    
    Returns:
        Mean Dice score trong khoảng [0, 1]
        Trả về NaN nếu không có lớp hợp lệ nào
    """
    if pred_labels.shape != gt_labels.shape:
        raise ValueError("dice_macro: pred và gt phải cùng kích thước")
    
    # Tự động xác định các lớp nếu không được chỉ định
    if labels is None:
        labels = np.unique(np.concatenate([np.unique(gt_labels), np.unique(pred_labels)]))
    
    dices = []
    for c in labels:
        # Tạo binary mask cho lớp c
        pred_mask = (pred_labels == c)
        gt_mask = (gt_labels == c)
        
        # Tính tổng pixel của cả hai mask
        denom = pred_mask.sum() + gt_mask.sum()
        
        # Bỏ qua lớp không xuất hiện ở cả hai ảnh
        if denom == 0:
            continue
        
        # Tính intersection
        inter = (pred_mask & gt_mask).sum()
        
        # Dice cho lớp c
        dice_c = (2.0 * inter + eps) / (denom + eps)
        dices.append(dice_c)
    
    # Trả về trung bình hoặc NaN nếu không có lớp hợp lệ
    return float(np.mean(dices)) if len(dices) > 0 else np.nan


def dice_per_class(pred_labels: np.ndarray, gt_labels: np.ndarray, labels=None, eps: float = 1e-9) -> dict:
    """Tính Dice cho từng lớp riêng biệt (để phân tích chi tiết).
    
    Args:
        pred_labels: Ảnh nhãn dự đoán
        gt_labels: Ảnh nhãn ground truth
        labels: Danh sách các lớp; None = tự động
        eps: Epsilon
    
    Returns:
        Dictionary {class_id: dice_score}
    """
    if pred_labels.shape != gt_labels.shape:
        raise ValueError("pred và gt phải cùng kích thước")
    
    if labels is None:
        labels = np.unique(np.concatenate([np.unique(gt_labels), np.unique(pred_labels)]))
    
    dice_dict = {}
    for c in labels:
        pred_mask = (pred_labels == c)
        gt_mask = (gt_labels == c)
        denom = pred_mask.sum() + gt_mask.sum()
        
        if denom == 0:
            dice_dict[int(c)] = np.nan  # Lớp không tồn tại
        else:
            inter = (pred_mask & gt_mask).sum()
            dice_dict[int(c)] = float((2.0 * inter + eps) / (denom + eps))
    
    return dice_dict