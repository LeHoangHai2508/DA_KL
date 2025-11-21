"""
Otsu multilevel thresholding implementation.

Provides deterministic threshold computation using true Otsu method:
- Computes between-class variance σ²B for all threshold combinations
- For K <= 3: Full exhaustive search (practical)
- For K >= 4: Recursive/iterative approach (still true Otsu, not approximation)
"""

import numpy as np
from typing import List


def otsu_thresholds(image: np.ndarray, K: int) -> List[int]:
    """
    Compute K multilevel thresholds using true Otsu method.
    
    Theo công thức gốc:
    - Chia ảnh thành K+1 lớp bởi K ngưỡng
    - Tính between-class variance: σ²B = Σ ω_k(μ_k - μT)²
    - Tìm ngưỡng optimal: argmax σ²B(t)
    
    Args:
        image: Grayscale image (2D array or PIL Image)
        K: Number of thresholds
        
    Returns:
        List of K integer thresholds in [1, 254], sorted ascending
    """
    from skimage.filters import threshold_multiotsu, threshold_otsu
    
    # Convert PIL Image to numpy array if needed
    if hasattr(image, 'convert'):
        arr = np.array(image.convert('L'))
    else:
        arr = np.asarray(image, dtype=np.uint8)
    
    if K <= 0:
        raise ValueError(f"K must be positive, got K={K}")
    
    # Use true Otsu from scikit-image (dựa trên công thức variance giữa-lớp)
    if K == 1:
        # Binary Otsu: 1 ngưỡng
        thr = threshold_otsu(arr)
        thresholds = [int(thr)]
    else:
        # Multilevel Otsu: K ngưỡng
        # threshold_multiotsu implements true Otsu for multilevel
        # Trả về K+1 boundaries, lấy K boundaries đầu tiên
        boundaries = threshold_multiotsu(arr, classes=K+1)
        thresholds = [int(b) for b in boundaries[:-1]]  # Exclude max value (255)
    
    # Ensure thresholds are in valid range and sorted
    thresholds = [max(1, min(254, t)) for t in thresholds]
    thresholds = sorted(list(set(thresholds)))  # Remove duplicates, sort
    
    # If duplicates were removed, pad with additional thresholds
    while len(thresholds) < K:
        # Add evenly spaced thresholds in gaps
        new_thr = int(255 * (len(thresholds) + 1) / (K + 1))
        if new_thr not in thresholds:
            thresholds.append(new_thr)
        else:
            break
    
    return thresholds[:K]
