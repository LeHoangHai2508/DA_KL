"""
Otsu multilevel thresholding implementation.

Provides deterministic threshold computation using:
- True Otsu method for K <= 3 (via scikit-image)
- Fast quantile approximation for K >= 4
"""

import numpy as np
from typing import List


def otsu_thresholds(image: np.ndarray, K: int) -> List[int]:
    """
    Compute K multilevel thresholds using Otsu method.
    
    For K <= 3: Uses true Otsu via scikit-image (O(256^K))
    For K >= 4: Uses fast quantile approximation (O(N log N))
    
    Args:
        image: Grayscale image (2D array or PIL Image)
        K: Number of thresholds
        
    Returns:
        List of K integer thresholds in [1, 254], sorted ascending
    """
    from skimage.filters import threshold_multiotsu
    
    # Convert PIL Image to numpy array if needed
    if hasattr(image, 'convert'):
        arr = np.array(image.convert('L'))
    else:
        arr = np.asarray(image, dtype=np.uint8)
    
    if K <= 0:
        raise ValueError(f"K must be positive, got K={K}")
    
    if K >= 4:
        # Fast quantile-based approximation: O(N log N)
        # For large K, true Otsu becomes intractable (O(256^K))
        intensity_sorted = np.sort(arr.ravel())
        # Place K thresholds evenly across intensity range
        thresholds = [int(np.quantile(intensity_sorted, i / (K + 1))) for i in range(1, K + 1)]
    else:
        # Use true Otsu for K=2 and K=3 (acceptable speed)
        # threshold_multiotsu returns K+1 boundaries, we take first K as thresholds
        thr = threshold_multiotsu(arr, classes=K+1) if K > 1 else threshold_multiotsu(arr, classes=2)
        thresholds = [int(t) for t in thr[:-1]]  # Exclude max value
    
    # Ensure thresholds are in valid range and sorted
    thresholds = [max(1, min(254, t)) for t in thresholds]
    thresholds = sorted(list(set(thresholds)))  # Remove duplicates, sort
    
    # Pad with default values if needed (e.g., duplicates were removed)
    while len(thresholds) < K:
        # Add evenly spaced thresholds in gaps
        new_thr = int(255 * (len(thresholds) + 1) / (K + 1))
        if new_thr not in thresholds:
            thresholds.append(new_thr)
        else:
            break
    
    return thresholds[:K]
