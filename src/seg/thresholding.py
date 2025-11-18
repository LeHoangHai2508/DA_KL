"""Multilevel thresholding utilities (pure functions).

Các hàm thuần (pure functions) cho áp dụng ngưỡng và tạo mask phân đoạn.
"""
from __future__ import annotations

import numpy as np
from typing import Sequence, Tuple


def apply_thresholds(image: np.ndarray, thresholds: Sequence[int]) -> np.ndarray:
    """Trả về ảnh phân đoạn (labels 0..K) bằng cách dùng np.digitize.
    
    Áp dụng tập hợp ngưỡng lên ảnh xám để tạo bản đồ class.
    Mỗi pixel được gán một class label (0..K) tương ứng với bin nó rơi vào.

    Ví dụ:
    - image = [10, 50, 150, 200]
    - thresholds = [100, 180]
    - bins = [100, 180]
    - digitize trả về: [0, 0, 1, 2] (classes: <100, [100-180), >=180)

    Args:
        image: Grayscale image (2D numpy array, uint8 hoặc float)
        thresholds: Sequence of threshold values (có thể unsorted)

    Returns:
        labels array (2D, dtype=int): mỗi pixel là class label (0..len(thresholds))
    """
    # ===== VALIDATION =====
    # Kiểm tra ảnh phải là 2D (xám)
    if image.ndim != 2:
        raise ValueError("image must be grayscale 2D array")
    
    # ===== SORT THRESHOLDS =====
    # np.digitize yêu cầu bins phải sorted
    # Sắp xếp thresholds tăng dần và chuyển thành int
    bins = np.array(sorted(int(t) for t in thresholds), dtype=int)
    
    # ===== DIGITIZE =====
    # np.digitize(image, bins, right=False):
    #   - Đếm bao nhiêu bin mà mỗi pixel vượt qua
    #   - right=False: bins[-1] < x <= bins[0] là class 1, không nằm < bins[0]
    #   - Kết quả: [0, 1, 2, ..., len(bins)] (K+1 classes)
    # Ví dụ: bins=[100, 180]
    #   - x < 100: digitize = 0
    #   - 100 <= x < 180: digitize = 1
    #   - x >= 180: digitize = 2
    labels = np.digitize(image, bins, right=False)
    
    return labels


def labels_to_mask(labels: np.ndarray, class_id: int) -> np.ndarray:
    """Trả về mask boolean cho class_id.
    
    Chuyển đổi bản đồ class thành binary mask (foreground/background).
    Mỗi pixel có class = class_id thành 255 (trắng/true), còn lại = 0 (đen/false).

    Ví dụ:
    - labels = [[0, 1, 2], [1, 1, 0]]
    - class_id = 1
    - result = [[0, 255, 0], [255, 255, 0]]

    Args:
        labels: Class label map (2D array, int)
        class_id: Target class ID

    Returns:
        Binary mask (2D array, uint8): 255 nếu label==class_id, 0 ngược lại
    """
    # ===== TẠO BOOLEAN MASK =====
    # (labels == class_id): so sánh từng pixel, trả về True/False array
    # .astype(np.uint8): chuyển True->1, False->0
    # * 255: scale từ 0-1 lên 0-255
    return (labels == class_id).astype(np.uint8) * 255
