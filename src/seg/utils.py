"""Utilities for threshold constraints and visualization.

Các hàm tiện ích cho:
- Ràng buộc ngưỡng (sorting, clipping, gap enforcing)
- Mapping level cho phân đoạn
- Áp dụng ngưỡng lên ảnh
"""
from typing import List, Tuple
import numpy as np


def enforce_threshold_constraints(thresholds: np.ndarray, min_gap: int = 1) -> np.ndarray:
    """Enforce constraints on thresholds: sorting, clipping, and minimum gap.
    
    Hàm này đảm bảo tất cả threshold tuân theo các ràng buộc:
    - Làm tròn (round) thành số nguyên
    - Sắp xếp tăng dần (sorted)
    - Nằm trong [1, 254]
    - Khoảng cách tối thiểu giữa các ngưỡng liên tiếp >= min_gap
    
    Args:
        thresholds: Array of threshold values (có thể unsorted, bị fractional, ngoài range)
        min_gap: Minimum gap between consecutive thresholds (default 1, tránh trùng)
    
    Returns:
        Constrained threshold array (sorted, clipped, enforced gap, integer)
    """
    # ===== LÀNG TRÒN VÀ SORT =====
    # np.round: làm tròn số thực thành số nguyên (0.4->0, 0.6->1)
    t = np.round(thresholds)
    # t.sort(): sắp xếp array in-place theo tăng dần
    t.sort()
    
    # ===== CLIP VÀO RANGE [1, 254] =====
    # np.clip(x, min, max): giới hạn giá trị x nằm trong [min, max]
    # Giá trị < 1 -> 1, giá trị > 254 -> 254
    t = np.clip(t, 1, 254)
    
    # ===== ENFORCE MINIMUM GAP =====
    # Vòng lặp qua các cặp ngưỡng liên tiếp
    # Đảm bảo t[i] - t[i-1] >= min_gap (tránh trùng, tránh quá gần)
    for i in range(1, len(t)):
        # Kiểm tra nếu khoảng cách < min_gap
        if t[i] - t[i-1] < min_gap:
            # Đẩy t[i] lên để tạo gap
            # t[i] = t[i-1] + min_gap
            t[i] = t[i-1] + min_gap
    
    # ===== CLIP LẠI =====
    # Sau khi enforce gap, một vài giá trị có thể vượt 254, nên clip lại
    t = np.clip(t, 1, 254)
    
    return t


def get_level_mapping(thresholds: List[int]) -> List[float]:
    """Get intensity level mapping for segmentation based on midpoints.
    
    Hàm này tính giá trị intensity đại diện cho mỗi segment.
    Sử dụng trung điểm (midpoint) của mỗi bin là representative value.
    
    Ví dụ: thresholds=[100, 200] -> bounds=[0, 100, 200, 255]
    - Bin 0: [0, 100) -> midpoint = 50
    - Bin 1: [100, 200) -> midpoint = 150
    - Bin 2: [200, 255] -> midpoint = 227.5
    
    Args:
        thresholds: List of threshold values (sorted)
    
    Returns:
        List of intensity levels (float) - giá trị đại diện cho mỗi segment
    """
    # ===== TÍNH BOUNDS =====
    # Bounds là ranh giới của các bin
    # [0] + sorted(thresholds) + [255]
    # Ví dụ: [0, 100, 200, 255]
    bounds = [0] + sorted(thresholds) + [255]
    
    # ===== TÍNH MIDPOINT CỦA MỖI BIN =====
    # Midpoint = (bounds[i] + bounds[i+1]) / 2
    # Vòng lặp qua tất cả bins
    return [(bounds[i] + bounds[i+1])/2 for i in range(len(bounds)-1)]


def apply_thresholds_with_levels(img: np.ndarray, thresholds: List[int]) -> Tuple[np.ndarray, List[int]]:
    """Apply thresholds and return both segmented image and pixel counts.
    
    Áp dụng ngưỡng lên ảnh: tạo ảnh segmented (gán mỗi pixel giá trị level đại diện)
    và đếm số pixel trong mỗi segment.
    
    Args:
        img: Input grayscale image (2D numpy array, uint8)
        thresholds: List of threshold values
    
    Returns:
        Tuple of:
        - segmented image (uint8, mỗi pixel gán level value)
        - pixel counts per class (list of int, số pixel mỗi segment)
    """
    # ===== KHỞI TẠO =====
    # Tính level mapping (giá trị đại diện cho mỗi bin)
    levels = get_level_mapping(thresholds)
    # Tính bounds
    bounds = [0] + sorted(thresholds) + [255]
    # Tạo mảng kết quả (float32, sau chuyển uint8)
    result = np.zeros_like(img, dtype=np.float32)
    # Danh sách đếm pixel
    pixel_counts = []
    
    # ===== PHÂN ĐOẠN TỪNG BIN =====
    # Vòng lặp qua mỗi segment (bin)
    for i in range(len(bounds)-1):
        # ===== TẠO MASK =====
        # Mask: pixel nào nằm trong range [bounds[i], bounds[i+1])
        # (img >= bounds[i]) & (img < bounds[i+1])
        # Ví dụ: i=0, bounds=[0, 100, 200, 255]
        #   mask = (img >= 0) & (img < 100)
        mask = (img >= bounds[i]) & (img < bounds[i+1])
        
        # ===== GÁN LEVEL VALUE =====
        # Gán pixels trong mask giá trị level[i]
        result[mask] = levels[i]
        
        # ===== ĐẾM PIXEL =====
        # mask.sum(): đếm True (số pixel trong mask)
        # int(...): chuyển thành Python int
        pixel_counts.append(int(mask.sum()))
    
    # ===== TRẢ VỀ KẾT QUẢ =====
    # Chuyển result từ float32 sang uint8 (8-bit unsigned)
    return result.astype(np.uint8), pixel_counts