"""CLI entrypoint to run optimization on an image.

Chương trình command-line để chạy MFWOA tối ưu ngưỡng phân đoạn trên ảnh.
Cú pháp: python -m src.cli.optimize --image <path> --K <num> --pop <pop> --iters <iters>
"""
from __future__ import annotations

# ===== IMPORT STANDARD LIBRARIES =====
# argparse: parse command-line arguments
import argparse
# json: đọc/ghi file JSON
import json
# pathlib.Path: xử lý đường dẫn file (cross-platform)
from pathlib import Path

# ===== IMPORT EXTERNAL LIBRARIES =====
# cv2 (OpenCV): đọc/ghi ảnh
import cv2
# numpy: xử lý array, phép toán số học
import numpy as np

# ===== IMPORT PROJECT MODULES =====
# histogram_from_image: chuyển ảnh -> histogram 256-bin
# compute_fuzzy_entropy: tính Fuzzy Entropy + penalties
from src.metrics.fuzzy_entropy import histogram_from_image, compute_fuzzy_entropy
# mfwoa_optimize: thuật toán MFWOA tối ưu
from src.optim.mfwoa import mfwoa_optimize
# apply_thresholds: áp dụng ngưỡng -> bản đồ class
from src.seg.thresholding import apply_thresholds


def main():
    """Hàm main của CLI.
    
    Quy trình:
    1. Parse command-line arguments (image path, K, pop, iters)
    2. Đọc ảnh từ file
    3. Tính histogram
    4. Tạo objective function (wrapping compute_fuzzy_entropy)
    5. Chạy MFWOA tối ưu
    6. Áp dụng ngưỡng tốt nhất -> ảnh phân đoạn
    7. Lưu kết quả (segmented image, JSON thresholds)
    """
    
    # ===== PARSE COMMAND-LINE ARGUMENTS =====
    # argparse.ArgumentParser: tạo parser cho CLI arguments
    parser = argparse.ArgumentParser(description="MFWOA fuzzy entropy multilevel thresholding")
    
    # --image: bắt buộc, path tới ảnh input
    parser.add_argument("--image", required=True, help="Path to grayscale image")
    # --K: số ngưỡng (default=2)
    parser.add_argument("--K", type=int, default=2, help="Number of thresholds")
    # --pop: population size (default=30)
    parser.add_argument("--pop", type=int, default=30)
    # --iters: số iteration (default=100)
    parser.add_argument("--iters", type=int, default=100)
    # --out: output directory (default='results')
    parser.add_argument("--out", default="results")
    # args: Namespace object chứa parsed values
    args = parser.parse_args()

    # ===== ĐỌC IMAGE =====
    # Path(args.image): convert string path -> Path object (cross-platform)
    img_path = Path(args.image)
    # Path(args.out): output directory
    out_dir = Path(args.out)
    # out_dir.mkdir(parents=True, exist_ok=True): tạo directory (recursive, không error nếu đã tồn tại)
    out_dir.mkdir(parents=True, exist_ok=True)

    # cv2.imread(path, cv2.IMREAD_GRAYSCALE): đọc ảnh thành grayscale (2D array)
    # str(img_path): convert Path -> string (OpenCV yêu cầu string)
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    # Kiểm tra xem có đọc được ảnh không
    if img is None:
        # Nếu không, in lỗi và thoát chương trình (exit code 1)
        raise SystemExit(f"Cannot read image {img_path}")
    
    # ===== TÍNH HISTOGRAM =====
    # histogram_from_image(img): chuyển ảnh -> histogram 256-bin
    hist = histogram_from_image(img)

    # ===== TẠO OBJECTIVE FUNCTION =====
    # Objective function dùng cho optimizer (MFWOA)
    # Nhận vị trí (continuous) và trả về fitness score
    def obj(pos):
        # ===== CONVERT POSITION TO THRESHOLDS =====
        # pos: array liên tục từ optimizer (K thresholds)
        # Làm tròn và chuyển thành int
        th = [int(round(x)) for x in pos]
        
        # ===== TÍNH FUZZY ENTROPY =====
        # compute_fuzzy_entropy(hist, th, membership="triangular")
        # membership="triangular": dùng tam giác membership function
        return compute_fuzzy_entropy(hist, th, membership="triangular")

    # ===== CHẠY MFWOA =====
    # mfwoa_optimize: tối ưu maximize Fuzzy Entropy
    # Args:
    #   hist: 256-bin histogram
    #   args.K: số ngưỡng
    #   pop_size=args.pop: kích thước quần thể
    #   iters=args.iters: số iteration
    #   objective=obj: objective function
    # Returns: (best_thresholds, best_score)
    best_th, best_score = mfwoa_optimize(hist, args.K, pop_size=args.pop, iters=args.iters, objective=obj)

    # ===== ÁP DỤNG NGƯỠNG =====
    # apply_thresholds(img, best_th): áp dụng best thresholds lên ảnh
    # Returns: class labels (0..K) cho mỗi pixel
    labels = apply_thresholds(img, best_th)
    
    # ===== SAVE SEGMENTED IMAGE =====
    # Đường dẫn output: out_dir / "segmented.png"
    seg_path = out_dir / "segmented.png"
    # Chuyển labels (class 0..K) thành grayscale (0..255) để visualize
    # labels.max(): max label value (= K)
    Klabels = labels.max()
    # Normalize: labels / max * 255 -> uint8
    vis = (labels.astype(np.float32) / max(1, Klabels) * 255.0).astype(np.uint8)
    # cv2.imwrite(path, array): ghi ảnh uint8 về PNG file
    cv2.imwrite(str(seg_path), vis)

    # ===== SAVE THRESHOLDS AS JSON =====
    # Đường dẫn output JSON
    json_path = out_dir / "thresholds.json"
    # open(path, 'w', encoding='utf-8'): mở file để ghi (UTF-8)
    with open(json_path, "w", encoding="utf-8") as f:
        # json.dump(dict, file, indent=2): ghi dict thành JSON (indented)
        json.dump({"thresholds": best_th, "score": best_score}, f, indent=2)

    # ===== PRINT RESULT =====
    print(f"done -> {seg_path}, {json_path}")


# ===== ENTRY POINT =====
if __name__ == "__main__":
    # Chạy hàm main khi file được execute trực tiếp (không import)
    main()
