"""Logging utility for tracking experiment results across seeds.

Quản lý ghi nhật ký kết quả thí nghiệm (metric, ảnh phân đoạn, overlay) theo từng seed.
Tạo CSV tập hợp toàn bộ seeds + CSV riêng cho mỗi seed.
"""
from __future__ import annotations

# pathlib.Path: xử lý đường dẫn file (cross-platform)
from pathlib import Path
# csv: ghi dữ liệu DictWriter format
import csv
# numpy: xử lý array ảnh
import numpy as np
# cv2 (OpenCV): đọc/ghi ảnh, color map
import cv2


class SeedLogger:
    """Ghi log kết quả optimization cho mỗi seed.
    
    Tạo thư mục structure:
    root/
    └── image_stem/
        └── noise_tag/
            ├── metrics_all_seeds.csv (tất cả seeds)
            └── seed_000/
                ├── metrics.csv (riêng seed này)
                ├── MFWOA_K2_mask.png
                ├── MFWOA_K2_overlay.png
                ├── ...
    """
    
    def __init__(self, root: str | Path, image_stem: str, noise_tag: str | None = None):
        """Khởi tạo SeedLogger.
        
        Args:
            root: Thư mục root để lưu log
            image_stem: Tên ảnh (e.g., "lena", "cameraman")
            noise_tag: Tag nhiễu (default="clean")
        """
        # Path(root): convert string -> Path object (cross-platform)
        self.root = Path(root)
        # Tên ảnh (không có extension)
        self.image_stem = image_stem
        # Tag nhiễu (default "clean" nếu không chỉ định)
        self.noise_tag = noise_tag or "clean"
        
        # base directory: root / image_stem / noise_tag
        self.base = self.root / image_stem / self.noise_tag
        # mkdir(parents=True, exist_ok=True): tạo thư mục (recursive, không error nếu tồn tại)
        self.base.mkdir(parents=True, exist_ok=True)

        # Đường dẫn CSV tập hợp toàn bộ seeds
        self.agg_csv = self.base / "metrics_all_seeds.csv"
        # Danh sách field header cho CSV
        # seed: số seed
        # algo: tên thuật toán (MFWOA, WOA, PSO, Otsu)
        # K: số ngưỡng
        # FE: Fuzzy Entropy score
        # PSNR: Peak Signal-to-Noise Ratio
        # SSIM: Structural Similarity Index
        # time_ms: thời gian chạy (milliseconds)
        # notes: ghi chú thêm
        # out_mask: đường dẫn PNG mask
        # out_overlay: đường dẫn PNG overlay (blend ảnh gốc + mask)
        self._csv_fields = ["seed","algo","K","FE","PSNR","SSIM","time_ms","notes","out_mask","out_overlay"]
        # Flag: CSV đã có header chưa
        # agg_csv.exists(): True nếu file tồn tại (=> header đã được ghi)
        self._csv_inited = self.agg_csv.exists()

    def _append_row(self, row: dict):
        """Ghi một row dữ liệu vào CSV tập hợp.
        
        Args:
            row: Dict chứa dữ liệu (key: field name, value: giá trị)
        """
        # Mở file CSV ở chế độ append (add vào cuối file)
        # newline="": yêu cầu csv module, không auto-convert newline
        # encoding="utf-8": hỗ trợ tiếng Việt
        with self.agg_csv.open("a", newline="", encoding="utf-8") as f:
            # csv.DictWriter: ghi dict thành CSV row
            # fieldnames: danh sách column name
            w = csv.DictWriter(f, fieldnames=self._csv_fields)
            # Nếu CSV mới (chưa ghi header), ghi header line
            if not self._csv_inited:
                # writeheader(): ghi dòng đầu (tên field)
                w.writeheader()
                # Đánh dấu đã ghi header
                self._csv_inited = True
            # writerow(dict): ghi dòng dữ liệu
            w.writerow(row)

    @staticmethod
    def _save_mask(mask: np.ndarray, path: Path):
        """Ghi binary mask thành PNG file.
        
        Args:
            mask: Array 2D (0 hoặc 1, hoặc class labels)
            path: Đường dẫn file output
        """
        # Chuyển mask thành uint8 (0-255) nếu cần
        # mask.astype(np.uint8) nếu chưa uint8, ngược lại giữ nguyên
        m = (mask.astype(np.uint8) if mask.dtype != np.uint8 else mask)
        # cv2.imwrite(path, array): ghi array thành PNG file
        # str(path): convert Path -> string (OpenCV yêu cầu)
        cv2.imwrite(str(path), m)

    @staticmethod
    def _save_overlay(img_gray: np.ndarray, mask: np.ndarray, path: Path, alpha: float = 0.6):
        """Blend ảnh grayscale với mask (colorized), lưu thành PNG.
        
        Quy trình:
        1. Chuyển ảnh xám thành BGR 3-channel (để blend với colormap)
        2. Áp dụng color map JET lên mask
        3. Blend: α*ảnh_gốc + (1-α)*mask_color
        4. Lưu kết quả
        
        Args:
            img_gray: Ảnh grayscale gốc (2D array)
            mask: Mask class labels (0..K)
            path: Đường dẫn output
            alpha: Hệ số blend (default 0.6: 60% ảnh gốc, 40% mask)
        """
        # Chuyển ảnh grayscale thành BGR 3-channel
        # img_gray.ndim == 2: ảnh grayscale (2D)
        if img_gray.ndim == 2:
            # cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR): chuyển grayscale -> BGR
            # Kết quả: 3-channel giống nhau (R=G=B=gray)
            g3 = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
        else:
            # Nếu đã là 3-channel, dùng luôn
            g3 = img_gray
        
        # Tạo colormap từ mask labels
        # mask.astype(np.int32): chuyển sang int32 (safe để nhân/chia)
        lab = mask.astype(np.int32)
        # Nếu tất cả label = 0, tạo array đen (0)
        if lab.max() == 0:
            # np.zeros(shape, dtype): tạo array toàn 0
            # (*lab.shape, 3): shape = (H, W, 3) - BGR 3-channel
            cm = np.zeros((*lab.shape, 3), np.uint8)
        else:
            # Normalize labels sang 0-255
            # lab * (255 // max_label): scale tuyến tính
            # max(int(lab.max()), 1): tránh chia cho 0
            norm = (lab * (255 // max(int(lab.max()), 1))).astype(np.uint8)
            # cv2.applyColorMap(array, cv2.COLORMAP_JET): áp color map JET lên array
            # COLORMAP_JET: map giá trị 0-255 -> màu JET (xanh-xanh lá-vàng-đỏ)
            cm = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
        
        # Blend hai ảnh: α*ảnh_gốc + (1-α)*mask_color
        # astype(np.float32): chuyển sang float để tính toán (tránh overflow)
        over = (alpha * g3 + (1 - alpha) * cm).astype(np.uint8)
        # cv2.imwrite: ghi PNG
        cv2.imwrite(str(path), over)

    def log(self, *, seed: int, algo: str, K: int, metrics: dict,
            img_gray: np.ndarray, mask: np.ndarray, subdir: str | None = None, notes: str | None = None):
        """Ghi log kết quả cho một optimization run.
        
        Tạo thư mục per-seed, lưu mask + overlay, ghi CSV.
        
        Args:
            seed: Seed number
            algo: Tên thuật toán (MFWOA, WOA, PSO, Otsu)
            K: Số ngưỡng
            metrics: Dict metric {FE, PSNR, SSIM, time_ms}
            img_gray: Ảnh grayscale gốc
            mask: Class labels (0..K)
            subdir: Thư mục con (optional, e.g., "run_1")
            notes: Ghi chú thêm (optional)
        """
        # Tạo thư mục per-seed: base / seed_NNN (e.g., seed_000, seed_001)
        seed_dir = self.base / f"seed_{seed:03d}"
        # Nếu có subdir, tạo thư mục con (e.g., seed_000/run_1)
        if subdir:
            # seed_dir / subdir
            seed_dir = seed_dir / subdir
        # mkdir(parents=True, exist_ok=True): tạo thư mục recursive
        seed_dir.mkdir(parents=True, exist_ok=True)

        # Tạo đường dẫn cho mask + overlay PNG
        # Tên file: {algo}_K{num}_mask.png (e.g., MFWOA_K2_mask.png)
        mask_path = seed_dir / f"{algo}_K{K}_mask.png"
        # Overlay: MFWOA_K2_overlay.png
        overlay_path = seed_dir / f"{algo}_K{K}_overlay.png"
        
        # Ghi mask + overlay PNG files
        self._save_mask(mask, mask_path)
        self._save_overlay(img_gray, mask, overlay_path)

        # Tạo dict row để ghi CSV
        # row: {seed, algo, K, FE, PSNR, SSIM, time_ms, notes, out_mask, out_overlay}
        row = {
            "seed": seed,
            "algo": algo,
            "K": K,
            "FE": metrics.get("FE"),           # Fuzzy Entropy
            "PSNR": metrics.get("PSNR"),       # Peak SNR
            "SSIM": metrics.get("SSIM"),       # Structural Similarity
            "time_ms": int(metrics.get("time_ms", 0)),  # Thời gian (ms), default 0
            "notes": notes or "",              # Ghi chú, default empty string
            "out_mask": str(mask_path.relative_to(self.base)),       # Path tương đối từ base
            "out_overlay": str(overlay_path.relative_to(self.base)), # Path tương đối từ base
        }
        # Ghi row vào CSV tập hợp (metrics_all_seeds.csv)
        self._append_row(row)

        # Ghi row vào CSV riêng cho seed này (seed_NNN/metrics.csv)
        per_seed_csv = seed_dir / "metrics.csv"
        # Kiểm tra xem CSV riêng đã tồn tại chưa
        # header_new = True nếu file mới (chưa ghi header)
        header_new = not per_seed_csv.exists()
        # Mở file CSV ở chế độ append
        with per_seed_csv.open("a", newline="", encoding="utf-8") as f:
            # csv.DictWriter: ghi dict thành CSV
            # fieldnames: danh sách cột (từ keys của row dict)
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            # Nếu file mới, ghi header trước
            if header_new:
                w.writeheader()
            # Ghi dòng dữ liệu
            w.writerow(row)
