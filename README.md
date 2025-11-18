# Phân đoạn ảnh xám sử dụng MFWOA với Fuzzy Entropy

Hệ thống phân đoạn ảnh đa cấp độ ngưỡng tối ưu hóa Fuzzy Entropy bằng MFWOA, với giao diện web Flask.

## Cài đặt nhanh

```powershell
# Tạo môi trường ảo
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Cài đặt phụ thuộc
python -m pip install -r requirements.txt
python -m pip install --upgrade pip setuptools wheel ninja cmake scikit-image
```

## Chạy ứng dụng

```powershell
python -m src.ui.app
```

Truy cập: **http://127.0.0.1:5000**

## 🚀 Tính năng chính

### Adaptive Iterations (Tối ưu hóa thích ứng cho K cao)

**Bài toán**: Khi chọn K ≥ 5 ngưỡng, hệ thống chạy chậm do:
- Độ phức tạp tính toán = O(pop_size × iterations × K × 256)
- VD: 30 pop × 500 iters × 8 thresholds × 256 = **30.7 triệu phép tính**

**Giải pháp**: Tự động giảm iterations theo K để duy trì thời gian chạy hợp lý

| Số ngưỡng (K) | Iterations | Thời gian | Tiết kiệm |
|---|---|---|---|
| K ≤ 4 | 100% | ~5-6s | baseline |
| K = 5 | 60% (~300) | ~4s | **-23%** ⚡ |
| K = 6 | 40% (~200) | ~2.7s | **-48%** ⚡ |
| K ≥ 8 | 25% (~125) | ~1.8s | **-65%** ⚡ |

**UI Feedback**: Khi K > 4, sẽ thấy `(→ 125 for K=8)` cho biết iterations thực tế

### Thuật toán hỗ trợ
- ✅ **MFWOA**: Tối ưu đa nhiệm, chia sẻ tri thức giữa các công việc K khác nhau
- ✅ **WOA**: Whale Optimization Algorithm (cơ sở của MFWOA)
- ✅ **PSO**: Particle Swarm Optimization
- ✅ **OTSU**: Phương pháp ngưỡng chuẩn (phân tích)

### Hàm mục tiêu: Fuzzy Entropy
Hỗ trợ các membership functions:
- **Triangular** (mặc định)
- **Gaussian**
- **S-shaped** (parametric)

## Kết quả & Đầu ra

### File kết quả
- **JSON**: Ngưỡng tối ưu từ mỗi thuật toán
- **PNG**: Ảnh phân đoạn (segmentation result)
- **CSV**: Metrics (PSNR, SSIM, Fuzzy Entropy value)

### Ví dụ
```json
{
  "mfwoa": [52, 107, 151, 203],
  "otsu": [52, 103, 158],
  "woa": [48, 105, 150, 200],
  "pso": [51, 108, 152, 202]
}
```

## Hiệu suất

### Benchmark: Lena (512×512), MFWOA, pop=30, iters=500

- **K=3**: 5.22s (baseline)
- **K=5**: 4.01s (-23%)
- **K=8**: 1.84s (-65%)

### Chất lượng phân đoạn
- FE (Fuzzy Entropy) > 4.0: Tốt
- PSNR > 25 dB: Chất lượng tốt
- Balance > 10%: Phân phối vùng hợp lý

## Kiến trúc

```
src/
├── ui/           # Web interface (Flask)
├── optim/        # Optimizers (MFWOA, WOA, PSO)
├── metrics/      # Fuzzy Entropy calculation
├── seg/          # Thresholding & segmentation
└── cli/          # Command-line tools
```

## Tài liệu

- **`PERFORMANCE_IMPROVEMENTS.md`**: Chi tiết về adaptive iterations & benchmark
- **`docs/detailed-design.md`**: Thiết kế chi tiết hệ thống
- **`docs/experiments.md`**: Kết quả thực nghiệm

## Phiên bản

- **v2.0**: Adaptive iterations, GA/FCM loại bỏ
- Optimization: MFWOA, WOA, PSO, OTSU
- UI: Flask web interface
