# Phân đoạn ảnh xám sử dụng MFWOA với Fuzzy Entropy

Hệ thống phân đoạn ảnh đa cấp độ ngưỡng tối ưu hóa Fuzzy Entropy bằng MFWOA, với giao diện web Flask.

## 📚 Tài liệu & Hướng dẫn

> **👉 START HERE**: Đọc [`HANDOFF.md`](HANDOFF.md) hoặc [`DOCUMENTATION_INDEX.md`](DOCUMENTATION_INDEX.md) để hiểu cấu trúc tài liệu.

### Các tài liệu chính (Nov 20, 2025)
| Tài liệu | Nội dung | Thời gian |
|---------|---------|----------|
| **[HANDOFF.md](HANDOFF.md)** | 🎉 Bảng tóm tắt, quick start | 5 phút |
| **[DOCUMENTATION_INDEX.md](DOCUMENTATION_INDEX.md)** | 📍 Bản đồ tất cả tài liệu | 10 phút |
| **[QUICK_REFERENCE.md](docs/QUICK_REFERENCE.md)** | ⚡ Hướng dẫn thực hành, tham số | 30 phút |
| **[KNOWLEDGE_TRANSFER_GUIDE.md](docs/KNOWLEDGE_TRANSFER_GUIDE.md)** | 🧠 Chi tiết thuật toán, chia sẻ tri thức | 45 phút |
| **[IMPLEMENTATION_SUMMARY.md](docs/IMPLEMENTATION_SUMMARY.md)** | 📊 Tổng quan hệ thống, benchmarks | 20 phút |
| **[PROJECT_STATUS.md](docs/PROJECT_STATUS.md)** | ✅ Trạng thái dự án, sẵn sàng triển khai | 15 phút |
| **[SESSION_SUMMARY.md](docs/SESSION_SUMMARY.md)** | 📝 Công việc hoàn tất trong phiên làm việc | 15 phút |

### Cách tiếp cận nhanh (5 phút)
1. Đọc phần "What You Have" trong [`HANDOFF.md`](HANDOFF.md)
2. Chạy: `python -m src.ui.app`
3. Truy cập: http://localhost:5000
4. Tải ảnh lên và chạy benchmark

---

## Cài đặt nhanh

```powershell
# Tạo môi trường ảo
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Cài đặt phụ thuộc
python -m pip install -r requirements.txt
python -m pip install --upgrade pip setuptools wheel ninja cmake scikit-image
```
pip install torch --index-url https://download.pytorch.org/whl/cu118
set CUDA_VISIBLE_DEVICES=1
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
