# 🚀 Hướng Dẫn Cài Đặt Đầy Đủ

## Yêu Cầu Hệ Thống

- **OS**: Windows 10/11, macOS, Linux
- **Python**: 3.10 hoặc cao hơn (khuyến nghị 3.11)
- **RAM**: Tối thiểu 4GB (8GB+ khuyến nghị cho benchmark)
- **Disk**: 500MB cho dependencies

## Bước 1: Kiểm Tra Python

```powershell
python --version
# Kết quả mong muốn: Python 3.10.x hoặc 3.11.x
```

Nếu Python chưa được cài, tải từ: https://www.python.org/downloads/

## Bước 2: Tạo Môi Trường Ảo (Virtual Environment)

**Windows PowerShell:**
```powershell
cd C:\Users\Admin\Desktop\DA_KLCN_VerChuan\DA_KLCN
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS/Linux:**
```bash
cd ~/DA_KLCN_VerChuan/DA_KLCN
python -m venv .venv
source .venv/bin/activate
```

✅ **Khi kích hoạt thành công**, dòng lệnh sẽ bắt đầu với `(.venv)`

## Bước 3: Nâng Cấp pip & setuptools

```powershell
python -m pip install --upgrade pip setuptools wheel
```

## Bước 4: Cài Đặt Dependencies

**Cách 1: Cài từ requirements.txt (khuyến nghị)**
```powershell
pip install -r requirements.txt
```

**Cách 2: Cài từng package (nếu gặp lỗi)**
```powershell
# Core
pip install numpy scipy pandas

# Image Processing
pip install opencv-python Pillow scikit-image

# Web
pip install Flask

# Visualization
pip install matplotlib seaborn

# Utilities
pip install python-dotenv
```

## Bước 5: Xác Minh Cài Đặt

```powershell
# Kiểm tra các imports quan trọng
python -c "import numpy; import cv2; import flask; print('✅ All imports OK')"
```

## Bước 6: Chạy Ứng Dụng

```powershell
# Đảm bảo môi trường ảo đang kích hoạt ((.venv) hiển thị)
python -m src.ui.app
```

**Kết quả mong muốn:**
```
 * Serving Flask app 'src.ui.app'
 * Debug mode: off
 * Running on http://127.0.0.1:5000
 * Press CTRL+C to quit
```

## Truy Cập Ứng Dụng

Mở trình duyệt web và truy cập:
```
http://127.0.0.1:5000
```

hoặc

```
http://localhost:5000
```

## Khắc Phục Sự Cố

### ❌ "Python is not recognized"
- Cài lại Python, chọn **"Add Python to PATH"**

### ❌ "ModuleNotFoundError: No module named 'flask'"
- Đảm bảo môi trường ảo đang kích hoạt: `(.venv)` hiển thị
- Cài lại: `pip install -r requirements.txt`

### ❌ "No module named 'cv2' on Windows"
- Cài lại OpenCV:
  ```powershell
  pip uninstall opencv-python
  pip install --upgrade pip setuptools wheel
  pip install opencv-python
  ```

### ❌ Lỗi "scikit-image build failed"
- Cài công cụ build:
  ```powershell
  pip install --upgrade pip setuptools wheel ninja cmake
  pip install scikit-image --force-reinstall
  ```

### ❌ Port 5000 đã được sử dụng
- Dùng port khác (chỉnh sửa `src/ui/app.py`):
  ```python
  app.run(host='127.0.0.1', port=5001, debug=False)
  ```

## Thư Viện Chi Tiết

| Package | Phiên Bản | Mục Đích |
|---------|-----------|---------|
| **numpy** | 1.24.3 | Tính toán khoa học, ma trận |
| **scipy** | 1.11.4 | Xử lý tín hiệu, hàm toán học |
| **opencv-python** | 4.8.1.78 | Xử lý ảnh, histogram |
| **Pillow** | 10.1.0 | I/O ảnh, chuyển đổi format |
| **scikit-image** | 0.22.0 | Otsu, segmentation, filters |
| **pandas** | 2.1.3 | Xử lý dữ liệu, CSV/JSON |
| **matplotlib** | 3.8.2 | Vẽ đồ thị, visualization |
| **seaborn** | 0.13.0 | Đồ thị thống kê |
| **Flask** | 3.0.0 | Web framework |
| **Werkzeug** | 3.0.1 | WSGI utility (Flask dependency) |
| **python-dotenv** | 1.0.0 | Biến môi trường |

## Cài Đặt Phát Triển (Optional)

Nếu muốn phát triển thêm:

```powershell
pip install ruff black pytest pytest-cov
```

## Lệnh Hữu Ích

```powershell
# Deactivate môi trường ảo
deactivate

# Liệt kê tất cả packages đã cài
pip list

# Xuất dependencies hiện tại
pip freeze > requirements_current.txt

# Xóa cache pip
pip cache purge
```

## Kiểm Tra Hoàn Toàn

```powershell
# 1. Kích hoạt môi trường ảo
.\.venv\Scripts\Activate.ps1

# 2. Chạy ứng dụng
python -m src.ui.app

# 3. Truy cập http://localhost:5000

# 4. Tải ảnh mẫu từ `dataset/` hoặc dùng ảnh của bạn

# 5. Chạy benchmark
```

## ✅ Tất Cả Xong!

Bây giờ bạn có thể:
- ✅ Chạy web UI
- ✅ Tải ảnh và phân đoạn
- ✅ So sánh MFWOA vs WOA vs PSO vs Otsu
- ✅ Xuất kết quả (JSON, PNG, CSV)

---

**Liên Hệ & Hỗ Trợ**: Nếu gặp vấn đề, kiểm tra file `README.md` hoặc `TROUBLESHOOTING.md`
