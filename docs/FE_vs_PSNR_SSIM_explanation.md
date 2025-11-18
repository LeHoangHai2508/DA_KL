# 🔍 Tại sao FE cao nhưng PSNR/SSIM thấp?

## Khái niệm cơ bản

### **Fuzzy Entropy (FE)**
- **Mục đích**: Đo độ "mờ" (fuzziness) của phân loại - tức là độ không chắc chắn khi gán mỗi pixel vào lớp
- **Tối ưu hóa**: MFWOA cố gắng **tối đa hóa FE** → tìm bộ ngưỡng mà sự phân loại mờ nhất
- **Công thức**: `H = Σ(μ_i(x) * S(μ_i(x)))` trong đó `μ_i(x)` = độ thuộc của pixel x vào lớp i
- **Ý nghĩa cao**: Sự phân loại rất "mờ" - ranh giới giữa các lớp không rõ ràng

### **PSNR (Peak Signal-to-Noise Ratio)**
- **Mục đích**: Đo độ gần giữa **ảnh gốc** và **ảnh tái tạo từ segmentation**
- **Cách tính**: 
  ```
  Seg labels → Reconstruction (map each class to mean gray value) → Compare with original
  ```
- **Giá trị cao**: Ảnh tái tạo rất giống ảnh gốc

### **SSIM (Structural Similarity Index)**
- **Mục đích**: Đo độ giống về **cấu trúc** giữa ảnh gốc và tái tạo
- **Cao hơn PSNR**: Tính đến sự nhận thức của con người (edges, contrast, structure)

### **DICE (Sorensen-Dice Coefficient)**
- **Mục đích**: Đo **overlap** giữa segmentation result và ground-truth
- **Công thức**: `DICE = 2|A ∩ B| / (|A| + |B|)`
- **Chỉ tính được** khi có ground-truth

---

## 🚨 Tại sao FE cao nhưng PSNR/SSIM thấp?

### **Lý do: FE ≠ PSNR/SSIM**

| Metric | Tối ưu hóa | Kết quả |
|--------|-----------|--------|
| **FE** | Độ mờ của phân loại | Ranh giới mềm mại, many shades of gray |
| **PSNR/SSIM** | Giống ảnh gốc | Cần tái tạo chính xác intensity values |

### Ví dụ minh họa:

**Ảnh gốc**: `[50, 100, 150, 200]` (4 pixel, intensities khác nhau)

**Bộ ngưỡng A** (FE cao):
- Ngưỡng ở `[75, 125, 175]` → mỗi pixel vào lớp khác nhau
- Mỗi lớp có **1 pixel** → tái tạo = `[50, 100, 150, 200]` → **PSNR cao ✓**

**Bộ ngưỡng B** (FE rất cao):
- Ngưỡng ở `[100, 150]` → đặt sai → 
  - Pixel 50 → lớp 0 (recon = mean([50]) = 50)
  - Pixel 100, 150 → lớp 1 (recon = mean([100, 150]) = 125)
  - Pixel 200 → lớp 2 (recon = mean([200]) = 200)
  - Tái tạo = `[50, 125, 125, 200]` → **PSNR thấp ✗**

Nhưng **FE cao** vì:
- Độ thuộc mờ rất lớn ở ranh giới (pixels gần ngưỡng có μ ≈ 0.5)
- Entropy của phân loại mờ rất cao

---

## ✅ Cách khắc phục

### **1. Thêm ràng buộc vào FE (Hybrid Optimization)**
```python
# Thay vì tối ưu hóa FE đơn thuần
# → Tối ưu hóa: FE * (1 - λ * reconstruction_error)
# hoặc: FE * PSNR_normalized
```

### **2. Sử dụng ground-truth khi có sẵn**
```python
# Nếu có GT → Tối ưu hóa: FE * DICE
# Điều này đảm bảo cả fuzzy entropy và accuracy vs GT
```

### **3. Chọn membership function phù hợp**
- **Triangular**: Ranh giới sắc nét → PSNR cao hơn
- **Gaussian**: Ranh giới mềm mại → FE cao hơn
- **S-shaped**: Trung bình

### **4. Kiểm tra "Optimal thresholds" trực quan**
- So sánh **histogram thresholds** của các algo
- MFWOA ngưỡng có thể cách xa hơn so với Otsu (cố gắng maximize FE)

---

## 📊 Giải pháp trong UI

### **Bổ sung**:

1. **Hiển thị relationship giữa FE vs PSNR/SSIM**
   - Thêm scatter plot hoặc correlation table
   - Để user thấy trade-off

2. **Cho phép chọn optimize target**:
   ```html
   <select name="optimize_target">
     <option value="fe">Fuzzy Entropy (mờ)</option>
     <option value="psnr">PSNR (chính xác tái tạo)</option>
     <option value="combined">Kết hợp (FE + PSNR)</option>
     <option value="dice">DICE (nếu có GT)</option>
   </select>
   ```

3. **Thêm Membership selector**:
   - User có thể thử `triangular` (PSNR cao) vs `gaussian` (FE cao)

4. **Hiển thị cả "Best by FE" và "Best by PSNR"**:
   - Thay vì chỉ hiện top 1 result

---

## 🎯 Tóm tắt

| Khi FE cao | Có thể là | Cách xử lý |
|-----------|----------|-----------|
| Nhưng PSNR/SSIM thấp | **Bình thường** - FE ≠ PSNR | Chấp nhận trade-off hoặc dùng hybrid fitness |
| Nhưng DICE thấp | Ngưỡng không khớp GT | Sử dụng GT trong tối ưu hóa |
| Nhưng visual xấu | Membership function không phù hợp | Thử `triangular` thay `gaussian` |

**Kết luận**: FE cao và PSNR/SSIM thấp là **bình thường** vì chúng tối ưu hóa mục tiêu khác nhau. 
Cần chọn **hàm mục tiêu phù hợp với yêu cầu ứng dụng**.
