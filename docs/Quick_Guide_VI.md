# 🎓 Quick Guide: Giải pháp cho các vấn đề của bạn

## ❓ Câu hỏi 1: Tại sao FE tốt nhưng các chỉ số khác (PSNR, SSIM) không tốt?

### ✅ **Đây là bình thường!**

**Lý do**: FE và PSNR/SSIM tối ưu hóa **mục tiêu khác nhau**:

| Metric | Mục tiêu | Khi cao | Khi thấp |
|--------|---------|--------|---------|
| **FE** | Độ mờ của phân loại | Ranh giới mềm mại, không chắc chắn | Ranh giới sắc nét, rõ ràng |
| **PSNR/SSIM** | Độ giống ảnh gốc | Ảnh tái tạo rất chính xác | Ảnh tái tạo rất khác ảnh gốc |

### 🎯 **Cách chọn**:

1. **Bạn muốn ưu tiên FE** (phân loại mờ)?
   - ✅ Dùng `gaussian` membership function
   - ✅ MFWOA sẽ maximize FE
   - ❌ Kỳ vọng PSNR/SSIM không cao

2. **Bạn muốn ưu tiên PSNR/SSIM** (tái tạo chính xác)?
   - ✅ Dùng `triangular` membership function
   - ✅ Otsu hoặc thủ công ngưỡng cách xa
   - ❌ FE sẽ thấp hơn

3. **Bạn không biết chọn cái nào**?
   - ✅ Chạy benchmark với **tất cả thuật toán**
   - ✅ Xem section **"Best Results Comparison"** trong kết quả
   - ✅ So sánh 3 card: "Best by FE", "Best by PSNR", "Best by SSIM"
   - ✅ Chọn cái phù hợp nhất với ứng dụng của bạn

---

## ❓ Câu hỏi 2: Output là bộ ngưỡng + ảnh mask, nhưng sao nó làm gì xấu?

### 🔍 **Phân tích**:

**Ảnh xấu** có thể do:

1. **K (số ngưỡng) không phù hợp**
   - VD: Ảnh 256 tones mà chỉ dùng K=2 → quá ít thông tin
   - ✅ Cơm:  Thử K=4,5,6 để xem hiệu quả

2. **Membership function sai**
   - `triangular` → Ranh giới sắc nét
   - `gaussian` → Ranh giới mềm (có thể quá mềm)
   - ✅ Cách: Thử cả 2 loại và so sánh

3. **Optimizer convergence kém**
   - Iterations quá ít → không đủ thời gian tìm tối ưu
   - ✅ Cách: Tăng `iters` lên 100-200 trong form

4. **Ảnh gốc không phù hợp**
   - Ảnh quá sáng/tối → histogram lệch
   - ✅ Cách: Cân bằng ảnh trước uploading

### ✅ **Cách kiểm tra**:

Khi benchmark xong:
1. Nhìn **"Best Results Comparison"** - 3 card với ảnh
2. So sánh các ảnh xem cái nào tốt nhất
3. Nếu tất cả đều xấu → thay đổi K hoặc ảnh
4. Tải CSV kết quả để analyze chi tiết

---

## ❓ Câu hỏi 3: Tại sao DICE không thể xem được?

### 📊 **Lý do**: DICE cần **Ground-Truth (GT)**

**DICE** = độ giống giữa kết quả + ảnh GT (0-1 scale)

#### Khi GT **không có** → DICE = `—` (không tính)
- Bình thường, không là lỗi
- Bạn sẽ thấy info box: *"Ground-truth was not provided"*

#### Khi GT **có** → DICE được tính
1. **Upload ảnh GT** (binary mask hoặc label image)
2. Chạy benchmark
3. Kết quả sẽ hiển thị **DICE score** cho mỗi algorithm

#### ✅ **Cách kích hoạt DICE**:

1. Mở form
2. Dưới "📷 Open Image" sẽ thấy "🎯 Ground Truth (optional)"
3. **Upload ảnh GT** (binary mask: đen=background, trắng=foreground)
4. Chạy benchmark
5. Kết quả sẽ có cột **DICE** đầy đủ

---

## ❓ Câu hỏi 4: Tôi muốn thêm phần so sánh giữa 2 ảnh sau khi có bộ ngưỡng tốt

### ✅ **Đã được thêm!**

Khi benchmark xong, kết quả sẽ hiển thị:

#### **1. "Best Results Comparison"** (SỚM)
- 3 cards hiển thị side-by-side:
  - 🏆 Best by FE
  - 🏆 Best by PSNR  
  - 🏆 Best by SSIM
- Mỗi card có:
  - **Ảnh segmentation** (màu hóa theo class)
  - **Metrics chính**: FE, PSNR, SSIM, DICE
  - **Thresholds** dùng cho kết quả này

#### **2. "Segmentation Comparison (All Algorithms)"** (NGAY DƯỚI)
- **Grid 3-4 column** hiển thị **tất cả** algorithms
- Mỗi card có:
  - Ảnh segmentation
  - Tên algorithm
  - Thresholds
  - FE value
  - Download button

#### **3. "Full Metrics Table"** (CÓ CHI TIẾT)
- Bảng đầy đủ với tất cả 7 metrics
- Sắp xếp theo algo
- Có collapsible "Why FE high but PSNR low?" explanation

### 🎨 **Cách nhận biết cái nào tốt nhất**:

1. **Nếu muốn reconstruction chính xác**: Xem "Best by PSNR" card
2. **Nếu muốn classification mờ**: Xem "Best by FE" card
3. **Nếu có GT**: Xem "Best by SSIM" hoặc check DICE column
4. **Muốn compare 2 cái**: So sánh 2 cards trong "Segmentation Comparison"

---

## 🚀 **Workflow Cơ Bản**

### Step 1: Upload & Config
```
1. Chọn ảnh
2. (Optional) Chọn GT mask
3. Chọn K (số ngưỡng) - mặc định 4
4. Chọn thuật toán: Otsu, MFWOA, WOA, GA
5. Chọn pop_size (mặc định 30) + iters (mặc định 100)
6. Chọn membership: triangular hoặc gaussian
```

### Step 2: Run Benchmark
```
7. ☑️ "Run as Benchmark" (để so sánh các algo)
8. Click "⚙️ Optimize & Benchmark"
9. Đợi processing (3s-2m tùy vào tham số)
```

### Step 3: Analyze Results
```
10. Xem "Histogram + Thresholds" - ngưỡng ở đâu?
11. Xem "Best Results Comparison" - cái nào tốt?
12. Xem "Full Metrics Table" - so sánh chi tiết
13. Download CSV hoặc ảnh nếu muốn
```

### Step 4: Iterate (nếu không hài lòng)
```
14. Nếu kết quả không tốt:
    - Thay đổi K (thử 3, 5, 6)
    - Thay đổi membership (thử cái kia)
    - Tăng iters (100 → 200)
    - Thay ảnh khác
15. Quay lại Step 1
```

---

## 📊 **Cách Đọc Kết Quả**

### Metrics Table Columns:

| Column | Ý nghĩa | Cao = Tốt? |
|--------|---------|-----------|
| **Algorithm** | Tên thuật toán | N/A |
| **Thresholds** | Bộ ngưỡng tìm được (vd: 50,100,150) | - |
| **FE** | Fuzzy Entropy (độ mờ) | ✅ Cao tốt (nếu muốn mờ) |
| **Time** | Thời gian chạy (giây) | ✅ Thấp tốt (nhanh) |
| **PSNR** | Peak Signal Noise Ratio (0-100 dB) | ✅ Cao tốt (chính xác) |
| **SSIM** | Structural Similarity (-1 to 1) | ✅ Cao tốt (giống ảnh gốc) |
| **DICE** | Sorensen-Dice (0-1, cần GT) | ✅ Cao tốt (khớp GT) |

### Khi nào là "tốt"?

- **PSNR** > 30 dB → ✅ Tốt
- **SSIM** > 0.8 → ✅ Tốt  
- **DICE** > 0.8 → ✅ Tốt (khi có GT)
- **FE** → Phụ thuộc vào mục tiêu (không có chuẩn)

---

## 💡 **Pro Tips**

1. **Muốn mau**: Dùng K=2-3, iters=20, pop=10
2. **Muốn chính xác**: Dùng K=5-6, iters=200, pop=50
3. **Muốn cân bằng**: K=4, iters=100, pop=30 (default)
4. **Kiểm tra visual**: Luôn so sánh ảnh segmentation, không chỉ metrics
5. **Giữ GT**: Nếu có ground-truth, luôn upload để so sánh DICE

---

## ❌ **Troubleshooting**

### "Kết quả xấu quá"
→ Thay K (2→5) hoặc tăng iters (100→200)

### "Mất quá lâu"
→ Giảm iters (200→50) hoặc pop_size (50→20)

### "DICE không xuất hiện"
→ Upload GT mask, nếu không có GT sẽ không tính

### "Ngưỡng là [255, 255, ...]"
→ Optimizer không converged, tăng iters hoặc đổi algorithm

### "Ảnh segmentation toàn màu 0"
→ Ngưỡng không hợp lệ, check log server

---

## 📚 **Để biết thêm**

- Giải thích chi tiết FE vs PSNR/SSIM:
  → `docs/FE_vs_PSNR_SSIM_explanation.md`

- Tất cả cải tiến UI:
  → `docs/UI_improvements_Nov15.md`

- Code reference:
  → `src/ui/app.py`, `src/ui/templates/benchmark_result.html`

---

## ✅ **Tóm tắt**

| Vấn đề | Giải pháp |
|--------|----------|
| FE cao nhưng PSNR/SSIM thấp | **Bình thường** - chúng tối ưu mục tiêu khác. Chọn cái phù hợp ứng dụng |
| Output ảnh xấu | Thay đổi K, membership, hoặc iters. Xem "Best Results Comparison" để compare |
| DICE không có | Upload GT mask nếu muốn tính DICE |
| Muốn so sánh 2 ảnh | Dùng "Best Results Comparison" (3 card) hoặc "Segmentation Comparison" (all algos) |

**Happy segmenting! 🎉**
