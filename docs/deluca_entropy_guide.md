# Hướng Dẫn Fuzzy Entropy De Luca với Công Thức Phạt

## 1. Công Thức Fuzzy Entropy De Luca

### 1.1 Định Nghĩa Toán Học

**Công thức chính:**
$$H = -K \sum_{i=1}^{n} \left[ \rho_i \log(\rho_i) + (1-\rho_i) \log(1-\rho_i) \right]$$

Trong đó:
- **H**: Fuzzy Entropy (giá trị entropy mờ)
- **K**: Hằng số chuẩn hoá = $\frac{1}{\ln(2)}$ ≈ 1.4427
- **n**: Số mức xám trong ảnh (256 cho ảnh 8-bit)
- **ρ_i** (rho_i): Độ thành viên mờ (membership) của pixel i trong lớp được chọn
  - ρ_i ∈ [0, 1]
  - ρ_i = 1: hoàn toàn thuộc lớp (membership cao)
  - ρ_i = 0.5: không chắc chắn (mờ tối đa)
  - ρ_i = 0: hoàn toàn không thuộc lớp

### 1.2 Shannon Entropy Component

Thành phần entropy Shannon cho mỗi pixel:
$$S_n(\rho) = -\rho \log(\rho) - (1-\rho) \log(1-\rho)$$

**Ý nghĩa:**
- Khi ρ = 0 hoặc 1: S_n = 0 (không mờ, entropy thấp, dự đoán chắc chắn)
- Khi ρ = 0.5: S_n = log(2) ≈ 0.693 (mờ cực đại, entropy cao, không chắc chắn)

### 1.3 Công Thức Tính FE Trên Histogram

Trên histogram 1D:
$$H = C \cdot \frac{1}{\ln(2)} \sum_{y=0}^{255} p(y) \sum_{c=1}^{C} S_n(\mu_c(y))$$

Trong đó:
- **C**: Số lớp (= k + 1, với k là số ngưỡng)
- **p(y)**: Xác suất mức xám y = hist[y] / tổng_pixel
- **μ_c(y)**: Độ thành viên của mức xám y trong lớp c
  - Phụ thuộc vào hàm membership (tam giác, Gaussian,...)
  - Được tính từ vị trí các ngưỡng

---

## 2. Công Thức Phạt (Penalty Function)

### 2.1 Công Thức Tổng Quát

**Fuzzy Entropy với phạt:**
$$F'(T) = F(T) - \lambda[\alpha P_A(T) + \beta P_\mu(T)]$$

Trong đó:
- **F'(T)**: Fuzzy Entropy đã điều chỉnh (có phạt)
- **F(T)**: Fuzzy Entropy gốc (không phạt)
- **λ** (lambda): Hệ số cân nặng chung (0 ≤ λ ≤ 1, thường λ=1.0)
- **α**: Hệ số cho penalty diện tích (thường α=0.1)
- **β**: Hệ số cho penalty membership (thường β=0.1)
- **P_A(T)**: Penalty diện tích (phạt lớp không cân bằng)
- **P_μ(T)**: Penalty membership (phạt membership quá tập trung)

### 2.2 Penalty Diện Tích - P_A(T)

**Mục đích:** Khuyến khích các lớp có kích thước cân bằng

**Công thức:**
$$P_A(T) = \sum_{c=1}^{C} \left( p_c - \frac{1}{C} \right)^2$$

Trong đó:
- **p_c**: Xác suất (kích thước tương đối) của lớp c
  - $p_c = \sum_{y=0}^{255} \mu_c(y) \cdot p(y)$
- **1/C**: Xác suất trung bình lý tưởng (mỗi lớp = 1/C)

**Ý nghĩa:**
- P_A = 0: Tất cả lớp có kích thước bằng nhau (lý tưởng)
- P_A > 0: Có lớp chênh lệch về kích thước (cần phạt)

**Ví dụ:** 
- Nếu K=4, lý tưởng mỗi lớp có 25% pixel
- Nếu lớp 1 có 40%, lớp 2 có 5% → P_A lớn → bị phạt

### 2.3 Penalty Membership - P_μ(T)

**Mục đích:** Phạt khi membership quá tập trung (spike)

**Công thức:**
$$P_\mu(T) = \max_{c,y} (\mu_c(y))^2$$

Hoặc có thể là:
$$P_\mu(T) = -\sum_{c=1}^{C} p_c \log(p_c) \quad \text{(entropy thấp)}$$

**Ý nghĩa:**
- Phạt nếu membership ở một pixel quá cao (dẫn đến overfitting)
- Khuyến khích membership mềm, phân tán

---

## 3. Cài Đặt Trong Code

### 3.1 Hàm compute_fuzzy_entropy

```python
def compute_fuzzy_entropy(
    hist: np.ndarray,
    thresholds: Sequence[int],
    membership: MembershipType = "triangular",
    for_minimization: bool = False,
    lambda_penalty: float = 1.0,      # Hệ số λ
    alpha_area: float = 0.1,          # Hệ số α
    beta_membership: float = 0.1,     # Hệ số β
) -> float:
```

**Các bước tính toán:**

1. **Chuẩn hoá histogram → phân phối xác suất**
   ```python
   p_levels = hist / sum(hist)  # p(y) cho y=0..255
   ```

2. **Tạo ma trận membership μ_c(y)**
   ```python
   mu = _triangular_membership(centers)  # shape (C, 256)
   # μ_c(y) = độ thành viên của mức xám y trong lớp c
   ```

3. **Tính xác suất mỗi lớp**
   ```python
   p_classes = mu.dot(p_levels)  # p_c = Σ_y μ_c(y) * p(y)
   ```

4. **Kiểm tra lớp rỗng**
   ```python
   if any(p_classes < ε):  # Lớp rỗng -> phạt nặng
       return FITNESS_PENALTY
   ```

5. **Tính Shannon Entropy cho từng pixel-lớp**
   ```python
   S = -μ * log(μ) - (1-μ) * log(1-μ)  # shape (C, 256)
   ```

6. **Tính Fuzzy Entropy tổng thể**
   ```python
   H = C * (1/ln(2)) * Σ_y p(y) * Σ_c S_n(μ_c(y))
   ```

7. **Tính Penalty Diện Tích**
   ```python
   mean_prob = 1.0 / C
   P_A = Σ_c (p_c - mean_prob)²
   ```

8. **Tính Penalty Membership**
   ```python
   P_μ = max(μ)²  # Giá trị membership lớn nhất
   ```

9. **Áp dụng công thức phạt**
   ```python
   F'(T) = F(T) - λ[α·P_A + β·P_μ]
   ```

10. **Trả về kết quả**
    ```python
    return -F' nếu for_minimization else F'
    ```

### 3.2 Các Tham Số Đề Xuất

| Tham số | Giá Trị | Mô Tả |
|--------|---------|-------|
| λ (lambda_penalty) | 1.0 | Hệ số cân nặng chung (dùng 1.0 cho ảnh cân bằng) |
| α (alpha_area) | 0.1 | Penalty diện tích (0.05-0.20) |
| β (beta_membership) | 0.1 | Penalty membership (0.05-0.20) |

**Hướng dẫn điều chỉnh:**
- **Tăng λ**: Phạt mạnh hơn → thresholds cân bằng hơn nhưng có thể mất FE cao
- **Tăng α**: Phạt lớp không cân bằng → lớp có kích thước đồng đều
- **Tăng β**: Phạt membership spike → membership mềm hơn

---

## 4. Hàm Membership

### 4.1 Tam Giác (Triangular)

```
μ_c(y)
  ^
  |     /\
1 |    /  \
  |   /    \
  |  /      \
0 |_/________\____→ y
  left  c  right
```

**Công thức:**
- Từ left → c: μ = (y - left) / (c - left)
- Từ c → right: μ = (right - y) / (right - c)
- Ngoài [left, right]: μ = 0

**Ưu điểm:**
- Đơn giản, nhanh, dễ hiểu
- Thường dùng cho ứng dụng thực tế

### 4.2 Gaussian

```
μ_c(y)
  ^
  |      ___
1 |     /   \
  |    |  c  |
  |   /       \
0 |__/         \__→ y
  left      right
```

**Công thức:**
$$\mu_c(y) = \exp\left(-\frac{1}{2}\left(\frac{y-c}{\sigma}\right)^2\right)$$

Với σ = α × (right - left)

**Ưu điểm:**
- Mềm mại, smooth
- Membership tại biên không bằng 0 (chuyển tiếp êm)

---

## 5. Ví Dụ Cụ Thể

### 5.1 Tính FE cho Lena (K=4 threshold)

**Input:**
- Ảnh: Lena (512×512 grayscale)
- Ngưỡng: [57, 90, 120, 155]
- Membership: tam giác
- Công thức: De Luca với λ=1, α=0.1, β=0.1

**Output:**
```
Otsu:   FE ≈ 4.332
MFWOA:  FE ≈ 4.658
WOA:    FE ≈ 4.451
PSO:    FE ≈ 4.627
```

**Giải thích:**
- MFWOA đạt FE cao nhất vì sử dụng tối ưu hoá (không bị giới hạn như Otsu)
- Công thức phạt làm cho ngưỡng cân bằng, không tập trung vào một phần

### 5.2 Tác Động Của Penalty

**Không phạt (λ=0):**
- FE có thể cao hơn
- Nhưng threshold có thể không cân bằng (ví dụ: tập trung ở một phần)

**Với phạt (λ=1):**
- FE thấp hơn một chút
- Nhưng threshold cân bằng, các lớp có kích thước đồng đều

---

## 6. Các Công Thức Sử Dụng Trong Thực Nghiệm

### 6.1 PSNR (Peak Signal-to-Noise Ratio)

$$\text{PSNR} = 10 \log_{10} \left( \frac{L^2}{MSE} \right)$$

- L = 255 (max value)
- MSE = mean squared error giữa ảnh gốc và ảnh phân đoạn được tái cấu trúc

### 6.2 SSIM (Structural Similarity Index)

$$\text{SSIM}(x,y) = \frac{(2\mu_x\mu_y + c_1)(2\sigma_{xy} + c_2)}{(\mu_x^2 + \mu_y^2 + c_1)(\sigma_x^2 + \sigma_y^2 + c_2)}$$

- Đo độ tương tự cấu trúc giữa hai ảnh
- SSIM ∈ [-1, 1], cao hơn = tốt hơn

### 6.3 DICE (Dice Similarity Coefficient)

$$\text{DICE} = \frac{2|X \cap Y|}{|X| + |Y|}$$

- X: tập hợp pixel foreground trong segmentation
- Y: tập hợp pixel foreground trong ground truth
- DICE ∈ [0, 1], cao hơn = tốt hơn

---

## 7. Tóm Tắt & Lời Khuyến Cáo

**Fuzzy Entropy De Luca:**
- ✅ Tính toán mức độ mờ (uncertainty) của phân đoạn
- ✅ Có thể điều chỉnh qua hàm membership
- ✅ Kết hợp penalty để cân bằng thresholds

**Khi sử dụng:**
- 📌 FE cao ≠ phân đoạn tốt (cần cân bằng FE + PSNR/SSIM/DICE)
- 📌 Penalty giúp tránh threshold "xấu" (không cân bằng)
- 📌 Điều chỉnh λ, α, β tuỳ theo bài toán cụ thể

**Tham khảo:** 
- De Luca, A., & Termini, S. (1972). A definition of a nonprobabilistic entropy in the setting of fuzzy sets theory.
- Gong, M., Zhou, Z., & Luan, J. (2010). Fuzzy c-means clustering with local information and kernel metric for image segmentation.
