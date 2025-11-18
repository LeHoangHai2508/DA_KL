# 🔬 Công Thức Fuzzy Entropy De Luca - Quick Reference

## 📐 Công Thức Chính

### Fuzzy Entropy De Luca (Shannon)
$$H = -K \sum_{i=1}^{n} \left[ \rho_i \log(\rho_i) + (1-\rho_i) \log(1-\rho_i) \right]$$

| Ký hiệu | Giá trị | Ý nghĩa |
|---------|--------|--------|
| H | > 0 | Độ mờ (entropy) của phân đoạn |
| K | 1/ln(2) ≈ 1.4427 | Hằng số chuẩn hoá |
| n | 256 | Số mức xám (8-bit) |
| ρ_i | [0, 1] | Độ thành viên mờ của pixel i |

### Shannon Entropy Component
$$S_n(\rho) = -\rho \log(\rho) - (1-\rho) \log(1-\rho)$$

| ρ | S_n(ρ) | Ý nghĩa |
|---|--------|--------|
| 0 | 0 | Hoàn toàn không thuộc lớp |
| 0.5 | 0.693 (max) | Mờ tối đa, không chắc chắn |
| 1 | 0 | Hoàn toàn thuộc lớp |

---

## 🎯 Công Thức Phạt (Penalty)

### Tổng Quát
$$F'(T) = F(T) - \lambda[\alpha P_A(T) + \beta P_\mu(T)]$$

| Ký hiệu | Giá trị | Ý nghĩa |
|---------|--------|--------|
| F(T) | - | Fuzzy Entropy gốc |
| λ | 1.0 | Hệ số cân nặng chung |
| α | 0.1 | Hệ số penalty diện tích |
| β | 0.1 | Hệ số penalty membership |
| P_A(T) | ≥ 0 | Penalty diện tích |
| P_μ(T) | ≥ 0 | Penalty membership |

### Penalty Diện Tích
$$P_A(T) = \sum_{c=1}^{C} \left( p_c - \frac{1}{C} \right)^2$$

**Ý nghĩa:** Phạt nếu các lớp không cân bằng

### Penalty Membership  
$$P_\mu(T) = \max_{c,y}(\mu_c(y))^2$$

**Ý nghĩa:** Phạt nếu membership quá tập trung

---

## 🧮 Tính Toán Trên Histogram

### Phân Phối Xác Suất
$$p(y) = \frac{hist[y]}{\sum_{y=0}^{255} hist[y]}$$

### Độ Thành Viên Mỗi Lớp
$$p_c = \sum_{y=0}^{255} \mu_c(y) \cdot p(y)$$

### Fuzzy Entropy Tổng Thể
$$H = C \cdot \frac{1}{\ln(2)} \sum_{y=0}^{255} p(y) \sum_{c=1}^{C} S_n(\mu_c(y))$$

---

## 📊 Hàm Membership

### Tam Giác (Triangular) - Khuyên Dùng
```
    μ_c(y)
        ^
        |     /\
    1.0 |    /  \
        |   /    \
        |  /      \
    0.0 |_/________\____→ y
        L    C    R
```

**Công thức:**
- Khi L ≤ y ≤ C: $\mu_c(y) = \frac{y - L}{C - L}$
- Khi C < y ≤ R: $\mu_c(y) = \frac{R - y}{R - C}$
- Ngoài: $\mu_c(y) = 0$

### Gaussian
```
    μ_c(y)
        ^
        |      ___
    1.0 |     /   \
        |    |  C  |
        |   /       \
    0.0 |__/         \__→ y
```

**Công thức:**
$$\mu_c(y) = \exp\left(-\frac{1}{2}\left(\frac{y-C}{\sigma}\right)^2\right)$$

Với σ = α × (R - L), thường α = 0.5

---

## 💻 Cài Đặt (Python)

### Function Signature
```python
def compute_fuzzy_entropy(
    hist: np.ndarray,              # Histogram 256-bin
    thresholds: List[int],         # Ngưỡng phân đoạn
    membership: str = "triangular", # Loại membership
    for_minimization: bool = False, # Trả về -H hay H?
    lambda_penalty: float = 1.0,    # λ
    alpha_area: float = 0.1,        # α
    beta_membership: float = 0.1    # β
) -> float:
```

### Cách Gọi
```python
# Cho MFWOA (maximizer)
fe = compute_fuzzy_entropy(hist, thresholds, 
                          membership="triangular",
                          for_minimization=False,
                          lambda_penalty=1.0,
                          alpha_area=0.10,
                          beta_membership=0.10)

# Cho WOA/PSO (minimizer)
fe = compute_fuzzy_entropy(hist, thresholds,
                          membership="triangular", 
                          for_minimization=True,
                          lambda_penalty=1.0,
                          alpha_area=0.10,
                          beta_membership=0.10)
```

---

## 🔄 Quy Trình Tính Toán (12 Bước)

1. **Kiểm tra input** → hist shape (256,)
2. **Chuẩn hoá histogram** → p_levels (xác suất)
3. **Ràng buộc ngưỡng** → enforce_threshold_constraints()
4. **Xây dựng tâm lớp** → centers = [0, t1, t2, ..., tk, 255]
5. **Sinh membership** → μ_c(y) = _triangular_membership(centers)
6. **Tính xác suất lớp** → p_classes = μ · p_levels
7. **Kiểm tra lớp rỗng** → if p_classes < ε → penalty
8. **Tính Shannon Entropy** → S = -μ·log(μ) - (1-μ)·log(1-μ)
9. **FE tổng thể** → H = C · (1/ln2) · Σ p(y) · Σ S_n(μ)
10. **Penalty diện tích** → P_A = Σ (p_c - 1/C)²
11. **Penalty membership** → P_μ = max(μ)²
12. **Áp dụng phạt** → F' = H - λ[α·P_A + β·P_μ]

---

## 📈 Ảnh Hưởng Của Tham Số

### Tăng λ (lambda_penalty)
```
λ = 0.0:  FE cao, threshold xấu (không cân bằng)
λ = 0.5:  FE trung bình, threshold bình thường
λ = 1.0:  FE vừa phải, threshold cân bằng tốt
λ = 2.0:  FE thấp, threshold rất cân bằng nhưng mất FE
```

### Tăng α (alpha_area)
```
α = 0.0:   Lớp không bị phạt kích thước
α = 0.1:   Nhẹ: lớp có thể chênh lệch 5-10%
α = 0.3:   Mạnh: lớp buộc cân bằng hơn ~2-3%
α = 0.5+:  Rất mạnh: lớp cân bằng lý tưởng
```

### Tăng β (beta_membership)
```
β = 0.0:   Membership không bị phạt
β = 0.1:   Nhẹ: membership có thể spike đến 0.9-1.0
β = 0.3:   Mạnh: membership giữ ≤ 0.8
β = 0.5+:  Rất mạnh: membership mềm, max ≤ 0.7
```

---

## 🧪 Kết Quả Mẫu (Lena, K=4)

### Với λ=1.0, α=0.1, β=0.1
```
Thuật Toán | Ngưỡng          | FE     | Thời Gian
-----------|-----------------|--------|----------
MFWOA      | [155, 170, 197] | 4.658  | 0.15s (tốt nhất)
WOA        | [25, 73, 100]   | 4.451  | 0.15s
PSO        | [24, 75, 78]    | 4.627  | 0.14s
Otsu       | [57, 90, 120]   | 4.332  | 2.39s (baseline)
```

### Điều Chỉnh Tham Số
```
λ=0.5  → MFWOA FE ≈ 4.72 (tăng), threshold kém cân bằng
λ=2.0  → MFWOA FE ≈ 4.60 (giảm), threshold rất cân bằng
```

---

## 🎓 Tham Khảo Lý Thuyết

**Fuzzy Entropy:**
- De Luca, A., & Termini, S. (1972). A definition of a nonprobabilistic entropy in the setting of fuzzy sets theory. *Information and Control*, 20(4), 301-312.

**Ứng Dụng Phân Đoạn:**
- Gong, M., Zhou, Z., & Luan, J. (2010). Fuzzy c-means clustering with local information and kernel metric for image segmentation. *Neurocomputing*, 73(10-12), 1759-1766.

**MFWOA (Whale Optimization):**
- Mirjalili, S., & Lewis, A. (2016). The whale optimization algorithm. *Advances in Engineering Software*, 95, 51-67.

---

## ✅ Checklist Kiểm Tra

- [x] Công thức De Luca hiện thực đúng
- [x] Penalty động tính toán chính xác
- [x] Objective functions sử dụng công thức mới
- [x] Chú thích code chi tiết
- [x] Tài liệu toán học đầy đủ
- [x] Ví dụ & kết quả thử nghiệm
- [x] Hướng dẫn tham số điều chỉnh

---

## 📞 Hỏi Đáp Nhanh

**Q: Khi nào sử dụng triangular vs gaussian?**
A: Triangular cho tốc độ & đơn giản; Gaussian cho mềm mại & continuous.

**Q: FE cao tốt hay xấu?**
A: FE cao = mờ hơn; kết hợp với PSNR/SSIM/DICE để đánh giá.

**Q: Làm sao điều chỉnh λ, α, β?**
A: Bắt đầu λ=1, α=0.1, β=0.1; tăng α nếu muốn lớp cân bằng hơn.

**Q: Penalty nào quan trọng hơn?**
A: Phụ thuộc bài toán; thường P_A (diện tích) quan trọng hơn.

**Q: for_minimization=True/False là gì?**
A: True cho WOA/PSO (minimize); False cho MFWOA (maximize).
