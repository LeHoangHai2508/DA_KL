"""
Visualization utilities for optimization convergence and performance analysis.

Module vẽ đồ thị hội tụ (convergence curves), so sánh ngưỡng, metrics hiệu năng.
Dùng matplotlib để tạo subplot grid (GridSpec) với nhiều subplot.
"""
# ===== IMPORT LIBRARIES =====
# numpy: xử lý array, tính toán số học
import numpy as np
# matplotlib.pyplot: tạo figure, subplot, vẽ đồ thị
import matplotlib.pyplot as plt
# GridSpec: chia figure thành grid (linh hoạt hơn subplot)
from matplotlib.gridspec import GridSpec
# pathlib.Path: xử lý đường dẫn file
from pathlib import Path
# typing: type hints (Dict, List, Tuple, Any)
from typing import Dict, List, Tuple, Any


def plot_convergence_curves(
    convergence_history: Dict[str, np.ndarray],
    title: str = "Convergence Curves",
    figsize: Tuple[int, int] = (14, 10),
    save_path: Path = None,
) -> plt.Figure:
    """
    Plot convergence curves for multiple algorithms.
    
    Vẽ đường hội tụ (iteration vs fitness) cho mỗi thuật toán.
    Dùng GridSpec để chia figure thành 3x3 subplot.
    
    Args:
        convergence_history: Dict mapping algo name -> array fitness values (shape: iters)
        title: Plot title
        figsize: Figure size (width, height)
        save_path: Path to save figure (optional)
    
    Returns:
        matplotlib.figure.Figure
    """
    # ===== TẠO FIGURE VỚI GRIDSPEC =====
    # plt.figure(figsize=figsize): tạo figure với kích thước (width, height)
    fig = plt.figure(figsize=figsize)
    # GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3):
    #   - Chia figure thành 3 hàng × 3 cột
    #   - hspace=0.3: khoảng cách giữa hàng (height space)
    #   - wspace=0.3: khoảng cách giữa cột (width space)
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # ===== LẤY DANH SÁCH THUẬT TOÁN =====
    # list(convergence_history.keys()): danh sách tên algo (MFWOA, WOA, PSO, Otsu)
    algorithms = list(convergence_history.keys())
    n_algos = len(algorithms)
    
    # ===== ĐỊNH NGHĨA BẢNG MÀU =====
    # colors: dict mapping tên algo (lowercase) -> mã hex (RGB)
    # Ví dụ: 'mfwoa' -> '#1f77b4' (xanh matplotlib default)
    colors = {
        'mfwoa': '#1f77b4',  # blue
        'woa': '#ff7f0e',    # orange
        'pso': '#2ca02c',    # green
        'otsu': '#d62728',   # red
        'ga': '#9467bd',     # purple
        'fcm': '#8c564b',    # brown
    }
    
    # ===== VẼ MỖI THUẬT TOÁN TRONG MỘT SUBPLOT =====
    for idx, algo in enumerate(algorithms):
        # fig.add_subplot(gs[idx]): tạo subplot tại vị trí thứ idx trong grid
        # gs[idx] tương đương gs[idx // 3, idx % 3] (row, col)
        ax = fig.add_subplot(gs[idx])
        # convergence_history[algo]: array fitness values (1D, shape: iters)
        history = convergence_history[algo]
        
        # ===== TÍNH TOÁN DỮ LIỆU VẼ =====
        # np.arange(len(history)): [0, 1, 2, ..., iters-1] (iteration numbers)
        iterations = np.arange(len(history))
        # colors.get(algo.lower(), '#1f77b4'):
        #   - Lấy màu từ dict theo tên algo (lowercase)
        #   - Nếu không tìm thấy, dùng xanh default
        color = colors.get(algo.lower(), '#1f77b4')
        
        # ===== VẼ ĐƯỜNG HỘI TỤ =====
        # ax.plot(x, y, ...): vẽ đường line
        # linewidth=2: độ dày đường (2 point)
        # color=color: màu từ bảng màu
        # label=algo.upper(): tên legend (MFWOA, WOA, ...)
        ax.plot(iterations, history, linewidth=2, color=color, label=algo.upper())
        # ax.fill_between(x, y, alpha=0.2, color=color):
        #   - Tô màu vùng dưới đường (dưới y, trên x-axis)
        #   - alpha=0.2: độ trong suốt (20% đục)
        ax.fill_between(iterations, history, alpha=0.2, color=color)
        
        # ===== THIẾT LẬP TRỤC =====
        # ax.set_xlabel("Iteration"): nhãn trục X
        ax.set_xlabel("Iteration")
        # ax.set_ylabel("Fitness Value"): nhãn trục Y (fitness = entropy)
        ax.set_ylabel("Fitness Value")
        # ax.set_title(...): tiêu đề subplot
        # fontweight='bold': đậm
        ax.set_title(f"{algo.upper()} Convergence", fontweight='bold')
        # ax.grid(True, alpha=0.3): vẽ lưới, độ trong = 0.3
        ax.grid(True, alpha=0.3)
        # ax.legend(loc='best'): vẽ legend, vị trí tự động tốt nhất
        ax.legend(loc='best')
        
        # ===== THÊM CHỈ THỊ GIÁ TRỊ CUỐI =====
        # final_value = history[-1]: giá trị fitness cuối cùng (iteration cuối)
        final_value = history[-1]
        # ax.text(...): vẽ text lên subplot
        # (0.98, 0.02): vị trí (x, y) trong tọa độ transform=ax.transAxes (0-1 scale)
        #   - (0.98, 0.02) = góc dưới phải
        # transform=ax.transAxes: tọa độ là phần trăm (0-1), không phải data coordinate
        # ha='right', va='bottom': horizontal align right, vertical align bottom
        # bbox=dict(...): vẽ hộp quanh text (boxstyle='round' = bo tròn góc)
        ax.text(
            0.98, 0.02, f"Final: {final_value:.4f}",
            transform=ax.transAxes, ha='right', va='bottom',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=9
        )
    
    # ===== VẼ TIÊU ĐỀ CHÍNH =====
    # fig.suptitle(...): tiêu đề cho toàn figure
    # fontsize=16: kích thước (16 point)
    # fontweight='bold': đậm
    # y=0.995: vị trí Y (0 = dưới, 1 = trên), 0.995 = gần trên cùng
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
    
    # ===== LƯU FIGURE =====
    # if save_path: chỉ lưu nếu user cung cấp đường dẫn
    if save_path:
        # save_path.parent.mkdir(parents=True, exist_ok=True):
        #   - Tạo thư mục parent (recursive), không error nếu tồn tại
        save_path.parent.mkdir(parents=True, exist_ok=True)
        # plt.savefig(path, dpi=150, bbox_inches='tight'):
        #   - Lưu figure thành PNG
        #   - dpi=150: độ phân giải (150 dots per inch)
        #   - bbox_inches='tight': loại bỏ margin trắng
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Convergence plot saved: {save_path}")
    
    return fig


def plot_schematic_comparison(
    results: List[Dict[str, Any]],
    n_thresholds: int,
    title: str = "Threshold Comparison",
    figsize: Tuple[int, int] = (14, 6),
    save_path: Path = None,
) -> plt.Figure:
    """
    Plot threshold distributions and FE values across algorithms.
    
    Vẽ hai subplot:
    1. Bên trái: vị trí ngưỡng cho mỗi algo (scatter + line)
    2. Bên phải: so sánh FE + PSNR (bar chart)
    
    Args:
        results: List of result dicts from run_algorithms_and_benchmark
        n_thresholds: Number of thresholds used
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure (optional)
    
    Returns:
        matplotlib.figure.Figure
    """
    # ===== TẠO FIGURE =====
    # plt.figure(figsize=figsize): tạo figure
    fig = plt.figure(figsize=figsize)
    # GridSpec(1, 2, ...): 1 hàng × 2 cột (subplot trái, phải)
    gs = GridSpec(1, 2, figure=fig, wspace=0.3)
    
    # ===== SUBPLOT TRÁI: VỊ TRÍ NGƯỠNG =====
    # fig.add_subplot(gs[0]): subplot cột 0 (trái)
    ax1 = fig.add_subplot(gs[0])
    
    # ===== CÁC BIẾN TRACKING =====
    # y_positions: dict {algo -> (y_pos, thresholds)}
    #   - y_pos: vị trí Y trên subplot (0, 1, 2, ...)
    #   - thresholds: danh sách ngưỡng
    y_positions = {}
    # colors: dict {algo -> mã hex color}
    colors = {}
    # color_palette: danh sách 6 màu
    color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    # ===== LOOP QUAS TỪNG KẾT QUẢ =====
    for idx, res in enumerate(results):
        # res: dict {algo, thresholds, fe, psnr, ssim, time, ...}
        # res.get('algo', 'unknown'): lấy tên algo, default 'unknown'
        algo = res.get('algo', 'unknown').upper()
        # res.get('thresholds', []): lấy danh sách ngưỡng, default []
        thresholds = res.get('thresholds', [])
        
        # Nếu có ngưỡng
        if thresholds:
            # y_pos: vị trí Y là index của algo (0, 1, 2, ...)
            y_pos = idx
            # Lưu (y_pos, thresholds)
            y_positions[algo] = (y_pos, thresholds)
            # Chọn màu từ palette (xoay vòng nếu vượt quá 6 màu)
            color = color_palette[idx % len(color_palette)]
            colors[algo] = color
            
            # ===== VẼ SCATTER: VỊ TRÍ NGƯỠNG =====
            # ax1.scatter(thresholds, [y_pos]*len(thresholds), ...):
            #   - X: giá trị ngưỡng (0-255)
            #   - Y: vị trí Y (y_pos) lặp lại cho mỗi ngưỡng
            #   - s=100: kích thước điểm (100 square point)
            #   - color=color: màu
            #   - alpha=0.7: độ trong 70%
            #   - zorder=3: độ sâu vẽ (cao = vẽ trên)
            #   - label=algo: tên legend
            ax1.scatter(thresholds, [y_pos] * len(thresholds), 
                       s=100, color=color, alpha=0.7, zorder=3, label=algo)
            
            # ===== VẼ ĐƯỜNG NỐI: KẾT NỐI NGƯỠNG =====
            # ax1.plot(thresholds, [y_pos]*len(thresholds), ...):
            #   - Vẽ đường nối các ngưỡng (cùng y)
            #   - linewidth=1: mảnh
            #   - alpha=0.5: mờ
            #   - zorder=1: độ sâu thấp (dưới scatter)
            ax1.plot(thresholds, [y_pos] * len(thresholds), 
                    color=color, linewidth=1, alpha=0.5, zorder=1)
    
    # ===== THIẾT LẬP TRỤC TRÁI =====
    # ax1.set_xlabel("Threshold Value (0-255)", fontweight='bold'): nhãn X
    ax1.set_xlabel("Threshold Value (0-255)", fontweight='bold')
    # ax1.set_ylabel("Algorithm", fontweight='bold'): nhãn Y
    ax1.set_ylabel("Algorithm", fontweight='bold')
    # ax1.set_xlim(-10, 265): giới hạn X (0-255 + margin)
    ax1.set_xlim(-10, 265)
    # ax1.set_ylim(-0.5, len(results) - 0.5): giới hạn Y
    ax1.set_ylim(-0.5, len(results) - 0.5)
    # ax1.set_yticks(range(len(results))): đặt Y ticks tại 0, 1, 2, ...
    ax1.set_yticks(range(len(results)))
    # ax1.set_yticklabels(...): đặt nhãn Y (tên algo)
    # [res.get('algo', 'unknown').upper() for res in results]: list comprehension
    ax1.set_yticklabels([res.get('algo', 'unknown').upper() for res in results])
    # ax1.grid(True, alpha=0.3, axis='x'): vẽ lưới (chỉ trục X)
    ax1.grid(True, alpha=0.3, axis='x')
    # ax1.set_title("Threshold Positions", fontweight='bold', fontsize=12): tiêu đề
    ax1.set_title("Threshold Positions", fontweight='bold', fontsize=12)
    
    # ===== SUBPLOT PHẢI: FE + PSNR =====
    # fig.add_subplot(gs[1]): subplot cột 1 (phải)
    ax2 = fig.add_subplot(gs[1])
    
    # ===== TRÍ TOÁN DỮ LIỆU =====
    # Lists để lưu dữ liệu
    algos = []
    fe_values = []
    psnr_values = []
    
    # Loop qua kết quả
    for res in results:
        # Lấy tên algo (uppercase)
        algo = res.get('algo', 'unknown').upper()
        algos.append(algo)
        # Lấy FE, default 0
        fe = res.get('fe', 0)
        # Lấy PSNR, default 0
        psnr = res.get('psnr', 0)
        
        # Kiểm tra FE hợp lệ (>-1000, tránh outlier)
        # Thêm vào list (hoặc 0 nếu không hợp lệ)
        fe_values.append(fe if fe and fe > -1000 else 0)
        # PSNR
        psnr_values.append(psnr if psnr else 0)
    
    # ===== VẼ BIỂU ĐỒ CỘT =====
    # np.arange(len(algos)): [0, 1, 2, ...] (X positions)
    x = np.arange(len(algos))
    # width = 0.35: độ rộng bar (35% của khoảng cách giữa bar)
    width = 0.35
    
    # ===== VẼ BAR FE =====
    # ax2.bar(x - width/2, fe_values, width, ...):
    #   - X: x - 0.175 (dịch trái 0.175)
    #   - Y: giá trị FE
    #   - width: 0.35
    #   - label='FE Value': legend
    #   - color='skyblue': màu xanh nhẹ
    #   - alpha=0.8: độ trong 80%
    bars1 = ax2.bar(x - width/2, fe_values, width, label='FE Value', color='skyblue', alpha=0.8)
    # ===== TẠO TRỤC Y PHẢI RIÊNG =====
    # ax2_twin = ax2.twinx(): tạo trục Y thứ 2 (bên phải, dùng chung trục X)
    ax2_twin = ax2.twinx()
    # ===== VẼ BAR PSNR =====
    # ax2_twin.bar(x + width/2, psnr_values, ...):
    #   - X: x + 0.175 (dịch phải 0.175)
    #   - Giá trị PSNR
    #   - color='salmon': màu hồng
    bars2 = ax2_twin.bar(x + width/2, psnr_values, width, label='PSNR (dB)', color='salmon', alpha=0.8)
    
    # ===== THIẾT LẬP TRỤC =====
    # ax2: trục Y trái (FE)
    ax2.set_xlabel("Algorithm", fontweight='bold')
    ax2.set_ylabel("Fuzzy Entropy", color='skyblue', fontweight='bold')
    # ax2_twin: trục Y phải (PSNR)
    ax2_twin.set_ylabel("PSNR (dB)", color='salmon', fontweight='bold')
    # ax2.set_xticks(x): đặt X ticks
    ax2.set_xticks(x)
    # ax2.set_xticklabels(algos): nhãn X (tên algo)
    ax2.set_xticklabels(algos)
    # Tiêu đề
    ax2.set_title("Quality Metrics", fontweight='bold', fontsize=12)
    # Lưới (chỉ trục Y)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # ===== THÊM NHÃN TRÊN BAR FE =====
    # for bar in bars1: loop qua mỗi bar
    for bar in bars1:
        # bar.get_height(): chiều cao bar (= giá trị FE)
        height = bar.get_height()
        # Nếu height > 0
        if height > 0:
            # ax2.text(...): vẽ text trên bar
            # bar.get_x() + bar.get_width()/2.: X tâm bar
            # height: Y = chiều cao
            # f'{height:.2f}': format 2 chữ số sau dấu phẩy
            # ha='center', va='bottom': căn giữa X, căn dưới Y
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=9)
    
    # ===== THÊM NHÃN TRÊN BAR PSNR =====
    for bar in bars2:
        height = bar.get_height()
        if height > 0:
            # Dùng ax2_twin để vẽ (trục Y phải)
            ax2_twin.text(bar.get_x() + bar.get_width()/2., height,
                         f'{height:.1f}', ha='center', va='bottom', fontsize=9)
    
    # ===== TẠO LEGEND HỢP NHẤT =====
    # ax2.get_legend_handles_labels(): lấy legend từ ax2
    lines1, labels1 = ax2.get_legend_handles_labels()
    # ax2_twin.get_legend_handles_labels(): lấy legend từ ax2_twin
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    # ax2.legend(lines1 + lines2, labels1 + labels2, ...): nối 2 legend
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    # ===== TIÊU ĐỀ CHÍNH =====
    # f"{title} (K={n_thresholds})": thêm số ngưỡng
    fig.suptitle(f"{title} (K={n_thresholds})", fontsize=14, fontweight='bold', y=0.98)
    
    # ===== LƯU FIGURE =====
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Schematic plot saved: {save_path}")
    
    return fig


def plot_performance_metrics(
    results: List[Dict[str, Any]],
    figsize: Tuple[int, int] = (14, 6),
    save_path: Path = None,
) -> plt.Figure:
    """
    Plot execution time and metric comparisons.
    
    Vẽ hai subplot:
    1. Bên trái: thời gian chạy (bar chart ngang)
    2. Bên phải: SSIM (bar chart ngang)
    
    Args:
        results: List of result dicts
        figsize: Figure size
        save_path: Path to save figure (optional)
    
    Returns:
        matplotlib.figure.Figure
    """
    # ===== TẠO FIGURE =====
    fig = plt.figure(figsize=figsize)
    # GridSpec(1, 2): 1 hàng × 2 cột
    gs = GridSpec(1, 2, figure=fig, wspace=0.3)
    
    # ===== LẤY DỮ LIỆU =====
    # List comprehension: lấy tên algo từ mỗi result
    algos = [res.get('algo', 'unknown').upper() for res in results]
    # Lấy thời gian chạy (mặc định 0 nếu không có)
    times = [res.get('time', 0) for res in results]
    # Lấy SSIM score
    ssim_values = [res.get('ssim', 0) for res in results]
    
    # ===== SUBPLOT TRÁI: THỜI GIAN CHẠY =====
    ax1 = fig.add_subplot(gs[0])
    # Bảng màu
    color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    # Lặp lại màu nếu vượt quá 6
    colors = [color_palette[i % len(color_palette)] for i in range(len(algos))]
    
    # ===== VẼ BAR NGANG THỜI GIAN =====
    # ax1.barh(algos, times, color=colors, alpha=0.8):
    #   - algos: Y labels (tên algo)
    #   - times: X values (thời gian)
    #   - color=colors: list màu
    #   - alpha=0.8: độ trong 80%
    bars = ax1.barh(algos, times, color=colors, alpha=0.8)
    # Nhãn trục
    ax1.set_xlabel("Execution Time (seconds)", fontweight='bold')
    # Tiêu đề
    ax1.set_title("Algorithm Execution Time", fontweight='bold', fontsize=12)
    # Lưới (trục X)
    ax1.grid(True, alpha=0.3, axis='x')
    
    # ===== THÊM NHÃN THỜI GIAN TRÊN BAR =====
    # for bar, time_val in zip(bars, times):
    #   - zip: kết nối bar object với giá trị thời gian
    for bar, time_val in zip(bars, times):
        # bar.get_width(): chiều rộng bar (= thời gian)
        width = bar.get_width()
        # ax1.text(...): vẽ text trên bar
        # (width, y_pos_bar): vị trí (right edge, center of bar)
        # ' {time_val:.3f}s': format 3 chữ số sau dấu phẩy + 's'
        # ha='left', va='center': align trái, căn giữa Y
        ax1.text(width, bar.get_y() + bar.get_height()/2.,
                f' {time_val:.3f}s', ha='left', va='center', fontsize=10, fontweight='bold')
    
    # ===== SUBPLOT PHẢI: SSIM =====
    ax2 = fig.add_subplot(gs[1])
    
    # ===== VẼ BAR NGANG SSIM =====
    bars = ax2.barh(algos, ssim_values, color=colors, alpha=0.8)
    # Nhãn trục
    ax2.set_xlabel("SSIM Score", fontweight='bold')
    # Giới hạn X (SSIM từ 0 đến 1)
    ax2.set_xlim(0, 1)
    # Tiêu đề
    ax2.set_title("Segmentation Quality (SSIM)", fontweight='bold', fontsize=12)
    # Lưới (trục X)
    ax2.grid(True, alpha=0.3, axis='x')
    
    # ===== THÊM NHÃN SSIM TRÊN BAR =====
    for bar, ssim_val in zip(bars, ssim_values):
        width = bar.get_width()
        # Format 4 chữ số sau dấu phẩy
        ax2.text(width, bar.get_y() + bar.get_height()/2.,
                f' {ssim_val:.4f}', ha='left', va='center', fontsize=10, fontweight='bold')
    
    # ===== TIÊU ĐỀ CHÍNH =====
    fig.suptitle("Performance Comparison", fontsize=14, fontweight='bold', y=0.98)
    
    # ===== LƯU FIGURE =====
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Performance metrics plot saved: {save_path}")
    
    return fig


def plot_histogram_with_thresholds(
    image: np.ndarray,
    thresholds_dict: Dict[str, List[int]],
    figsize: Tuple[int, int] = (14, 8),
    save_path: Path = None,
) -> plt.Figure:
    """
    Plot image histogram with threshold markers from different algorithms.
    
    Vẽ histogram ảnh + vị trí ngưỡng (vertical lines) từ mỗi algo.
    Layout: 1 subplot lớn (top) + 3 subplot nhỏ (bottom).
    
    Args:
        image: Grayscale image (2D array)
        thresholds_dict: Dict mapping algo name -> list of thresholds
        figsize: Figure size
        save_path: Path to save figure (optional)
    
    Returns:
        matplotlib.figure.Figure
    """
    # ===== TẠO FIGURE =====
    fig = plt.figure(figsize=figsize)
    # GridSpec(2, 2): 2 hàng × 2 cột, nhưng hàng 0 dùng toàn bộ 2 cột
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # ===== TÍNH HISTOGRAM =====
    # cv2.calcHist([image], [0], None, [256], [0, 256]):
    #   - [image]: list ảnh input
    #   - [0]: channel 0 (grayscale)
    #   - None: không dùng mask
    #   - [256]: số bin (0-255 -> 256 bin)
    #   - [0, 256]: range (0-256)
    # Kết quả: array shape (256, 1), cần .flatten() thành 1D
    hist = cv2.calcHist([image], [0], None, [256], [0, 256]).flatten()
    
    # ===== SUBPLOT TOP: HISTOGRAM + TẤT CẢ NGƯỠNG =====
    # gs[0, :]: hàng 0, cột 0:2 (toàn bộ chiều rộng)
    ax1 = fig.add_subplot(gs[0, :])
    # ===== VẼ HISTOGRAM =====
    # ax1.bar(range(256), hist, color='gray', alpha=0.6, width=1):
    #   - X: 0-255 (pixel intensity)
    #   - Y: tần số
    #   - color='gray': xám
    #   - alpha=0.6: độ trong 60%
    #   - width=1: mỗi bar = 1 pixel
    ax1.bar(range(256), hist, color='gray', alpha=0.6, width=1)
    # Nhãn trục
    ax1.set_xlabel("Pixel Intensity", fontweight='bold')
    ax1.set_ylabel("Frequency", fontweight='bold')
    # Tiêu đề
    ax1.set_title("Image Histogram with Thresholds", fontweight='bold', fontsize=12)
    # Lưới (trục Y)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # ===== VẼ CÁC ĐƯỜNG NGƯỠNG =====
    # Bảng màu
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    # for idx, (algo, thresholds) in enumerate(thresholds_dict.items()):
    #   - thresholds_dict.items(): list [(algo_name, thresholds_list), ...]
    for idx, (algo, thresholds) in enumerate(thresholds_dict.items()):
        # color: chọn màu từ palette (xoay vòng)
        color = colors[idx % len(colors)]
        # for threshold in thresholds: loop qua mỗi ngưỡng
        for threshold in thresholds:
            # ax1.axvline(threshold, color=color, linestyle='--', linewidth=2, alpha=0.7, label=...):
            #   - Vẽ đường thẳng đứng tại X=threshold
            #   - linestyle='--': nét đứt (dashed)
            #   - linewidth=2: độ dày 2 point
            #   - alpha=0.7: độ trong 70%
            #   - label=algo nếu ngưỡng đầu tiên, rồi '' cho ngưỡng tiếp theo (tránh duplicate legend)
            # Điều kiện: "threshold == thresholds[0]" = ngưỡng đầu tiên
            ax1.axvline(threshold, color=color, linestyle='--', linewidth=2, alpha=0.7, 
                       label=algo if threshold == thresholds[0] else '')
    
    # ===== LOẠI BỎ DUPLICATE LEGEND =====
    # ax1.get_legend_handles_labels(): lấy handles (line objects) và labels (names)
    handles, labels = ax1.get_legend_handles_labels()
    # dict(zip(labels, handles)): tạo dict {label: handle}
    #   - Nếu label duplicate, giữ lại một (dict key unique)
    by_label = dict(zip(labels, handles))
    # ax1.legend(by_label.values(), by_label.keys(), ...): vẽ legend (remove duplicate)
    ax1.legend(by_label.values(), by_label.keys(), loc='upper right')
    
    # ===== SUBPLOT BOTTOM: TỪ TỪNG ALGO (TỐI ĐA 3) =====
    # enumerate(thresholds_dict.items()): [(idx, (algo, thresholds)), ...]
    for idx, (algo, thresholds) in enumerate(thresholds_dict.items()):
        # Chỉ vẽ 3 algo đầu tiên (để tránh quá đông)
        if idx >= 3:
            break
        # gs[1, idx]: hàng 1, cột idx
        ax = fig.add_subplot(gs[1, idx])
        # ===== VẼ HISTOGRAM =====
        ax.bar(range(256), hist, color='gray', alpha=0.6, width=1)
        
        # ===== VẼ NGƯỠNG CỦA ALGO NÀY =====
        color = colors[idx % len(colors)]
        for threshold in thresholds:
            # Đường ngưỡng (mạnh hơn top plot)
            # linewidth=2.5: độ dày (mạnh hơn 2)
            # alpha=0.8: độ trong (đậm hơn 0.7)
            ax.axvline(threshold, color=color, linestyle='--', linewidth=2.5, alpha=0.8)
        
        # ===== THIẾT LẬP TRỤC =====
        # fontsize=10: nhỏ hơn top plot
        ax.set_xlabel("Intensity", fontweight='bold', fontsize=10)
        ax.set_ylabel("Frequency", fontweight='bold', fontsize=10)
        # Tiêu đề: tên algo + ngưỡng
        # str(thresholds): [100, 180, ...] -> '[100, 180, ...]'
        ax.set_title(f"{algo.upper()}: {thresholds}", fontweight='bold', fontsize=11)
        # Lưới
        ax.grid(True, alpha=0.3, axis='y')
    
    # ===== TIÊU ĐỀ CHÍNH =====
    fig.suptitle("Threshold Distribution Analysis", fontsize=14, fontweight='bold', y=0.995)
    
    # ===== LƯU FIGURE =====
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Histogram plot saved: {save_path}")
    
    return fig


# ===== HELPER FUNCTIONS =====

def create_convergence_data_for_algorithm(
    optimizer_instance,
    fitness_history: List[float]
) -> np.ndarray:
    """Extract convergence curve from optimizer history.
    
    Chuyển list fitness values thành numpy array.
    
    Args:
        optimizer_instance: Optimizer object (unused, nhưng có thể dùng sau)
        fitness_history: List fitness values qua iterations
    
    Returns:
        np.ndarray: shape (iters,)
    """
    # np.array(fitness_history): chuyển list -> numpy array
    return np.array(fitness_history)


def save_all_plots(
    results: List[Dict[str, Any]],
    convergence_histories: Dict[str, np.ndarray],
    image: np.ndarray,
    n_thresholds: int,
    output_dir: Path = Path("results/plots")
) -> None:
    """
    Generate and save all visualization plots.
    
    Wrapper hàm: gọi toàn bộ hàm vẽ (convergence, schematic, metrics, histogram).
    Tạo thư mục output nếu chưa tồn tại.
    
    Args:
        results: Benchmark results (list of algo result dicts)
        convergence_histories: Dict {algo_name -> convergence_array}
        image: Original image (for histogram)
        n_thresholds: Number of thresholds
        output_dir: Output directory for plots (default "results/plots")
    """
    # Path(output_dir): convert to Path object
    output_dir = Path(output_dir)
    # mkdir(parents=True, exist_ok=True): tạo thư mục recursive
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ===== PRINT THÔNG BÁO =====
    print(f"\n📊 Generating visualization plots...")
    
    # ===== VẼ ĐƯỜNG HỘI TỤ =====
    # if convergence_histories: chỉ vẽ nếu có dữ liệu
    if convergence_histories:
        # plot_convergence_curves(...): vẽ convergence
        # save_path=output_dir / "convergence_curves.png": lưu PNG
        plot_convergence_curves(
            convergence_histories,
            title=f"Algorithm Convergence (K={n_thresholds})",
            save_path=output_dir / "convergence_curves.png"
        )
    
    # ===== VẼ SO SÁNH NGƯỠNG =====
    # plot_schematic_comparison(...): vẽ vị trí ngưỡng + metrics
    plot_schematic_comparison(
        results,
        n_thresholds,
        title="Threshold Comparison",
        save_path=output_dir / "threshold_comparison.png"
    )
    
    # ===== VẼ HIỆU NĂNG =====
    # plot_performance_metrics(...): vẽ thời gian + SSIM
    plot_performance_metrics(
        results,
        save_path=output_dir / "performance_metrics.png"
    )
    
    # ===== VẼ HISTOGRAM =====
    # Tạo dict {algo -> thresholds} từ results
    # {res.get('algo', 'unknown'): res.get('thresholds', []) for res in results}
    thresholds_dict = {res.get('algo', 'unknown'): res.get('thresholds', []) for res in results}
    plot_histogram_with_thresholds(
        image,
        thresholds_dict,
        save_path=output_dir / "histogram_thresholds.png"
    )
    
    # ===== PRINT KẾT QUẢ =====
    print(f"✓ All plots saved to {output_dir}")


# ===== IMPORT CV2 (Optional) =====
# Cố gắng import cv2 (nếu không có, set cv2 = None)
# Module này dùng cv2.calcHist trong plot_histogram_with_thresholds
try:
    # Nếu cv2 đã được import ở đầu file, không cần import lại
    # Nhưng để chắc chắn, ta thêm dòng này
    import cv2
except ImportError:
    # Nếu cv2 không cài đặt, set cv2 = None
    # Hàm plot_histogram_with_thresholds sẽ fail nếu gọi (cv2.calcHist sẽ raise error)
    cv2 = None
