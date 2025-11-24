# src/ui/app.py
"""
Run: python -m src.ui.app
Flask UI that imports compute_fuzzy_entropy from src.metrics.fuzzy_entropy and
optionally uses an optimizer from src.optim.* (if available). If no optimizer
is found, a fallback simple WOA-like optimizer is used for demos.
"""
from time import time
import os
import base64
import random

from flask import Flask, request, render_template, url_for, jsonify
import threading
import uuid
import json
import numpy as np
from io import BytesIO
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


from skimage import img_as_float
from skimage.metrics import peak_signal_noise_ratio as sk_ms_psnr
from skimage.metrics import structural_similarity as sk_ms_ssim
import pandas as pd
import csv
import time as _time
# Import compute_fuzzy_entropy from the project's metrics module.
# Ensure you run this module from project root so package imports work.
from src.metrics.fuzzy_entropy import compute_fuzzy_entropy

app = Flask(
    __name__,
    template_folder=os.path.join(os.path.dirname(__file__), "templates"),
    static_folder=os.path.join(os.path.dirname(__file__), "static"),
)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB


def image_to_histogram(pil_image):
    """Convert PIL Image (grayscale) to 256-bin histogram."""
    img = pil_image.convert("L")
    arr = np.asarray(img).flatten()
    # Use range=(0,256) so intensity 255 is included (np.histogram upper bound exclusive)
    hist, _ = np.histogram(arr, bins=256, range=(0, 256))
    return hist


def pil_to_data_url(pil_image, fmt="PNG", resize_max=None):
    """Convert PIL image to data URL (base64) for inline display."""
    img = pil_image.copy()
    if resize_max:
        img.thumbnail((resize_max, resize_max))
    buf = BytesIO()
    img.save(buf, format=fmt)
    buf.seek(0)
    return "data:image/{};base64,{}".format(fmt.lower(), base64.b64encode(buf.read()).decode("ascii"))


def histogram_plot_to_data_url(hist):
    """Render histogram with matplotlib and return as data URL PNG."""
    fig, ax = plt.subplots(figsize=(6, 2.5), dpi=100)
    ax.bar(np.arange(256), hist, width=1.0)
    ax.set_xlim(0, 255)
    ax.set_xlabel("Gray level")
    ax.set_ylabel("Count")
    ax.set_title("Histogram (256 bins)")
    plt.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode("ascii")


def plot_histogram_with_thresholds(hist, results):
    """Plot histogram and overlay threshold lines for each algorithm in results.
    results: list of dicts with keys 'algo' and 'thresholds'
    Returns data-url PNG.
    """
    fig, ax = plt.subplots(figsize=(8, 4), dpi=100)
    ax.bar(np.arange(256), hist, width=1.0, color="#dfeaf6", edgecolor=None)
    ax.set_xlim(0, 255)
    ax.set_xlabel("Gray level")
    ax.set_ylabel("Count")
    ax.set_title("Histogram + Thresholds")

    # color cycle
    import itertools

    colors = plt.rcParams['axes.prop_cycle'].by_key().get('color', ['C0','C1','C2','C3','C4','C5'])
    color_cycle = itertools.cycle(colors)

    # group thresholds by algorithm name (so we draw one legend entry per algo)
    algo_map = {}
    for r in results:
        name = (r.get('algo') or 'algo')
        thr = r.get('thresholds', []) or []
        # ensure thresholds are list of ints
        try:
            thr_int = [int(x) for x in thr]
        except Exception:
            thr_int = []
        algo_map.setdefault(name, []).append(thr_int)

    # choose distinct colors from a qualitative colormap (tab10)
    cmap = plt.get_cmap('tab10')
    algo_names = sorted(algo_map.keys())
    color_map = {name: cmap(i % 10) for i, name in enumerate(algo_names)}

    # Draw all algos except mfwoa first, then draw mfwoa last to make it prominent
    draw_order = [n for n in algo_names if n.lower() != 'mfwoa'] + [n for n in algo_names if n.lower() == 'mfwoa']

    for name in draw_order:
        thr_lists = algo_map.get(name, [])
        col = color_map.get(name, None)
        # flatten lists of thresholds (some algos may have multiple runs) and unique-sort
        flat = sorted(set([int(x) for sub in thr_lists for x in (sub or [])]))
        if name.lower() == 'mfwoa':
            lw = 2.2
            alpha = 1.0
            z = 10
        else:
            lw = 1.2
            alpha = 0.9
            z = 5
        for t in flat:
            ax.axvline(t, color=col, linestyle='-', linewidth=lw, alpha=alpha, zorder=z)
        ax.plot([], [], color=col, label=name)

    ax.legend(loc='upper right', fontsize='small')
    plt.tight_layout()
    buf = BytesIO(); fig.savefig(buf, format='png'); plt.close(fig); buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode('ascii')


def save_dataurl_to_file(data_url: str, out_path: str):
    """Save a data:image/png;base64,... URL to a file path."""
    if not data_url.startswith("data:image"):
        raise ValueError("Unsupported data URL")
    header, b64 = data_url.split(",", 1)
    data = base64.b64decode(b64)
    with open(out_path, "wb") as f:
        f.write(data)


def save_segmentation_visual(seg_labels: np.ndarray, out_path: str):
    """Save segmentation labels (integers 0..k) as a grayscale PNG for download/visualization."""
    import PIL.Image as PILImage

    if seg_labels is None:
        raise ValueError("seg_labels is None")
    arr = np.array(seg_labels, dtype=np.int32)
    if arr.size == 0:
        raise ValueError("empty seg_labels")
    # Map classes 0..k to 0..255 for visualization
    k = int(arr.max())
    if k <= 0:
        vis = (arr == 0).astype(np.uint8) * 255
    else:
        vis = (arr.astype(np.float32) / max(1, k) * 255.0).astype(np.uint8)
    img = PILImage.fromarray(vis)
    img.save(out_path)


# ---------- MFWOA Optimizer ----------
# Now using MFWOA_MULTITASK for all optimizations (single-task and multitask)
_optimize_fn = None


# ---------- Fallback: simple WOA-like optimizer (discrete thresholds) ----------
def simple_woa_optimize(hist, k, membership="triangular", max_iter=200, pop_size=20):
    """
    Simple discrete WOA-like optimizer for demo purposes.

    Representation:
      - Candidate solution: sorted list of k unique integers in [1..254]
    Behavior:
      - Randomly initialize population, perturb towards the best, keep improvements.
    Returns:
      (best_thresholds_sorted, best_fe)
    Notes:
      - This is a lightweight heuristic, not the full MFWOA algorithm.
    """
    if k <= 0:
        raise ValueError("k must be positive")

    def rnd_solution():
        # sample k unique points in 1..254 and sort
        pts = sorted(random.sample(range(1, 255), k))
        return pts

    def evaluate(sol):
        try:
            fe = compute_fuzzy_entropy(hist, sol, membership=membership, for_minimization=False)
            return float(fe)
        except Exception:
            # very low score if evaluation fails
            return -1e9

    # Initialize population
    pop = [rnd_solution() for _ in range(pop_size)]
    scores = [evaluate(s) for s in pop]
    best_idx = int(max(range(len(pop)), key=lambda i: scores[i]))
    best = pop[best_idx]
    best_score = scores[best_idx]

    for _it in range(max_iter):
        for i in range(pop_size):
            if random.random() < 0.6:
                # perturb current towards best
                cand = []
                for a, b in zip(pop[i], best):
                    if random.random() < 0.7:
                        delta = int(round((b - a) * random.random()))
                        cand_val = min(254, max(1, a + delta))
                        cand.append(cand_val)
                    else:
                        cand.append(a)
                # occasional random replacement to preserve diversity
                if random.random() < 0.3:
                    idx = random.randrange(k)
                    cand[idx] = random.randrange(1, 255)
                # remove duplicates and fill if needed
                cand = sorted(list(dict.fromkeys(cand)))
                while len(cand) < k:
                    x = random.randrange(1, 255)
                    if x not in cand:
                        cand.append(x)
                cand = sorted(cand)[:k]
            else:
                cand = rnd_solution()

            sc = evaluate(cand)
            if sc > scores[i]:
                pop[i] = cand
                scores[i] = sc
            if sc > best_score:
                best_score = sc
                best = cand

    return best, best_score
# --- helper: apply thresholds -> segmentation mask (class indices 0..k) ---
def apply_thresholds_to_image(img_pil, thresholds):
    """
    Input:
      - img_pil: PIL Image (any mode) -> convert to grayscale
      - thresholds: sorted list of k thresholds integers in [0..255]
    Output:
      - seg: 2D numpy array of integer labels {0..k} (k+1 classes)
    Rule:
      - class 0: gray <= t0
      - class 1: t0 < gray <= t1
      - ...
      - class k: gray > t_{k-1}
    """
    img_gray = img_pil.convert('L')
    arr = np.array(img_gray, dtype=np.uint8)
    if len(thresholds) == 0:
        # trivial: single class
        return np.zeros_like(arr, dtype=np.int32)
    bins = [0] + thresholds + [255]
    # numpy.digitize to get bins 1..len(thresholds)+1, subtract 1 to get 0-based class ids
    labels = np.digitize(arr, thresholds, right=True)
    return labels.astype(np.int32)
# --- helper: compute PSNR, SSIM between original image and reconstructed segmented->reconstructed image ---
def compute_psnr_ssim(original_pil, seg_labels):
    """
    Compute PSNR and SSIM between original grayscale image and reconstructed image
    from segmentation labels.
    """
    if seg_labels is None:
        return None, None

    # Ảnh gốc xám
    orig_gray = original_pil.convert('L')
    arr = np.array(orig_gray, dtype=np.float32)

    # Đảm bảo seg_labels là mảng 2D int32, cùng kích thước với ảnh gốc
    seg = np.asarray(seg_labels)
    if seg.ndim > 2:
        seg = np.squeeze(seg)

    from PIL import Image as PILImage

    if seg.shape != arr.shape:
        try:
            seg_pil = PILImage.fromarray(seg.astype(np.int32))
            seg_pil = seg_pil.resize(orig_gray.size, resample=PILImage.NEAREST)
            seg = np.array(seg_pil, dtype=np.int32)
        except Exception:
            return None, None
    else:
        seg = seg.astype(np.int32)

    if seg.size == 0:
        return None, None

    max_cls = int(seg.max())
    if max_cls < 0:
        return None, None

    # Base reconstruction: mỗi lớp dùng median intensity trong ảnh gốc
    recon = np.zeros_like(arr, dtype=np.float32)
    for cls in range(max_cls + 1):
        mask = (seg == cls)
        if not mask.any():
            continue
        recon[mask] = np.median(arr[mask])

    # Pixel chưa gán (nếu có) gán xám trung bình
    unmapped = (recon == 0) & (seg >= 0)
    if unmapped.any():
        recon[unmapped] = 127.5

    # Edge-preserving refinement
    from scipy.ndimage import gaussian_filter

    edges = np.zeros_like(seg, dtype=bool)
    # biên dọc
    vertical_diff = np.abs(np.diff(seg, axis=0)) > 0
    edges[:-1, :] |= vertical_diff
    edges[1:, :] |= vertical_diff
    # biên ngang
    horizontal_diff = np.abs(np.diff(seg, axis=1)) > 0
    edges[:, :-1] |= horizontal_diff
    edges[:, 1:] |= horizontal_diff

    recon_smooth = gaussian_filter(recon, sigma=0.8)
    recon = np.where(edges, recon, recon_smooth)

    # PSNR và SSIM
    try:
        psnr = float(sk_ms_psnr(arr, recon, data_range=255))
    except Exception as e:
        print(f"PSNR error: {e}")
        psnr = None

    try:
        ssim = float(sk_ms_ssim(arr.astype(np.uint8),
                                recon.astype(np.uint8),
                                data_range=255))
    except Exception as e:
        print(f"SSIM error: {e}")
        ssim = None

    return psnr, ssim
def compute_fsim_original(original_pil, seg_labels):
    """Compute FSIM giữa ảnh gốc xám và ảnh tái tạo từ nhãn phân đoạn."""
    from src.metrics.metrics import fsim
    import numpy as np
    from PIL import Image as PILImage

    if seg_labels is None:
        return None

    # Ảnh gốc xám
    orig_gray = original_pil.convert('L')
    arr = np.array(orig_gray, dtype=np.float64)

    # Nhãn phân đoạn
    seg = np.asarray(seg_labels, dtype=np.int32)
    if seg.ndim > 2:
        seg = np.squeeze(seg)

    if seg.shape != arr.shape:
        try:
            seg_pil = PILImage.fromarray(seg.astype(np.int32))
            seg_pil = seg_pil.resize(orig_gray.size, resample=PILImage.NEAREST)
            seg = np.array(seg_pil, dtype=np.int32)
        except Exception:
            return None

    max_cls = int(seg.max())
    if max_cls < 0:
        return None

    # Tái tạo ảnh từ nhãn: mỗi lớp lấy median intensity
    recon = np.zeros_like(arr, dtype=np.float64)
    for cls in range(max_cls + 1):
        mask = (seg == cls)
        if not mask.any():
            continue
        recon[mask] = float(np.median(arr[mask]))

    # Pixel chưa gán (nếu có)
    unmapped = (recon == 0) & (seg >= 0)
    if unmapped.any():
        recon[unmapped] = 127.5

    try:
        score = fsim(arr, recon)
        return float(score)
    except Exception:
        return None


# --- helper: compute FSIM between segmentation and ground-truth mask ---
def compute_fsim(seg_labels, gt_labels):
    """
    Compute FSIM (Feature Similarity Index Measure) between segmentation and ground-truth.
    
    FSIM measures structural similarity by comparing:
    - Phase Congruency (local phase alignment)
    - Gradient Magnitude (edge strength)
    
    Args:
        seg_labels: Segmentation result (class indices, H x W)
        gt_labels: Ground-truth labels (class indices, H x W)
    
    Returns: 
        fsim_score (float) in [0, 1], or None if shapes mismatch
    """
    if seg_labels is None or gt_labels is None:
        return None
    if seg_labels.shape != gt_labels.shape:
        return None
    
    from src.metrics.metrics import fsim
    
    # Convert to float64 for FSIM computation
    seg_float = seg_labels.astype(np.float64)
    gt_float = gt_labels.astype(np.float64)
    
    try:
        fsim_score = fsim(gt_float, seg_float)
        return float(fsim_score)
    except Exception as e:
        print(f"[FSIM Error] {e}")
        return None


def compute_spacing_penalty(thresholds, min_spacing=10):
    """
    Penalize thresholds that are too close together (clustering).
    
    Args:
        thresholds: list of threshold values (sorted)
        min_spacing: minimum desired spacing between consecutive thresholds
    
    Returns:
        penalty: 0 if all spacing >= min_spacing, otherwise penalty based on violations
    """
    if len(thresholds) < 2:
        return 0.0
    
    thresholds = sorted(thresholds)
    spacings = np.diff(thresholds)
    
    # Penalty: sum of squared violations for spacings below min_spacing
    violations = np.maximum(0, min_spacing - spacings)
    spacing_penalty = float(np.sum(violations ** 2)) / len(violations)
    
    return spacing_penalty


def _run_single_algo_for_K(pil_image, hist, algo, K, opt_iters, pop_size, membership):
    """
    Run a single algorithm (algo) with a single K value, return result dict.
    
    All algorithms use same parameters for fair, scientific comparison.
    
    Returns:
        { 'algo': str, 'thresholds': [...], 'fe': float | None, 'time': float, 'seg_labels': np.array | None }
    """
    # Fair comparison: all algorithms use same parameters
    used_pop_size = pop_size
    used_iters = opt_iters
    import inspect
    import time as _time
    
    start_t = _time.perf_counter()
    thresholds = []
    fe_val = None
    
    try:
        # ===== OTSU =====
        if algo == 'otsu':
            try:
                from src.seg.otsu import otsu_thresholds
                
                if K >= 4:
                    print(f"  [OTSU] K={K}: Using fast quantile approximation (O(N log N))")
                else:
                    print(f"  [OTSU] K={K}: Using true Otsu method")
                
                thresholds = otsu_thresholds(pil_image, K)
                
                # Apply SAME penalty as optimizers for fair comparison
                fe_val = float(compute_fuzzy_entropy(hist, thresholds, membership=membership, for_minimization=False,
                                                     lambda_penalty=1.0, alpha_area=0.50, beta_membership=0.80, gamma_spacing=0.90))
            except Exception as e:
                print(f"Otsu error (K={K}): {e}")
                thresholds = []
                fe_val = None
        elif algo == 'mfwoa':
            # Skip single-task MFWOA - only run in multitask mode
            print(f"  [MFWOA] K={K}: Skipping single-task, will be handled by multitask mode")
            thresholds = []
            fe_val = None
        # # ===== MFWOA (single-task mode) =====
        # # Uses MFWOA_MULTITASK with single task (rmp_init=0.0 disables knowledge transfer)
        # elif algo == 'mfwoa':
        #     try:
        #         from src.optim.mfwoa_multitask import mfwoa_multitask
        #         # Run as single-task (T=1) with knowledge transfer disabled
        #         print(f"  [MFWOA] K={K}: Running MFWOA_MULTITASK in single-task mode (rmp_init=0.0)")
        #         thresholds_st, scores_st, _ = mfwoa_multitask(
        #             hists=[hist],
        #             Ks=[K],
        #             pop_size=used_pop_size,
        #             iters=used_iters,
        #             membership=membership,
        #             lambda_penalty=1.0,
        #             alpha_area=0.50,
        #             beta_membership=0.80,
        #             gamma_spacing=0.90,
        #             rmp_init=0.0,          # Single task: no cross-task transfer
        #             elitism=None,          # ✅ NEW: Adaptive elitism
        #             enable_mutation=True,  
        #             mutation_rate=0.05,    
        #         )
        #         thresholds = thresholds_st[0] if thresholds_st else []
        #         fe_val = float(compute_fuzzy_entropy(hist, thresholds, membership=membership, for_minimization=False,
        #                                              lambda_penalty=1.0, alpha_area=0.50, beta_membership=0.80, gamma_spacing=0.90))
        #     except Exception as e:
        #         print(f"MFWOA error (K={K}): {e}")
        #         thresholds = []
        #         fe_val = None

        # ===== WOA =====
        elif algo == 'woa':
            try:
                from src.optim.woa import woa_optimize
                def objective_woa(thr_or_pos):
                    try:
                        arr = np.asarray(thr_or_pos)
                        from src.seg.utils import enforce_threshold_constraints
                        try:
                            th_int = np.round(arr).astype(np.int32)
                        except Exception:
                            th_int = np.array([int(x) for x in arr], dtype=np.int32)
                        th_arr = enforce_threshold_constraints(th_int)
                        thr = [int(x) for x in th_arr]
                        fe_val = float(compute_fuzzy_entropy(hist, thr, membership=membership, for_minimization=False,
                                                             lambda_penalty=1.0, alpha_area=0.50, beta_membership=0.80, gamma_spacing=0.90))
                        # WOA minimizes, so negate FE to maximize it
                        return -float(fe_val)
                    except Exception as exc:
                        print(f"WOA objective error (K={K}): {exc}")
                        return 1e8  # Large positive error value for minimizer
                
                res = woa_optimize(hist, K, pop_size=used_pop_size, iters=used_iters, objective=objective_woa)
                if isinstance(res, tuple) and len(res) >= 2:
                    thresholds = [int(x) for x in res[0]]
                else:
                    thresholds = [int(x) for x in res] if res else []
                # Compute FE once on final thresholds only
                if thresholds:
                    fe_val = float(compute_fuzzy_entropy(hist, thresholds, membership=membership, for_minimization=False,
                                                         lambda_penalty=1.0, alpha_area=0.50, beta_membership=0.80, gamma_spacing=0.90))
            except Exception as e:
                print(f"WOA error (K={K}): {e}")
                thresholds = []
                fe_val = None

        # ===== PSO =====
        elif algo == 'pso':
            try:
                from src.optim.pso import pso_optimize
                def objective_pso(thr_or_pos):
                    try:
                        arr = np.asarray(thr_or_pos)
                        from src.seg.utils import enforce_threshold_constraints
                        try:
                            th_int = np.round(arr).astype(np.int32)
                        except Exception:
                            th_int = np.array([int(x) for x in arr], dtype=np.int32)
                        th_arr = enforce_threshold_constraints(th_int)
                        thr = [int(x) for x in th_arr]
                        # PSO minimizes, so negate FE (we want to maximize FE)
                        fe_val = -float(compute_fuzzy_entropy(hist, thr, membership=membership, for_minimization=False,
                                                             lambda_penalty=1.0, alpha_area=0.50, beta_membership=0.80, gamma_spacing=0.90))
                        return fe_val
                    except Exception as exc:
                        print(f"PSO objective error (K={K}): {exc}")
                        return 1e8  # Large positive error value for minimizer
                
                res = pso_optimize(hist, K, pop_size=used_pop_size, iters=used_iters, objective=objective_pso)
                if isinstance(res, tuple) and len(res) >= 2:
                    thresholds = [int(x) for x in res[0]] if res[0] else []
                    opt_score = res[1]
                else:
                    thresholds = [int(x) for x in res] if res else []
                    opt_score = None
                
                # Compute FE once on final thresholds only
                if thresholds:
                    fe_val = float(compute_fuzzy_entropy(hist, thresholds, membership=membership, for_minimization=False,
                                                         lambda_penalty=1.0, alpha_area=0.50, beta_membership=0.80, gamma_spacing=0.90))
                else:
                    fe_val = None
            except Exception as e:
                print(f"PSO error (K={K}): {e}")
                import traceback
                traceback.print_exc()
                thresholds = []
                fe_val = None

        else:
            # unknown algo
            thresholds = []
            fe_val = None

    except Exception as e:
        print(f"Algorithm {algo} failed (K={K}): {e}")
        thresholds = []
        fe_val = None

    end_t = _time.perf_counter()
    elapsed = end_t - start_t

    # Create segmentation
    try:
        from src.seg.utils import enforce_threshold_constraints
        # ⚠️ NOTE: MFWOA, WOA, PSO already enforce constraints in their objective function
        # Only need to enforce constraints if thresholds come from non-optimized source (e.g., Otsu)
        # For optimizers, they return already-constrained thresholds
        if thresholds and algo != 'mfwoa' and algo != 'woa' and algo != 'pso':
            # Only enforce for Otsu (non-optimizer)
            try:
                th_arr = np.array([int(x) for x in thresholds])
                th_arr = enforce_threshold_constraints(th_arr)
                thresholds = [int(x) for x in th_arr]
                # Recompute FE on enforced thresholds
                try:
                    fe_val = float(compute_fuzzy_entropy(hist, thresholds, membership=membership, for_minimization=False,
                                                         lambda_penalty=1.0, alpha_area=0.50, beta_membership=0.80, gamma_spacing=0.90))
                except Exception:
                    pass
            except Exception:
                pass
    except Exception:
        pass

    try:
        seg_labels = apply_thresholds_to_image(pil_image, thresholds) if thresholds else np.zeros(np.array(pil_image.convert('L')).shape, dtype=np.int32)
    except Exception:
        seg_labels = None

    return {
        'algo': algo,
        'thresholds': [int(x) for x in thresholds] if thresholds else [],
        'fe': float(fe_val) if fe_val is not None else None,
        'time': elapsed,
        'opt_iters': opt_iters,
        'K': K,
        'seg_labels': seg_labels
    }


def run_algorithms_and_benchmark(pil_image, hist, membership, selected_algos, opt_iters, pop_size):
    """
    Chạy tất cả các thuật toán được chọn cho một phạm vi giá trị K được xác định trước [2, 3, 4, 5]
    để tìm mức ngưỡng tốt nhất.

    Returns:
      - results: list of dict { 'algo': str, 'thresholds': [...], 'fe': float, 'time': sec, 'seg_labels': np.array }
    """
    results = []
    K_to_test = [2, 3, 4, 5, 6, 7, 8, 9, 10]  # Phạm vi K để tìm kiếm
    
    # # CONFIG: Change K range here
    # # K_MIN = 2, K_MAX = 254 (for L=256)
    # K_MIN = 2
    # K_MAX = 254
    # N_SAMPLES = 20  # Number of K values to test (evenly spaced)
    
    # # Generate K values: evenly spaced from K_MIN to K_MAX
    # import numpy as np
    # K_to_test = list(np.linspace(K_MIN, K_MAX, N_SAMPLES, dtype=int))
    # K_to_test = sorted(list(set(K_to_test)))  # Remove duplicates and sort

    UI_POP = int(max(4, int(pop_size)))
    UI_ITERS = int(opt_iters)
    
    print(f"[BENCHMARK] pop_size={pop_size}, opt_iters={opt_iters}, UI_POP={UI_POP}, UI_ITERS={UI_ITERS}")
    
    # ===== CALCULATE ADAPTIVE ITERATIONS FOR ALL K =====
    # Use same adaptive strategy as single-task algorithms for fair comparison
    ENABLE_ADAPTIVE_ITERS_BENCHMARK = False  # Set to False to use fixed iterations for all K
    
    adaptive_iters = {}
    for k_val in K_to_test:
        current_iters = UI_ITERS
        if ENABLE_ADAPTIVE_ITERS_BENCHMARK:
           
            if k_val >= 10:
                current_iters = max(100, int(UI_ITERS * 0.60))  # 60% (was 25%)
            elif k_val >= 8:
                current_iters = max(120, int(UI_ITERS * 0.70))  # 70% (was 25%)
            elif k_val >= 6:
                current_iters = max(140, int(UI_ITERS * 0.80))  # 80% (was 40%)
            elif k_val >= 5:
                current_iters = max(160, int(UI_ITERS * 0.90))  # 90% (was 60%)
            # K≤4: 100%
        adaptive_iters[k_val] = current_iters

    print(f"[START] Bat dau benchmark cho K trong {K_to_test} tren {len(selected_algos)} thuat toan...")
    print(f"  Adaptive iterations: {adaptive_iters}")


    # ===== MFWOA MULTITASK (nếu có) =====
    # Chạy MFWOA một lần với tất cả K để chuyển giao kiến thức
    # mfwoa_multitask_results = {}
    # if 'mfwoa' in selected_algos:
    #     try:
    #         print(f"\n  🔗 Chạy MFWOA Multitask cho tất cả K = {K_to_test}...")
    #         from src.optim.mfwoa_multitask import mfwoa_multitask
    #         import time as _time
    #         start_mt = _time.perf_counter()
            
    #         # Prepare hists (same cho mỗi K vì cùng 1 ảnh)
    #         hists_list = [hist] * len(K_to_test)
            
    #         # Dùng SỐ VÒNG LẶP TRUNG BÌNH của các K (fair cho multitask)
    #         avg_iters = int(np.mean(list(adaptive_iters.values())))
    #         print(f"    Using average adaptive iterations: {avg_iters} (from {adaptive_iters})")
            
    #         # Chạy MFWOA multitask (mỗi task ứng với 1 K)
    #         thresholds_list, scores, diag = mfwoa_multitask(
    #             hists=hists_list,
    #             Ks=K_to_test,
    #             pop_per_task=UI_POP,   # cùng pop với PSO/WOA
    #             iters=avg_iters,        # dùng số vòng lặp trung bình
    #             membership=membership,
    #             lambda_penalty=1.0,
    #             alpha_area=0.50,
    #             beta_membership=0.80,
    #             gamma_spacing=0.90,
    #             rmp_init=0.3,          # cho phép chuyển giao tri thức giữa các K
    #             elitism=int(0.05 * UI_POP),
    #             enable_mutation=True,
    #             mutation_rate=0.05,
    #         )
                        
    #         print(f"    [DEBUG MT RETURN] len(thresholds_list)={len(thresholds_list)}, len(scores)={len(scores)}")
            
    #         time_mt = _time.perf_counter() - start_mt
            
    #         # Lưu lại kết quả theo từng K
    #         for k_idx, k_val in enumerate(K_to_test):
    #             if k_idx >= len(thresholds_list):
    #                 continue

    #             thr = thresholds_list[k_idx]
                
    #             # Tính lại FE với cùng cấu hình penalty như các thuật toán khác
    #             if thr:
    #                 fe_score = float(compute_fuzzy_entropy(
    #                     hist, thr,
    #                     membership=membership,
    #                     for_minimization=False,
    #                     lambda_penalty=1.0,
    #                     alpha_area=0.50,
    #                     beta_membership=0.80,
    #                     gamma_spacing=0.90,
    #                 ))
    #             else:
    #                 fe_score = None

    #             # In chẩn đoán nếu diag là dict
    #             if isinstance(diag, dict) and k_idx == 0:
    #                 nfe = diag.get('nfe', 0)
    #                 cross_task = diag.get('cross_task_count', 0)
    #                 mutations = diag.get('mutation_count', 0)
    #                 print(f"        [DIAG] NFE={nfe}, cross_task={cross_task}, mutations={mutations}")
                
    #             mfwoa_multitask_results[k_val] = {
    #                 'thresholds': thr,
    #                 'fe': fe_score,
    #                 'time': time_mt / len(K_to_test),  # chia đều thời gian cho từng K
    #                 'iters': avg_iters                 # lưu số vòng lặp đã dùng
    #             }
    #             fe_log = f"{fe_score:.6f}" if fe_score is not None else "N/A"
    #             print(f"      [DEBUG] mfwoa_K{k_val}: thresholds={thr}, FE_final={fe_log}")
            
    #         print(f"    [OK] MFWOA Multitask hoàn tất (tổng time={time_mt:.2f}s)")
    #     except Exception as e:
    #         print(f"    [WARN] MFWOA Multitask không khả dụng: {e}, fallback to single-task")
    # ===== MFWOA MULTITASK (nếu có) =====
    # ✅ MFWOA LUÔN chạy multitask mode cho TẤT CẢ K
    mfwoa_multitask_results = {}
    if 'mfwoa' in selected_algos:
        try:
            print(f"\n  🔗 [MFWOA MULTITASK] Chạy cho tất cả K = {K_to_test}...")
            from src.optim.mfwoa_multitask import mfwoa_multitask
            import time as _time
            start_mt = _time.perf_counter()
            
            # Prepare hists (same for all K since same image)
            hists_list = [hist] * len(K_to_test)
            
            # ✅ FIX: Dùng MEDIAN iterations (thay vì mean) để cân bằng K nhỏ và lớn
            # Median giúp tránh bias từ K cực lớn (K=10 với 60% iters)
            iters_values = list(adaptive_iters.values())
            median_iters = int(np.median(iters_values))
            # if len(K_to_test) >= 5:
            #     # Multitask với nhiều tasks cần thêm iterations
            #     median_iters = max(median_iters, 200)  # Minimum 200 iters
            #     print(f"    Boosted iterations for multitask: {median_iters}")

            print(f"    Using MEDIAN adaptive iterations: {median_iters} (from {adaptive_iters})")
            
            # ✅ CRITICAL FIX: Tăng pop_per_task để đủ diversity cho multitask
            # Paper recommendation: pop_size = 100-200 per task for good knowledge transfer
            pop_per_task_mt = max(4, UI_POP * max(K_to_test))  # at least double the single-task pop
            print(f"    Population: {pop_per_task_mt} per task × {len(K_to_test)} tasks = {pop_per_task_mt * len(K_to_test)} total")
            
            # Chạy MFWOA multitask (mỗi task ứng với 1 K)
            thresholds_list, scores, diag = mfwoa_multitask(
                hists=hists_list,
                Ks=K_to_test,
                pop_per_task=pop_per_task_mt,  # ✅ Tăng population
                iters=UI_ITERS,             # ✅ Dùng median thay vì mean
                membership=membership,
                lambda_penalty=1.0,
                alpha_area=0.50,
                beta_membership=0.80,
                gamma_spacing=0.90,
                rmp_init=0.3,          # ✅ Enable knowledge transfer
                elitism=None,          # ✅ Auto-compute: 5% of pop_size
                enable_mutation=True,
                mutation_rate=0.1,     # ✅ FIX: Paper uses 0.1, not 0.05
                verbose=True,          # ✅ Enable logging
            )
            
            print(f"    [DEBUG MT RETURN] len(thresholds_list)={len(thresholds_list)}, len(scores)={len(scores)}")
            
            time_mt = _time.perf_counter() - start_mt
            
            # Lưu lại kết quả theo từng K
            for k_idx, k_val in enumerate(K_to_test):
                if k_idx >= len(thresholds_list):
                    print(f"    [WARN] Missing result for K={k_val} (idx={k_idx})")
                    continue

                thr = thresholds_list[k_idx]
                
                # Tính lại FE với cùng cấu hình penalty như các thuật toán khác
                if thr:
                    fe_score = float(compute_fuzzy_entropy(
                        hist, thr,
                        membership=membership,
                        for_minimization=False,
                        lambda_penalty=1.0,
                        alpha_area=0.50,
                        beta_membership=0.80,
                        gamma_spacing=0.90,
                    ))
                else:
                    fe_score = None

                # In chẩn đoán nếu diag là dict
                if isinstance(diag, dict) and k_idx == 0:
                    nfe = diag.get('nfe', 0)
                    cross_task = diag.get('cross_task_count', 0)
                    mutations = diag.get('mutation_count', 0)
                    print(f"        [DIAG] NFE={nfe}, cross_task={cross_task}, mutations={mutations}")
                
                mfwoa_multitask_results[k_val] = {
                    'thresholds': thr,
                    'fe': fe_score,
                    'time': time_mt / len(K_to_test),  # chia đều thời gian cho từng K
                    'iters': median_iters              # lưu số vòng lặp đã dùng
                }
                fe_log = f"{fe_score:.6f}" if fe_score is not None else "N/A"
                print(f"      [K={k_val}] MFWOA: thresholds={thr}, FE={fe_log}")
            
            print(f"    [OK] MFWOA Multitask hoàn tất (total time={time_mt:.2f}s, avg={time_mt/len(K_to_test):.2f}s per K)")
        
        except Exception as e:
            print(f"    [ERROR] MFWOA Multitask failed: {e}")
            import traceback
            traceback.print_exc()
            # ❌ Không fallback sang single-task nữa
            print(f"    [CRITICAL] MFWOA requires multitask mode - skipping all K values")
            mfwoa_multitask_results = {}
    # VÒNG LẶP BENCHMARK CHO TỪNG K
    for k_val in K_to_test:
        print(f"\n  Đang chạy benchmark cho K={k_val}...")
        # dùng số vòng lặp adaptive đã tính trước
        current_iters = adaptive_iters[k_val]

        for algo in selected_algos:
            print(f"    -> Đang chạy {algo}...")
            
            # ƯU TIÊN DÙNG KẾT QUẢ MFWOA MULTITASK NẾU CÓ
            if algo == 'mfwoa' and k_val in mfwoa_multitask_results:
                mt_res = mfwoa_multitask_results[k_val]
                thr_list = mt_res['thresholds']
                print(f"    [DEBUG MULTITASK] K={k_val}, thresholds={thr_list}, type={type(thr_list)}")
                thr_list = thr_list if isinstance(thr_list, list) else list(thr_list)

                res = {
                    'algo': f"mfwoa_K{k_val}",
                    'thresholds': thr_list,
                    'fe': mt_res['fe'],
                    'time': mt_res['time'],
                    'opt_iters': UI_ITERS,   # vẫn báo cáo theo adaptive_iters của K
                    'K': k_val,
                    'seg_labels': apply_thresholds_to_image(pil_image, thr_list) if thr_list else None
                }
                results.append(res)
                fe_str = f"{res['fe']:.4f}" if res['fe'] is not None else "N/A"
                print(f"    [OK] mfwoa_K{k_val}: FE={fe_str}, thresholds={res['thresholds']}, time={res['time']:.2f}s (multitask)")
                continue
            
            # FALLBACK: CHẠY SINGLE-TASK BÌNH THƯỜNG
            res = _run_single_algo_for_K(
                pil_image=pil_image,
                hist=hist,
                algo=algo,
                K=k_val,
                opt_iters=UI_ITERS,
                pop_size=UI_POP,
                membership=membership
            )
            # Thêm K vào tên thuật toán
            res['algo'] = f"{algo}_K{k_val}"
            results.append(res)
            fe_val_str = f"{res['fe']:.4f}" if res['fe'] is not None else "N/A"
            print(f"    [OK] {res['algo']}: FE={fe_val_str}, time={res['time']:.2f}s [iters={current_iters}, pop={UI_POP}]")

    print(f"\n[DONE] Benchmark hoan tat. Tong so ket qua: {len(results)}")
    return results


# ---------- Routes ----------
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/compute", methods=["POST"])
def compute():
    if "image" not in request.files:
        return "No image uploaded", 400
    f = request.files["image"]
    try:
        pil = Image.open(f.stream)
        # Convert color images to grayscale
        if pil.mode != 'L':
            pil = pil.convert('L')
    except Exception as e:
        return f"Failed to open image: {e}", 400

    # parse
    n_thresholds_raw = request.form.get("n_thresholds", "5").strip()
    try:
        n_thresholds = int(n_thresholds_raw) if n_thresholds_raw else 5
    except ValueError:
        n_thresholds = 5
    auto_opt = request.form.get("auto_optimize") is not None
    benchmark = request.form.get("benchmark") is not None
    membership = request.form.get("membership", "triangular")
    try:
        opt_iters = int(request.form.get("opt_iters", "200"))
    except ValueError:
        opt_iters = 200

    # population size from UI
    try:
        pop_size = int(request.form.get("pop_size", "15"))
    except ValueError:
        pop_size = 15

    UI_POP_CAP = 500
    if pop_size < 4:
        pop_size = 4
    if pop_size > UI_POP_CAP:
        pop_size = UI_POP_CAP

    print(f"[DEBUG] Form inputs: opt_iters={opt_iters}, pop_size={pop_size}")

    # If running from the web UI (benchmark mode), cap iterations to keep jobs reasonably fast
    # (user can run heavier experiments from CLI). This prevents the server from running very long
    # jobs by default. Cap value is conservative; adjust as needed.
    UI_OPT_ITERS_CAP = 500  # Increased from 120 to allow better convergence

    # ===== FIXED vs ADAPTIVE ITERATIONS =====
    # ENABLE_ADAPTIVE_ITERS = False: Disable adaptive, use fixed iterations for all K
    # ENABLE_ADAPTIVE_ITERS = True: Enable adaptive iterations (scale down with higher K)

    # Parse optimization mode (single vs multitask)
    optimization_mode = request.form.get("optimization_mode", "single").strip().lower()
    if optimization_mode not in ['single', 'multitask']:
        optimization_mode = 'single'

    # which algos selected (list)
    selected_algos = request.form.getlist("algos")
    if not selected_algos:
        # default
        selected_algos = ['mfwoa','woa','otsu']

    # ground-truth optional
    gt_file = request.files.get("gt", None)
    gt_labels = None
    if gt_file and gt_file.filename:
        try:
            gt_pil = Image.open(gt_file.stream).convert('L')
            # Resize GT to match input image size if needed (nearest neighbor to preserve labels)
            try:
                if gt_pil.size != pil.size:
                    gt_pil = gt_pil.resize(pil.size, resample=Image.NEAREST)
            except Exception:
                # fallback: ignore resizing error
                pass
            gt_arr = np.array(gt_pil)
            # Normalize GT to small label set: if binary, set 0/1 or 0/255
            # If values are 0/255, keep as 0/255; else keep numeric labels
            gt_labels = gt_arr
        except Exception:
            gt_labels = None

    # compute histogram
    hist = image_to_histogram(pil)

    # if benchmark requested -> run multiple algos; else run selected single mode
    if benchmark:
        # cap heavy optimizer iterations for web requests
        if opt_iters > UI_OPT_ITERS_CAP:
            opt_iters = UI_OPT_ITERS_CAP
        
        # NOTE: Adaptive iterations scaling has been REMOVED from here.
        # Each K value in K_to_test will now use the exact UI_ITERS value.
        # Adaptive scaling is handled internally in run_algorithms_and_benchmark() 
        # via ENABLE_ADAPTIVE_ITERS_BENCHMARK flag if needed.
        
        # run benchmark asynchronously in background thread to avoid blocking the request
        job_id = uuid.uuid4().hex
        job_subdir = f"exports/jobs/{job_id}"
        job_dir = os.path.join(app.static_folder, job_subdir)
        try:
            os.makedirs(job_dir, exist_ok=True)
        except Exception:
            pass

        def _worker_save():
            status = {"status": "running", "started": int(_time.time())}
            try:
                with open(os.path.join(job_dir, "status.json"), "w", encoding="utf-8") as sf:
                    json.dump(status, sf)

                results = run_algorithms_and_benchmark(pil, hist, membership, selected_algos, opt_iters, pop_size)

                # compute PSNR/SSIM/DICE where GT available
                rows = []
                # ⚠️ Convert histogram to list for JSON serialization
                hist_list = hist.tolist() if isinstance(hist, np.ndarray) else list(hist)
                
                for r in results:
                    seg = r['seg_labels']
                    if seg is None:
                        psnr = ssim = fsim_score = None
                    else:
                        psnr, ssim = compute_psnr_ssim(pil, seg)
                        if gt_labels is None:
                            # Không có GT: dùng FSIM giữa ảnh gốc và ảnh tái tạo từ segmentation
                            fsim_score = compute_fsim_original(pil, seg)
                        else:
                            # Ensure GT and segmentation have same size
                            if seg.shape != gt_labels.shape:
                                try:
                                    from PIL import Image as PILImage
                                    gt_pil = PILImage.fromarray(gt_labels.astype(np.uint8))
                                    gt_pil = gt_pil.resize(pil.size, resample=PILImage.NEAREST)
                                    gt_resized = np.array(gt_pil, dtype=np.uint8)
                                    fsim_score = compute_fsim(seg, gt_resized)
                                except Exception:
                                    fsim_score = None
                            else:
                                fsim_score = compute_fsim(seg, gt_labels)
                    rows.append({
                        'algo': r['algo'],
                        'thresholds': ','.join(map(str, r['thresholds'])) if r['thresholds'] else '',
                        'fe': r['fe'] if r['fe'] is not None else None,
                        'time': round(r['time'], 4),
                        'opt_iters': r.get('opt_iters', 0),
                        'K': r.get('K', 0),
                        'psnr': psnr,
                        'ssim': ssim,
                        'fsim': fsim_score,
                        'hist_data': hist_list,  # ⚠️ Add histogram for interactive chart
                    })
                    # Debug: print threshold info
                    if not r['thresholds']:
                        print(f"  [WARNING] {r['algo']}: Empty thresholds! r={r}")

                # save CSV and JSON
                df = pd.DataFrame(rows)
                csv_path = os.path.join(job_dir, "results.csv")
                df.to_csv(csv_path, index=False)

                json_path = os.path.join(job_dir, "results.json")
                with open(json_path, "w", encoding="utf-8") as jf:
                    json.dump({"rows": rows, "timestamp": int(_time.time())}, jf, default=lambda o: None)

                # save histogram + thresholds image
                hist_img_data = plot_histogram_with_thresholds(hist, results)
                try:
                    save_dataurl_to_file(hist_img_data, os.path.join(job_dir, "hist_thresholds.png"))
                except Exception as e:
                    print(f"Failed saving hist image in job: {e}")

                # save per-algo segmentation visuals
                export_entries = []
                for r in results:
                    algo = r.get('algo')
                    seg = r.get('seg_labels')
                    seg_fname = None
                    if seg is not None:
                        try:
                            safe_algo = str(algo).replace(' ', '_')
                            seg_fname = f"seg_{safe_algo}.png"
                            save_segmentation_visual(seg, os.path.join(job_dir, seg_fname))
                        except Exception as e:
                            print(f"Failed saving seg for job {job_id} algo {algo}: {e}")
                    export_entries.append({'algo': algo, 'seg_fname': seg_fname})

                status = {"status": "done", "rows": rows, "exports": export_entries}
                with open(os.path.join(job_dir, "status.json"), "w", encoding="utf-8") as sf:
                    json.dump(status, sf)
            except Exception as exc:
                status = {"status": "error", "error": str(exc)}
                with open(os.path.join(job_dir, "status.json"), "w", encoding="utf-8") as sf:
                    json.dump(status, sf)

        t = threading.Thread(target=_worker_save, daemon=True)
        t.start()

        return render_template('processing.html', job_id=job_id)
    else:
        # previous single-algo behavior (use selected first algo or manual)
        # re-use existing logic: if auto_opt -> run optimizer for first selected algo else use manual thresholds
        selected_algo = selected_algos[0] if selected_algos else 'mfwoa'
        # re-use run_algorithms_and_benchmark for single algo
        results = run_algorithms_and_benchmark(pil, hist, membership, [selected_algo], opt_iters, pop_size)
        r = results[0]
        thresholds = r['thresholds']
        fe = r['fe']
        img_preview = pil_to_data_url(pil, fmt='PNG', resize_max=512)
        hist_img = histogram_plot_to_data_url(hist)
        info = f"mode: single ({selected_algo})"
        return render_template('result.html',
                               fe=fe,
                               thresholds=thresholds,
                               membership=membership,
                               img_preview=img_preview,
                               hist_img=hist_img,
                               info=info)
    


@app.route('/job_status/<job_id>', methods=['GET'])
def job_status(job_id):
    job_subdir = f"exports/jobs/{job_id}"
    job_dir = os.path.join(app.static_folder, job_subdir)
    status_path = os.path.join(job_dir, 'status.json')
    if not os.path.exists(status_path):
        return jsonify({'status': 'pending'})
    try:
        with open(status_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return jsonify(data)
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)})


@app.route('/job_result/<job_id>', methods=['GET'])
def job_result(job_id):
    job_subdir = f"exports/jobs/{job_id}"
    job_dir = os.path.join(app.static_folder, job_subdir)
    json_path = os.path.join(job_dir, 'results.json')
    csv_path = os.path.join(job_dir, 'results.csv')
    hist_url = None
    if os.path.exists(os.path.join(job_dir, 'hist_thresholds.png')):
        hist_url = url_for('static', filename=f"{job_subdir}/hist_thresholds.png")
    rows = []
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as jf:
                data = json.load(jf)
                rows = data.get('rows', [])
        except Exception:
            rows = []

    csv_data_url = None
    if os.path.exists(csv_path):
        csv_data_url = url_for('static', filename=f"{job_subdir}/results.csv")

    # build export_entries list from saved seg files
    export_entries = []
    for fname in os.listdir(job_dir) if os.path.exists(job_dir) else []:
        if fname.startswith('seg_') and fname.lower().endswith('.png'):
            algo = fname[len('seg_'):-4]
            export_entries.append({'algo': algo, 'seg_url': url_for('static', filename=f"{job_subdir}/{fname}"), 'seg_fname': fname})

    # SAVE ALL RESULTS BEFORE FILTERING FOR COMPARISON
    all_rows_original = [r.copy() for r in rows] if rows else []
    all_entries_original = [e.copy() for e in export_entries]

    # Find BEST result for EACH algorithm (based on FE - optimization target)
    best_per_algo = {}  # algo_base -> best_row
    
    if rows:
        # Get valid rows with FE metric
        valid_rows = [r for r in rows if r.get('fe') is not None]
        
        if valid_rows:
            # Group by algorithm base name
            algo_groups = {}
            for row in valid_rows:
                algo_base = row.get('algo', '').split('_K')[0]
                if algo_base not in algo_groups:
                    algo_groups[algo_base] = []
                algo_groups[algo_base].append(row)
            
            # For each algorithm, find the one with highest FE
            for algo_base, algo_rows in algo_groups.items():
                best_row = max(algo_rows, key=lambda r: r.get('fe', 0))
                best_per_algo[algo_base] = best_row
        
        # Keep only the best result per algorithm
        if best_per_algo:
            rows = [best_per_algo[algo] for algo in sorted(best_per_algo.keys())]
            # Filter export_entries to only include best algos
            best_algos = set([r.get('algo') for r in rows])
            export_entries = [e for e in export_entries if e.get('algo') in best_algos]
    
    # Set best_by_* to the best for each metric independently (from filtered rows)
    best_by_fe = None
    best_by_psnr = None
    best_by_ssim = None
    
    if rows:
        # Find best by FE (highest)
        fe_rows = [r for r in rows if r.get('fe') is not None]
        if fe_rows:
            best_by_fe = max(fe_rows, key=lambda r: r['fe'])
        
        # Find best by PSNR (highest)
        psnr_rows = [r for r in rows if r.get('psnr') is not None]
        if psnr_rows:
            best_by_psnr = max(psnr_rows, key=lambda r: r['psnr'])
        
        # Find best by SSIM (highest)
        ssim_rows = [r for r in rows if r.get('ssim') is not None]
        if ssim_rows:
            best_by_ssim = max(ssim_rows, key=lambda r: r['ssim'])

    # Create small plot placeholders
    fe_plot = time_plot = psnr_plot = ssim_plot = dice_plot = None

    # ⚠️ Prepare chart data (Python side to avoid Jinja2 + JS conflict)
    # NOTE: Use all_rows_original to get ALL K values, not just filtered rows
    chart_data = {
        "histogram": None,
        "thresholds_dict": {}
    }
    
    rows_for_chart = all_rows_original if all_rows_original else rows
    if rows_for_chart:
        # Get histogram from first row
        first_hist = rows_for_chart[0].get('hist_data')
        if first_hist:
            if isinstance(first_hist, str):
                # Parse string representation to list
                import ast
                try:
                    chart_data["histogram"] = ast.literal_eval(first_hist)
                    print(f"[DEBUG] Histogram parsed: {len(chart_data['histogram'])} bins")
                except Exception as e:
                    try:
                        chart_data["histogram"] = json.loads(first_hist) if first_hist.startswith('[') else None
                        print(f"[DEBUG] Histogram parsed (JSON): {len(chart_data.get('histogram', [])) if chart_data['histogram'] else 0} bins")
                    except Exception as e2:
                        print(f"[DEBUG] Histogram parse failed: {e2}")
                        pass
            else:
                chart_data["histogram"] = first_hist
                print(f"[DEBUG] Histogram already list: {len(first_hist)} bins")
        
        # Collect thresholds by algorithm - FROM ALL ROWS
        print(f"[DEBUG] Starting to collect thresholds from {len(rows_for_chart)} rows")
        for row in rows_for_chart:
            algo = row.get('algo', '')
            k_val = row.get('K', 0)
            thr_str = row.get('thresholds', '')
            
            print(f"[DEBUG] Row: algo={algo}, K={k_val}, thr_str={thr_str[:50] if thr_str else 'EMPTY'}")
            
            if algo and k_val:
                if algo not in chart_data["thresholds_dict"]:
                    chart_data["thresholds_dict"][algo] = {}
                
                # Parse thresholds
                if thr_str:
                    try:
                        thrs = [int(x.strip()) for x in thr_str.split(',') if x.strip().isdigit()]
                        chart_data["thresholds_dict"][algo][k_val] = thrs
                        print(f"[DEBUG] Added {algo} K={k_val}: {len(thrs)} thresholds = {thrs[:5]}")
                    except Exception as e:
                        print(f"[DEBUG] Failed to parse thresholds for {algo} K={k_val}: {e}")
                        pass
        
        print(f"[DEBUG] Chart data prepared: histogram={bool(chart_data['histogram'])}, algos={list(chart_data['thresholds_dict'].keys())}, total_rows={len(rows_for_chart)}")
        print(f"[DEBUG] Full thresholds_dict: {chart_data['thresholds_dict']}")

    # Debug: print chart_data for verification
    chart_data_json = json.dumps(chart_data)
    print(f"[DEBUG] chart_data_json length: {len(chart_data_json)} bytes")
    
    try:
        return render_template('benchmark_result.html',
                               df_rows=rows,
                               all_rows_original=all_rows_original,
                               csv_data_url=csv_data_url,
                               img_preview=None,
                               hist_img=hist_url,
                               hist_thresh_img=hist_url,
                               hist_thresh_url=hist_url,
                               export_entries=export_entries,
                               all_entries_original=all_entries_original,
                               best_by_fe=best_by_fe,
                               best_by_psnr=best_by_psnr,
                               best_by_ssim=best_by_ssim,
                               fe_plot=fe_plot,
                               time_plot=time_plot,
                               psnr_plot=psnr_plot,
                               ssim_plot=ssim_plot,
                               dice_plot=dice_plot,
                               chart_data=chart_data_json)
    except Exception as e:
        print(f"[ERROR] render_template failed: {e}")
        import traceback
        traceback.print_exc()
        raise



if __name__ == "__main__":
    # Run development server when invoked as module:
    # from project root: python -m src.ui.app
    app.run(host="0.0.0.0", port=5000, debug=True)

