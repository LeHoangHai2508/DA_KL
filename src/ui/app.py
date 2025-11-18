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


# ---------- Try import project MFWOA optimizer (if available) ----------
_optimize_fn = None
# Try several common names/locations used in projects; adjust as needed.
try:
    # e.g., src/optim/mfwoa.py -> mfwoa_optimize(...)
    from src.optim.mfwoa import mfwoa_optimize as _optimize_fn
except Exception:
    try:
        # e.g., src/optim/optimize.py -> optimize_thresholds(...)
        from src.optim.optimize import optimize_thresholds as _optimize_fn
    except Exception:
        try:
            # alternate names
            from src.optim.mfwoa import optimize as _optimize_fn
        except Exception:
            try:
                from src.optim.optimize import optimize as _optimize_fn
            except Exception:
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
    
    Two approaches:
    1. Class mean: assign each pixel its class mean gray value (simple but blurs edges)
    2. Threshold-based: use class centroid or mean for better gradient preservation
    
    Returns: psnr (float) or None, ssim (float) or None
    """
    orig_gray = original_pil.convert('L')
    arr = np.array(orig_gray, dtype=np.float32)
    
    # Reconstruct image by assigning each pixel its class mean
    recon = np.zeros_like(arr, dtype=np.float32)
    
    # Handle case where seg_labels is empty or all zeros
    if seg_labels is None or seg_labels.size == 0:
        return None, None
    
    max_cls = int(seg_labels.max()) if seg_labels.max() > 0 else 0
    
    # If no valid segmentation, return None
    if max_cls == 0 and (seg_labels == 0).all():
        return None, None
    
    # Compute representative value for each class (class mean)
    class_values = {}
    for cls in range(max_cls + 1):
        mask = (seg_labels == cls)
        if mask.sum() == 0:
            # No pixels in this class - use placeholder (e.g., middle value)
            class_values[cls] = 127.5
            continue
        class_values[cls] = arr[mask].mean()
    
    # Reconstruct: assign each pixel its class representative value
    for cls in range(max_cls + 1):
        mask = (seg_labels == cls)
        recon[mask] = class_values[cls]
    
    # Compute PSNR and SSIM
    try:
        psnr = float(sk_ms_psnr(arr, recon, data_range=255))
    except Exception as e:
        print(f"PSNR error: {e}")
        psnr = None
    try:
        ssim = float(sk_ms_ssim(arr.astype(np.uint8), recon.astype(np.uint8), data_range=255))
    except Exception as e:
        print(f"SSIM error: {e}")
        ssim = None
    return psnr, ssim

# --- helper: compute DICE between segmentation and ground-truth mask ---
def compute_dice(seg_labels, gt_labels):
    """
    Compute DICE score between segmentation and ground-truth labels.
    
    If gt_labels is binary mask (0/1 or 0/255): compute DICE for foreground.
    If gt_labels has multiple labels (0..n): compute average DICE across classes.
    
    Returns: dice_score (float) or None if shapes mismatch or no valid data.
    """
    if seg_labels is None or gt_labels is None:
        return None
    if seg_labels.shape != gt_labels.shape:
        return None
    
    seg = seg_labels.astype(np.int32)
    gt = gt_labels.astype(np.int32)
    
    unique_gt = np.unique(gt)
    
    # Binary case: GT is {0, 1} or {0, 255}
    if len(unique_gt) <= 2 and (set(unique_gt.tolist()) <= {0, 1} or set(unique_gt.tolist()) <= {0, 255}):
        gt_bin = (gt != 0).astype(np.uint8)
        seg_bin = (seg != 0).astype(np.uint8)
        inter = np.logical_and(gt_bin, seg_bin).sum()
        denom = gt_bin.sum() + seg_bin.sum()
        if denom == 0:
            return None
        dice = 2.0 * inter / denom
        return float(dice)
    else:
        # Multi-class case: compute mean DICE for each class
        # Exclude background (0) from multi-class
        labels = [int(l) for l in unique_gt if l != 0]
        if not labels:
            return None
        
        dices = []
        for lbl in labels:
            gt_bin = (gt == lbl).astype(np.uint8)
            seg_bin = (seg == lbl).astype(np.uint8)
            inter = np.logical_and(gt_bin, seg_bin).sum()
            denom = gt_bin.sum() + seg_bin.sum()
            if denom == 0:
                # Class has no pixels in either GT or seg - skip
                continue
            dice = 2.0 * inter / denom
            dices.append(dice)
        
        if len(dices) == 0:
            return None
        
        # Return average DICE across all classes
        return float(np.mean(dices))
        return float(np.mean(dices))


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
    
    Returns:
      { 'algo': str, 'thresholds': [...], 'fe': float | None, 'time': float, 'seg_labels': np.array | None }
    """
    import inspect
    import time as _time
    
    start_t = _time.perf_counter()
    thresholds = []
    fe_val = None
    
    try:
        # ===== OTSU =====
        if algo == 'otsu':
            try:
                from skimage.filters import threshold_multiotsu
                arr = np.array(pil_image.convert('L'))
                thr = threshold_multiotsu(arr, classes=K+1) if K > 1 else threshold_multiotsu(arr, classes=2)
                thresholds = [int(t) for t in thr]
                fe_val = float(compute_fuzzy_entropy(hist, thresholds, membership=membership, for_minimization=False,
                                                     lambda_penalty=1.0, alpha_area=0.10, beta_membership=0.10, gamma_spacing=0.20))
            except Exception as e:
                print(f"Otsu error (K={K}): {e}")
                thresholds = []
                fe_val = None

        # ===== MFWOA (single-task) =====
        elif algo == 'mfwoa':
            try:
                from src.optim.mfwoa import mfwoa_optimize
                def objective_mfwoa(thr_or_pos):
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
                                                             lambda_penalty=1.0, alpha_area=0.10, beta_membership=0.10, gamma_spacing=0.20))
                        return fe_val
                    except Exception as exc:
                        print(f"MFWOA objective error (K={K}): {exc}")
                        return 0
                
                res = mfwoa_optimize(hist, K, pop_size=pop_size, iters=opt_iters, objective=objective_mfwoa)
                if isinstance(res, tuple) and len(res) >= 2:
                    thresholds, score = res[0], res[1]
                    fe_val = float(compute_fuzzy_entropy(hist, thresholds, membership=membership, for_minimization=False,
                                                         lambda_penalty=1.0, alpha_area=0.10, beta_membership=0.10, gamma_spacing=0.20))
                else:
                    thresholds = res
                    fe_val = float(compute_fuzzy_entropy(hist, thresholds, membership=membership, for_minimization=False,
                                                         lambda_penalty=1.0, alpha_area=0.10, beta_membership=0.10, gamma_spacing=0.20))
            except Exception as e:
                print(f"MFWOA error (K={K}): {e}")
                thresholds = []
                fe_val = None

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
                        fe_val = float(compute_fuzzy_entropy(hist, thr, membership=membership, for_minimization=True,
                                                             lambda_penalty=1.0, alpha_area=0.10, beta_membership=0.10, gamma_spacing=0.20))
                        return float(fe_val)
                    except Exception as exc:
                        print(f"WOA objective error (K={K}): {exc}")
                        return 1e8
                
                res = woa_optimize(hist, K, pop_size=pop_size, iters=opt_iters, objective=objective_woa)
                if isinstance(res, tuple) and len(res) >= 2:
                    thresholds, score = res[0], res[1]
                    fe_val = float(compute_fuzzy_entropy(hist, thresholds, membership=membership, for_minimization=False,
                                                         lambda_penalty=1.0, alpha_area=0.10, beta_membership=0.10, gamma_spacing=0.20))
                else:
                    thresholds = res
                    fe_val = float(compute_fuzzy_entropy(hist, thresholds, membership=membership, for_minimization=False,
                                                         lambda_penalty=1.0, alpha_area=0.10, beta_membership=0.10, gamma_spacing=0.20))
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
                        fe_val = float(compute_fuzzy_entropy(hist, thr, membership=membership, for_minimization=True,
                                                             lambda_penalty=1.0, alpha_area=0.10, beta_membership=0.10, gamma_spacing=0.20))
                        return float(fe_val)
                    except Exception as exc:
                        print(f"PSO objective error (K={K}): {exc}")
                        return 1e8
                
                res = pso_optimize(hist, K, pop_size=pop_size, iters=opt_iters, objective=objective_pso)
                if isinstance(res, tuple) and len(res) >= 2:
                    thresholds, score = res[0], res[1]
                    fe_val = float(compute_fuzzy_entropy(hist, thresholds, membership=membership, for_minimization=False,
                                                         lambda_penalty=1.0, alpha_area=0.10, beta_membership=0.10, gamma_spacing=0.20))
                else:
                    thresholds = res
                    fe_val = float(compute_fuzzy_entropy(hist, thresholds, membership=membership, for_minimization=False,
                                                         lambda_penalty=1.0, alpha_area=0.10, beta_membership=0.10, gamma_spacing=0.20))
            except Exception as e:
                print(f"PSO error (K={K}): {e}")
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
        if thresholds:
            try:
                th_arr = np.array([int(x) for x in thresholds])
                th_arr = enforce_threshold_constraints(th_arr)
                thresholds = [int(x) for x in th_arr]
                # Recompute FE on enforced thresholds
                try:
                    fe_val = float(compute_fuzzy_entropy(hist, thresholds, membership=membership, for_minimization=False,
                                                         lambda_penalty=1.0, alpha_area=0.10, beta_membership=0.10, gamma_spacing=0.20))
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
        'seg_labels': seg_labels
    }


def run_algorithms_and_benchmark(pil_image, hist, n_thresholds, membership, selected_algos, opt_iters, pop_size, optimization_mode='single'):
    """
    selected_algos: list of keys e.g. ['mfwoa','woa','otsu','pso']
    optimization_mode: 'single' (single-task per algo) or 'multitask' (MFWOA with knowledge transfer)
    
    Khi optimization_mode='single':
      - Chạy tất cả selected_algos với n_thresholds cố định
    
    Khi optimization_mode='multitask':
      - MFWOA chạy với Ks=[2,3,4,...n_thresholds] (knowledge transfer)
      - Các algo khác chạy lại cho từng K riêng
      - Trả về kết quả kiểu: mfwoa_multitask_K2, otsu_K2, woa_K2, ..., mfwoa_multitask_K3, otsu_K3, ...
    
    Returns:
      - results: list of dict { 'algo': str, 'thresholds': [...], 'fe': float, 'time': sec, 'seg_labels': np.array }
    """
    results = []

    # Helper to call an optimizer with a controlled budget (pop_size, iters)
    import inspect
    
    # Set a balanced budget for UI runs so all algorithms get comparable search effort
    UI_POP = int(max(4, int(pop_size)))
    UI_ITERS = int(opt_iters)

    # ===== MULTITASK MODE: MFWOA với Knowledge Transfer + tất cả algos cho từng K =====
    if optimization_mode == 'multitask' and 'mfwoa' in selected_algos:
        try:
            from src.optim.mfwoa_multitask import mfwoa_multitask
            
            # Tạo danh sách K từ 2 đến n_thresholds
            Ks_multi = list(range(2, n_thresholds + 1))  # [2, 3, 4, ..., n_thresholds]
            
            # Dùng cùng histogram cho tất cả tasks
            hists_multi = [hist] * len(Ks_multi)
            
            # RNG cho reproducibility
            rng_multi = np.random.default_rng()
            
            start_t = _time.perf_counter()
            
            # Gọi MFWOA multitask
            best_thresholds_all, best_scores_all, diagnostics = mfwoa_multitask(
                hists=hists_multi,
                Ks=Ks_multi,
                pop_size=int(max(4, pop_size)),
                iters=int(opt_iters),
                rng=rng_multi,
                rmp_init=0.3,
                membership=membership,
                lambda_penalty=1.0,
                alpha_area=0.10,
                beta_membership=0.10,
                gamma_spacing=0.20
            )
            
            end_t = _time.perf_counter()
            elapsed = end_t - start_t
            time_per_k = elapsed / max(1, len(Ks_multi))
            
            results = []
            
            # Loop through all K values
            for k_idx, k_val in enumerate(Ks_multi):
                # 1) MFWOA multitask result for this K
                thresholds_mf = best_thresholds_all[k_idx]
                fe_mf = best_scores_all[k_idx]
                
                try:
                    th_arr = np.array([int(x) for x in thresholds_mf]) if thresholds_mf else np.array([], dtype=int)
                    if th_arr.size > 0:
                        from src.seg.utils import enforce_threshold_constraints
                        th_arr = enforce_threshold_constraints(th_arr)
                        thresholds_mf = [int(x) for x in th_arr]
                except Exception:
                    pass

                try:
                    seg_labels_mf = apply_thresholds_to_image(pil_image, thresholds_mf) if thresholds_mf else None
                except Exception:
                    seg_labels_mf = None

                results.append({
                    'algo': f'mfwoa_multitask_K{k_val}',
                    'thresholds': thresholds_mf if thresholds_mf else [],
                    'fe': float(fe_mf) if fe_mf is not None else None,
                    'time': time_per_k,
                    'seg_labels': seg_labels_mf,
                })
                
                print(f"  └─ K={k_val} (multitask): FE={fe_mf:.4f}, T={','.join(map(str, thresholds_mf))}")

            print(f"✓ MFWOA multitask (Ks={Ks_multi}): {elapsed:.2f}s total, {len(results)} results")
            
            # Now run other algos (not mfwoa) in SINGLE-TASK mode with fixed K=n_thresholds
            print(f"\n  Running other algos in single-task mode (K={n_thresholds})...")
            for algo in selected_algos:
                if algo == 'mfwoa':
                    continue  # Already done in multitask
                
                res_other = _run_single_algo_for_K(
                    pil_image=pil_image,
                    hist=hist,
                    algo=algo,
                    K=n_thresholds,  # Single K value for single-task algos
                    opt_iters=UI_ITERS,
                    pop_size=UI_POP,
                    membership=membership,
                )
                # Rename algo to distinguish: algo_single
                res_other['algo'] = f"{algo}_single"
                results.append(res_other)
                print(f"  └─ {algo}_single (K={n_thresholds}): FE={res_other['fe']:.4f}")
            
            print(f"✓ Combined results: MFWOA multitask + other algos single-task = {len(results)} total")
            return results
        
        except Exception as e:
            print(f"MFWOA multitask error: {e}")
            import traceback
            traceback.print_exc()
            # Fallback: continue with single-task mode
            pass

    def _call_optimizer(fn, hist_arg, K, pop_size, iters, membership=None):
        """Call optimizer fn(hist, K, ...) trying to pass pop_size/iters/membership when supported."""
        if fn is None:
            raise ValueError("optimizer fn is None")
        sig = inspect.signature(fn)
        kwargs = {}
        if 'pop_size' in sig.parameters:
            kwargs['pop_size'] = pop_size
        if 'pop' in sig.parameters and 'pop_size' not in kwargs:
            kwargs['pop'] = pop_size
        if 'iters' in sig.parameters:
            kwargs['iters'] = iters
        if 'max_iter' in sig.parameters:
            kwargs['max_iter'] = iters
        if 'membership' in sig.parameters and membership is not None:
            kwargs['membership'] = membership
        # try to call with (hist, K, **kwargs)
        try:
            return fn(hist_arg, K, **kwargs)
        except TypeError:
            # fallback: try (hist, K, membership) positional
            try:
                if membership is not None:
                    return fn(hist_arg, K, membership)
                return fn(hist_arg, K)
            except Exception:
                # last resort: call with only hist (some multi-task variants use different signature)
                return fn(hist_arg, K)

    # SINGLE-TASK MODE: Run each algo with fixed K (n_thresholds)
    for algo in selected_algos:
        res = _run_single_algo_for_K(
            pil_image=pil_image,
            hist=hist,
            algo=algo,
            K=n_thresholds,
            opt_iters=UI_ITERS,
            pop_size=UI_POP,
            membership=membership
        )
        results.append(res)
        print(f"✓ {algo}: FE={res['fe']:.4f}, time={res['time']:.2f}s")
    
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
    except Exception as e:
        return f"Failed to open image: {e}", 400

    # parse
    n_thresholds_raw = request.form.get("n_thresholds", "").strip()
    try:
        n_thresholds = int(n_thresholds_raw) if n_thresholds_raw else None
    except ValueError:
        return "n_thresholds must be integer", 400
    auto_opt = request.form.get("auto_optimize") is not None
    benchmark = request.form.get("benchmark") is not None
    membership = request.form.get("membership", "triangular")
    try:
        opt_iters = int(request.form.get("opt_iters", "200"))
    except ValueError:
        opt_iters = 200

    # population size from UI
    try:
        pop_size = int(request.form.get("pop_size", "30"))
    except ValueError:
        pop_size = 30

    UI_POP_CAP = 500
    if pop_size < 4:
        pop_size = 4
    if pop_size > UI_POP_CAP:
        pop_size = UI_POP_CAP

    # If running from the web UI (benchmark mode), cap iterations to keep jobs reasonably fast
    # (user can run heavier experiments from CLI). This prevents the server from running very long
    # jobs by default. Cap value is conservative; adjust as needed.
    UI_OPT_ITERS_CAP = 500  # Increased from 120 to allow better convergence

    # Adaptive iterations: Khi K tăng, giảm iterations để giữ cân bằng thời gian
    # Độ phức tạp: O(pop × iters × K × 256)
    # K≤4: 100% iterations (nhanh)
    # K=5: 60% iterations (tối ưu)
    # K=6: 40% iterations (tối ưu)
    # K≥7: 25-30% iterations (tối ưu)
    if n_thresholds >= 8:
        opt_iters = max(30, int(opt_iters * 0.25))  # 25% iterations
    elif n_thresholds >= 6:
        opt_iters = max(50, int(opt_iters * 0.40))  # 40% iterations
    elif n_thresholds >= 5:
        opt_iters = max(75, int(opt_iters * 0.60))  # 60% iterations
    # Ngược lại (K≤4): giữ 100%

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

    # If no GT provided and benchmark requested, create a neutral binary GT at midpoint
    # This ensures DICE is computed for all algorithms (neutral baseline, not Otsu-derived)
    if gt_labels is None and benchmark:
        try:
            arr = np.array(pil.convert('L'))
            gt_labels = (arr >= 128).astype(np.uint8) * 255
            print(f"Auto-generated neutral GT (threshold=128), shape={gt_labels.shape}")
        except Exception as e:
            print(f"Failed to auto-generate GT: {e}")
            gt_labels = None

    # if benchmark requested -> run multiple algos; else run selected single mode
    if benchmark:
        # cap heavy optimizer iterations for web requests
        if opt_iters > UI_OPT_ITERS_CAP:
            opt_iters = UI_OPT_ITERS_CAP
        
        # Adaptive iterations based on K (number of thresholds)
        # Higher K increases computational complexity O(K*256*pop*iters), so reduce iterations
        # Scales down: K>=5:60%, K>=6:40%, K>=8:25% - ensures bounded total runtime
        if n_thresholds >= 8:
            opt_iters = max(30, int(opt_iters * 0.25))  # K>=8: 25% iterations (min 30)
        elif n_thresholds >= 6:
            opt_iters = max(50, int(opt_iters * 0.4))   # K>=6: 40% iterations (min 50)
        elif n_thresholds >= 5:
            opt_iters = max(75, int(opt_iters * 0.6))   # K>=5: 60% iterations (min 75)
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

                results = run_algorithms_and_benchmark(pil, hist, n_thresholds, membership, selected_algos, opt_iters, pop_size, optimization_mode=optimization_mode)

                # compute PSNR/SSIM/DICE where GT available
                rows = []
                for r in results:
                    seg = r['seg_labels']
                    if seg is None:
                        psnr = ssim = dice = None
                    else:
                        psnr, ssim = compute_psnr_ssim(pil, seg)
                        if gt_labels is None:
                            dice = None
                        else:
                            # Ensure GT and segmentation have same size
                            if seg.shape != gt_labels.shape:
                                print(f"Shape mismatch: seg {seg.shape} vs gt {gt_labels.shape}")
                                # Try to resize GT if needed
                                try:
                                    from PIL import Image as PILImage
                                    gt_pil_temp = PILImage.fromarray(gt_labels).resize(seg.shape[::-1], resample=PILImage.NEAREST)
                                    gt_resized = np.array(gt_pil_temp)
                                    dice = compute_dice(seg, gt_resized)
                                except:
                                    dice = None
                            else:
                                dice = compute_dice(seg, gt_labels)
                    rows.append({
                        'algo': r['algo'],
                        'thresholds': ','.join(map(str, r['thresholds'])) if r['thresholds'] else '',
                        'fe': r['fe'] if r['fe'] is not None else None,
                        'time': round(r['time'], 4),
                        'psnr': psnr,
                        'ssim': ssim,
                        'dice': dice,
                    })

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
        results = run_algorithms_and_benchmark(pil, hist, n_thresholds, membership, [selected_algo], opt_iters, pop_size, optimization_mode=optimization_mode)
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

    # Compute best results for comparison
    best_by_fe = None
    best_by_psnr = None
    best_by_ssim = None
    
    if rows:
        # Best by FE (highest FE value, excluding None)
        fe_rows = [r for r in rows if r.get('fe') is not None]
        if fe_rows:
            best_by_fe = max(fe_rows, key=lambda r: r['fe'])
        
        # Best by PSNR (highest PSNR, excluding None)
        psnr_rows = [r for r in rows if r.get('psnr') is not None]
        if psnr_rows:
            best_by_psnr = max(psnr_rows, key=lambda r: r['psnr'])
        
        # Best by SSIM (highest SSIM, excluding None)
        ssim_rows = [r for r in rows if r.get('ssim') is not None]
        if ssim_rows:
            best_by_ssim = max(ssim_rows, key=lambda r: r['ssim'])

    # Create small plot placeholders
    fe_plot = time_plot = psnr_plot = ssim_plot = dice_plot = None

    return render_template('benchmark_result.html',
                           df_rows=rows,
                           csv_data_url=csv_data_url,
                           img_preview=None,
                           hist_img=hist_url,
                           hist_thresh_img=hist_url,
                           hist_thresh_url=hist_url,
                           export_entries=export_entries,
                           best_by_fe=best_by_fe,
                           best_by_psnr=best_by_psnr,
                           best_by_ssim=best_by_ssim,
                           fe_plot=fe_plot,
                           time_plot=time_plot,
                           psnr_plot=psnr_plot,
                           ssim_plot=ssim_plot,
                           dice_plot=dice_plot)



if __name__ == "__main__":
    # Run development server when invoked as module:
    # from project root: python -m src.ui.app
    app.run(host="0.0.0.0", port=5000, debug=True)

