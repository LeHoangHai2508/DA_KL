"""CLI to compare algorithms on images and report PSNR/SSIM/DICE/time with benchmark plots."""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import List, Dict, Any

import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.cli.seed_logger import SeedLogger
from src.metrics.fuzzy_entropy import histogram_from_image, compute_fuzzy_entropy
from src.optim.woa import woa_optimize
from src.optim.mfwoa_multitask import mfwoa_multitask
from src.optim.pso import pso_optimize
from src.seg.thresholding import apply_thresholds
from src.metrics.metrics import psnr, ssim, dice_macro


def greedy_otsu_multi(hist: np.ndarray, K: int) -> List[int]:
    """Greedy multi-level Otsu: iteratively add thresholds that maximize between-class variance."""
    levels = np.arange(256)
    prob = hist / (hist.sum() + 1e-12)

    def between_class_variance(thresholds: List[int]) -> float:
        bins = [0] + sorted(thresholds) + [256]
        total_mean = (prob * levels).sum()
        bc_var = 0.0
        for i in range(len(bins) - 1):
            a, b = bins[i], bins[i + 1]
            p = prob[a:b].sum()
            if p <= 0:
                continue
            mean = (prob[a:b] * levels[a:b]).sum() / (p + 1e-12)
            bc_var += p * (mean - total_mean) ** 2
        return float(bc_var)

    thresholds: List[int] = []
    candidates = list(range(1, 255))
    for _ in range(K):
        best_t = None
        best_score = -1.0
        for t in candidates:
            if t in thresholds:
                continue
            sc = between_class_variance(thresholds + [t])
            if sc > best_score:
                best_score = sc
                best_t = t
        if best_t is None:
            break
        thresholds.append(best_t)
    return sorted(thresholds)


def visualize_result(img_gray: np.ndarray, thresholds: List[int]) -> np.ndarray:
    """Map mỗi lớp thành midpoint cường độ để tính PSNR/SSIM nhất quán."""
    labels = apply_thresholds(img_gray, thresholds)
    bins = [0] + sorted(thresholds) + [255]
    mids = [int(round((bins[i] + bins[i + 1]) / 2.0)) for i in range(len(bins) - 1)]
    out = np.zeros_like(img_gray, dtype=np.uint8)
    for cls_id, mid in enumerate(mids):
        out[labels == cls_id] = mid
    return out


def run_algorithms_on_image(
    img: np.ndarray,
    K: int,
    pop: int,
    iters: int,
    *,
    seed: int = 42,
    save_per_seed: bool = False,
    save_root: str | Path = "results",
    image_stem: str | None = None,
    noise_tag: str | None = None,
    ground_truth: np.ndarray | None = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Chạy Otsu/PSO/WOA/MFWOA (single-K) trên 1 ảnh và trả dict kết quả.
    Nếu save_per_seed=True thì sẽ lưu CSV + mask + overlay cho từng seed/thuật toán/K.
    """
    try:
        from src.cli.seed_logger import SeedLogger
    except Exception:
        SeedLogger = None

    def _thresholds_to_mask(img_gray: np.ndarray, ths: list[int]) -> np.ndarray:
        if img_gray.ndim == 3:
            img_gray = cv2.cvtColor(img_gray, cv2.COLOR_BGR2GRAY)
        t = np.sort(np.array(ths, dtype=np.int32))
        bins = np.concatenate(([0], t, [255]))
        mask = np.zeros_like(img_gray, dtype=np.uint8)
        C = len(bins) - 1
        for c in range(C):
            lo, hi = bins[c], bins[c+1]
            if c == C - 1:
                mask[img_gray >= lo] = c
            else:
                mask[(img_gray >= lo) & (img_gray < hi)] = c
        return mask

    img_gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist = histogram_from_image(img)
    results: Dict[str, Dict[str, Any]] = {}

    def fitness_from_pos(pos):
        ths = [int(round(x)) for x in pos]
        return compute_fuzzy_entropy(hist, ths, for_minimization=True)

    logger = None
    if save_per_seed and SeedLogger is not None:
        image_stem = image_stem or "image"
        logger = SeedLogger(save_root, image_stem, noise_tag)

    # ===================== OTSU =====================
    t0 = time.perf_counter()
    otsu_th = greedy_otsu_multi(hist, K)
    t_otsu = time.perf_counter() - t0
    vis_otsu = visualize_result(img, otsu_th)
    fe_otsu = compute_fuzzy_entropy(hist, otsu_th, for_minimization=False)
    fitness_otsu = compute_fuzzy_entropy(hist, otsu_th, for_minimization=True)
    psnr_otsu = float(psnr(img_gray, vis_otsu))
    ssim_otsu = float(ssim(img_gray, vis_otsu))
    results['Otsu'] = {
        'K': K, 'thresholds': otsu_th, 'time': t_otsu,
        'vis': vis_otsu, 'fe': fe_otsu, 'fitness': fitness_otsu,
        'psnr': psnr_otsu, 'ssim': ssim_otsu,
    }
    if logger:
        try:
            logger.log(seed=seed, algo="Otsu", K=K,
                metrics={"FE": fe_otsu, "PSNR": psnr_otsu, "SSIM": ssim_otsu, "time_ms": int(t_otsu*1000)},
                img_gray=img_gray, mask=_thresholds_to_mask(img_gray, otsu_th))
        except Exception as e:
            print(f"[SeedLogger] Otsu warn: {e}")

    # ===================== PSO =====================
    t0 = time.perf_counter()
    pso_th, pso_score = pso_optimize(hist, K, pop_size=pop, iters=iters, objective=fitness_from_pos)
    t_pso = time.perf_counter() - t0
    vis_pso = visualize_result(img, pso_th)
    fe_pso = compute_fuzzy_entropy(hist, pso_th, for_minimization=False)
    psnr_pso = float(psnr(img_gray, vis_pso))
    ssim_pso = float(ssim(img_gray, vis_pso))
    results['PSO'] = {
        'K': K, 'thresholds': pso_th, 'time': t_pso,
        'vis': vis_pso, 'fe': fe_pso, 'fitness': float(pso_score),
        'psnr': psnr_pso, 'ssim': ssim_pso,
    }
    if logger:
        try:
            logger.log(seed=seed, algo="PSO", K=K,
                metrics={"FE": fe_pso, "PSNR": psnr_pso, "SSIM": ssim_pso, "time_ms": int(t_pso*1000)},
                img_gray=img_gray, mask=_thresholds_to_mask(img_gray, pso_th))
        except Exception as e:
            print(f"[SeedLogger] PSO warn: {e}")

    # ===================== GA =====================
    t0 = time.perf_counter()

    # ===================== WOA =====================
    t0 = time.perf_counter()
    woa_th, woa_score = woa_optimize(hist, K, pop_size=pop, iters=iters, objective=fitness_from_pos)
    t_woa = time.perf_counter() - t0
    vis_woa = visualize_result(img, woa_th)
    fe_woa = compute_fuzzy_entropy(hist, woa_th, for_minimization=False)
    psnr_woa = float(psnr(img_gray, vis_woa))
    ssim_woa = float(ssim(img_gray, vis_woa))
    results['WOA'] = {
        'K': K, 'thresholds': woa_th, 'time': t_woa,
        'vis': vis_woa, 'fe': fe_woa, 'fitness': float(woa_score),
        'psnr': psnr_woa, 'ssim': ssim_woa,
    }
    if logger:
        try:
            logger.log(seed=seed, algo="WOA", K=K,
                metrics={"FE": fe_woa, "PSNR": psnr_woa, "SSIM": ssim_woa, "time_ms": int(t_woa*1000)},
                img_gray=img_gray, mask=_thresholds_to_mask(img_gray, woa_th))
        except Exception as e:
            print(f"[SeedLogger] WOA warn: {e}")

    # ===================== MFWOA (single-task) =====================
    t0 = time.perf_counter()
    hists = [hist]
    Ks = [K]
    best_ths, best_scores, mf_diag = mfwoa_multitask(hists, Ks, pop_size=pop, iters=iters)
    t_mf = time.perf_counter() - t0
    mf_th = best_ths[0]
    vis_mf = visualize_result(img, mf_th)
    fe_mf = float(best_scores[0]) if (isinstance(best_scores, (list, tuple)) and len(best_scores) > 0) \
            else compute_fuzzy_entropy(hist, mf_th, for_minimization=False)
    psnr_mf = float(psnr(img_gray, vis_mf))
    ssim_mf = float(ssim(img_gray, vis_mf))
    results['MFWOA'] = {
        'K': K, 'thresholds': mf_th, 'time': t_mf,
        'vis': vis_mf, 'fe': fe_mf,
        'fitness': float(-fe_mf),
        'psnr': psnr_mf, 'ssim': ssim_mf,
        'mf_history': mf_diag.get('history') if isinstance(mf_diag, dict) else None,
        'mf_nfe': mf_diag.get('nfe') if isinstance(mf_diag, dict) else None,
    }
    if logger:
        try:
            logger.log(seed=seed, algo="MFWOA", K=K,
                metrics={"FE": fe_mf, "PSNR": psnr_mf, "SSIM": ssim_mf, "time_ms": int(t_mf*1000)},
                img_gray=img_gray, mask=_thresholds_to_mask(img_gray, mf_th))
        except Exception as e:
            print(f"[SeedLogger] MFWOA warn: {e}")

    # ===================== TÍNH DICE CHO TẤT CẢ THUẬT TOÁN =====================
    gt_labels = None
    if ground_truth is not None:
        gt_labels = ground_truth if ground_truth.ndim == 2 else cv2.cvtColor(ground_truth, cv2.COLOR_BGR2GRAY)
        gt_labels = _thresholds_to_mask(gt_labels, results['MFWOA']['thresholds'])
    elif 'MFWOA' in results:
        gt_labels = _thresholds_to_mask(img_gray, results['MFWOA']['thresholds'])
    
    if gt_labels is not None:
        for algo, info in results.items():
            try:
                pred_labels = _thresholds_to_mask(img_gray, info['thresholds'])
                dice_score = dice_macro(pred_labels, gt_labels)
                results[algo]['dice'] = float(dice_score) if not np.isnan(dice_score) else 0.0
                print(f"[DEBUG] {algo}: DICE = {results[algo]['dice']:.4f}")
            except Exception as e:
                print(f"[WARN] DICE computation failed for {algo}: {e}")
                results[algo]['dice'] = 0.0
    else:
        for algo in results.keys():
            results[algo]['dice'] = np.nan

    return results


def plot_benchmark(results_data: list, out_dir: Path):
    """Plot benchmark comparisons."""
    import pandas as pd
    df = pd.DataFrame(results_data)

    if df.empty:
        return

    plt.style.use('seaborn')
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Algorithm Comparison')

    sns.boxplot(data=df, x='algo', y='time', ax=ax1)
    ax1.set_title('Execution Time')
    ax1.set_xlabel('Algorithm')
    ax1.set_ylabel('Time (seconds)')
    ax1.tick_params(axis='x', rotation=45)

    sns.boxplot(data=df, x='algo', y='psnr', ax=ax2)
    ax2.set_title('PSNR')
    ax2.set_xlabel('Algorithm')
    ax2.set_ylabel('PSNR')
    ax2.tick_params(axis='x', rotation=45)

    sns.boxplot(data=df, x='algo', y='ssim', ax=ax3)
    ax3.set_title('SSIM')
    ax3.set_xlabel('Algorithm')
    ax3.set_ylabel('SSIM')
    ax3.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(out_dir / 'benchmark.png', dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10, 6))
    for algo in sorted(df['algo'].unique()):
        scores = df[df['algo'] == algo]['score'].values
        if scores.size > 0:
            plt.plot(scores, label=algo)
    plt.title('Convergence Comparison (Fuzzy Entropy)')
    plt.xlabel('Image Index')
    plt.ylabel('Fuzzy Entropy')
    plt.legend()
    plt.grid(True)
    plt.savefig(out_dir / 'convergence.png', dpi=300, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Compare segmentation algorithms')
    parser.add_argument('--input', required=True)
    parser.add_argument('--K', type=int, default=3, help='Number of thresholds')
    parser.add_argument('--pop', type=int, default=30)
    parser.add_argument('--iters', type=int, default=100)
    parser.add_argument('--out', default='compare_results')
    args = parser.parse_args()

    from datetime import datetime
    out_dir = Path(args.out) if args.out and args.out.strip() != "" else Path("results") / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    input_path = Path(args.input)
    csv_path = out_dir / 'compare.csv'
    results_data = []

    with open(csv_path, 'w', newline='', encoding='utf-8') as cf:
        writer = csv.writer(cf)
        writer.writerow(['image', 'algo', 'K', 'thresholds', 'time', 'psnr', 'ssim', 'dice', 'fe', 'fitness'])

        if input_path.is_dir():
            images = sorted(list(input_path.glob('*.png')) + list(input_path.glob('*.tif')) + list(input_path.glob('*.jpg')))
        else:
            images = [input_path]

        for p in images:
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"[WARN] Cannot read image: {p}")
                continue

            res = run_algorithms_on_image(img, args.K, args.pop, args.iters)

            for algo, info in res.items():
                vis = info['vis']
                ps = info.get('psnr', 0.0)
                ss = info.get('ssim', 0.0)
                dice_val = info.get('dice', np.nan)
                fe_val = info.get('fe', 0.0)
                fitness_val = info.get('fitness', 0.0)
                K_used = info.get('K', args.K)

                writer.writerow([
                    p.name, algo, K_used,
                    ';'.join(map(str, info['thresholds'])),
                    f"{info['time']:.4f}", f"{ps:.4f}", f"{ss:.4f}",
                    f"{dice_val:.6f}",
                    f"{fe_val:.6f}", f"{fitness_val:.6f}"
                ])

                results_data.append({
                    'image': p.name, 'algo': algo, 'K': K_used,
                    'time': info['time'], 'psnr': ps, 'ssim': ss,
                    'dice': dice_val, 'score': fe_val, 'fe': fe_val,
                    'fitness': fitness_val,
                })

                vis_path = out_dir / f"{p.stem}_{algo}_K{K_used}.png"
                json_path = out_dir / f"{p.stem}_{algo}_K{K_used}.json"
                cv2.imwrite(str(vis_path), vis)
                with open(json_path, 'w', encoding='utf-8') as jf:
                    json.dump({
                        'algo': algo, 'K': K_used,
                        'thresholds': info['thresholds'],
                        'time': info['time'],
                        'psnr': ps, 'ssim': ss, 'dice': dice_val,
                        'fe': fe_val, 'fitness': fitness_val,
                    }, jf, indent=2)

    plot_benchmark(results_data, out_dir)
    print(f"Comparison saved -> {csv_path}")
    print(f"Benchmark plots saved -> {out_dir}/benchmark.png, {out_dir}/convergence.png")


if __name__ == '__main__':
    main()