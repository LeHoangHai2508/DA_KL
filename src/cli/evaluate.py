"""CLI: evaluate an input grayscale image with MFWOA multitask and compute metrics.

Usage examples:
  python -m src.cli.evaluate --image dataset/lena.gray.png --Ks 2,3,4 --iters 200 --pop 50 --out results/run1 --gt path/to/gt.png

This script will:
 - run `mfwoa_multitask` on the image histogram repeated for each K
 - compute FE (Fuzzy Entropy), PSNR, SSIM, DICE (if GT provided), and timing
 - save `results.csv` and `results.json` into the output folder
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import csv

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim

from src.metrics.fuzzy_entropy import histogram_from_image, compute_fuzzy_entropy
from src.optim.mfwoa_multitask import mfwoa_multitask
from src.seg.thresholding import apply_thresholds


def reconstruct_from_labels(orig_arr: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Reconstruct grayscale image from segmentation labels by mapping each class to its mean gray value."""
    recon = np.zeros_like(orig_arr, dtype=np.float32)
    for cls in np.unique(labels):
        mask = labels == cls
        if mask.sum() == 0:
            continue
        recon[mask] = float(orig_arr[mask].mean())
    return recon.astype(np.uint8)


def compute_psnr_ssim(orig_arr: np.ndarray, recon_arr: np.ndarray):
    try:
        psnr = float(sk_psnr(orig_arr, recon_arr, data_range=255))
    except Exception:
        psnr = None
    try:
        # sk_ssim expects same dtype; use uint8
        ssim = float(sk_ssim(orig_arr.astype(np.uint8), recon_arr.astype(np.uint8), data_range=255))
    except Exception:
        ssim = None
    return psnr, ssim


def compute_dice(seg_labels: np.ndarray, gt_labels: np.ndarray):
    if seg_labels.shape != gt_labels.shape:
        return None
    gt = gt_labels
    seg = seg_labels
    unique_gt = np.unique(gt)
    if set(unique_gt.tolist()) <= {0, 1} or set(unique_gt.tolist()) <= {0, 255}:
        gt_bin = (gt != 0).astype(np.uint8)
        seg_bin = (seg != 0).astype(np.uint8)
        inter = int((gt_bin & seg_bin).sum())
        denom = int(gt_bin.sum() + seg_bin.sum())
        if denom == 0:
            return None
        return float(2.0 * inter / denom)
    else:
        labels = [int(l) for l in unique_gt if l != 0]
        dices = []
        for lbl in labels:
            gt_bin = (gt == lbl).astype(np.uint8)
            seg_bin = (seg == lbl).astype(np.uint8)
            inter = int((gt_bin & seg_bin).sum())
            denom = int(gt_bin.sum() + seg_bin.sum())
            if denom == 0:
                continue
            dices.append(2.0 * inter / denom)
        if len(dices) == 0:
            return None
        return float(np.mean(dices))


def main():
    parser = argparse.ArgumentParser(description="Evaluate image with MFWOA multitask and compute metrics")
    parser.add_argument("--image", required=True, help="Path to grayscale image")
    parser.add_argument("--Ks", required=True, help="Comma-separated list of K values (e.g. 2,3,4)")
    parser.add_argument("--iters", type=int, default=200, help="MFWOA iterations")
    parser.add_argument("--pop", type=int, default=50, help="population size total or per-task (see code)")
    parser.add_argument("--out", default="results/eval", help="Output directory")
    parser.add_argument("--gt", default=None, help="Optional ground-truth mask path (grayscale)")
    parser.add_argument("--seed", type=int, default=123, help="RNG seed")
    args = parser.parse_args()

    img_path = Path(args.image)
    if not img_path.exists():
        raise SystemExit(f"Image not found: {img_path}")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise SystemExit(f"Failed to read image as grayscale: {img_path}")
    hist = histogram_from_image(img)

    Ks = [int(k) for k in args.Ks.split(",") if k.strip()]
    if len(Ks) == 0:
        raise SystemExit("No Ks provided")

    rng = np.random.default_rng(args.seed)

    # Run MFWOA multitask: use same histogram repeated for each task (same image)
    hists = [hist for _ in Ks]
    t0 = time.perf_counter()
    best_thresholds_list, best_scores, diagnostics = mfwoa_multitask(
        hists,
        Ks,
        pop_size=args.pop,
        iters=args.iters,
        rng=rng,
    )
    t1 = time.perf_counter()
    total_time = t1 - t0

    rows = []
    # For each task, compute PSNR/SSIM/DICE using segmentation reconstructed
    gt_arr = None
    if args.gt:
        gt = cv2.imread(str(Path(args.gt)), cv2.IMREAD_UNCHANGED)
        if gt is None:
            print(f"Warning: cannot read GT: {args.gt}")
        else:
            if gt.ndim == 3:
                gt = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)
            gt_arr = gt

    for idx, K in enumerate(Ks):
        thr = best_thresholds_list[idx] if idx < len(best_thresholds_list) else []
        fe = None
        try:
            fe = float(compute_fuzzy_entropy(hist, thr, membership="triangular", for_minimization=False))
        except Exception:
            fe = None

        # segmentation labels
        labels = apply_thresholds(img, thr) if thr else np.zeros_like(img, dtype=np.int32)
        recon = reconstruct_from_labels(img, labels)
        psnr, ssim = compute_psnr_ssim(img.astype(np.float32), recon.astype(np.float32))
        dice = compute_dice(labels, gt_arr) if gt_arr is not None else None

        rows.append({
            "K": K,
            "thresholds": thr,
            "fe": fe,
            "psnr": psnr,
            "ssim": ssim,
            "dice": dice,
            "time_total": total_time,
        })

    # Save CSV and JSON
    csv_path = out_dir / "evaluation_results.csv"
    json_path = out_dir / "evaluation_results.json"
    with open(csv_path, "w", newline="", encoding="utf-8") as cf:
        writer = csv.writer(cf)
        writer.writerow(["K", "thresholds", "fe", "psnr", "ssim", "dice", "time_total"])
        for r in rows:
            writer.writerow([r["K"], ";".join(map(str, r["thresholds"])), r["fe"], r["psnr"], r["ssim"], r["dice"], r["time_total"]])

    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump({"image": str(img_path), "results": rows, "diagnostics": diagnostics, "seed": int(args.seed)}, jf, indent=2)

    print(f"Saved CSV -> {csv_path}")
    print(f"Saved JSON -> {json_path}")
    for r in rows:
        print(f"K={r['K']}: thresholds={r['thresholds']} FE={r['fe']} PSNR={r['psnr']} SSIM={r['ssim']} DICE={r['dice']}")


if __name__ == "__main__":
    main()
