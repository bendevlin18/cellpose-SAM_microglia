"""
Grid-search key segmentation parameters on a single central 512x512 crop.

Produces two grids:
  sweep/sweep_grid_iba1only.png  — IBA1 grayscale input (recommended for microglia processes)
  sweep/sweep_grid_2ch.png       — IBA1+DAPI two-channel input (for comparison)

Rows = diameter, Cols = cellprob_threshold.

Usage:
    conda run -n cellpose python parameter_sweep.py
    conda run -n cellpose python parameter_sweep.py --crop-size 256  # faster
"""
import argparse
import itertools
import csv
import numpy as np
import tifffile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from cellpose.models import CellposeModel
from cellpose.plot import mask_overlay


# Larger diameters to capture microglia processes (typical range 60-150px)
DIAMETERS = [None, 40, 60, 80, 100, 150]
CELLPROB_THRESHOLDS = [-2.0, -1.0, 0.0, 0.5]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--image', default='images/Stitch_1B3_insula_A01_G001.tif')
    p.add_argument('--iba1-channel', type=int, default=1)
    p.add_argument('--dapi-channel', type=int, default=0)
    p.add_argument('--flow-threshold', type=float, default=0.4)
    p.add_argument('--crop-size', type=int, default=512)
    p.add_argument('--output-dir', default='sweep')
    return p.parse_args()


def norm8(arr):
    lo, hi = np.percentile(arr, 1), np.percentile(arr, 99)
    arr = np.clip(arr, lo, hi)
    if hi > lo:
        arr = (arr - lo) / (hi - lo)
    return (arr * 255).astype(np.uint8)


def center_crop(arr_chw, size):
    h, w = arr_chw.shape[1], arr_chw.shape[2]
    y0, x0 = (h - size) // 2, (w - size) // 2
    return arr_chw[:, y0:y0+size, x0:x0+size]


def run_grid(model, img_input, channel_axis, flow_threshold, bg, label, out_path):
    n_rows, n_cols = len(DIAMETERS), len(CELLPROB_THRESHOLDS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))
    fig.suptitle(
        f'{label} — flow_threshold={flow_threshold}\n'
        f'rows=diameter  cols=cellprob_threshold',
        fontsize=11,
    )

    results = []
    for i, diam in enumerate(DIAMETERS):
        for j, cpt in enumerate(CELLPROB_THRESHOLDS):
            masks, _, _ = model.eval(
                img_input,
                diameter=diam,
                channel_axis=channel_axis,
                flow_threshold=flow_threshold,
                cellprob_threshold=cpt,
            )
            n_cells = int(masks.max())
            results.append((diam, cpt, n_cells))
            print(f'  [{label}] d={str(diam):>4}, cp={cpt:>5}: {n_cells} cells')

            ax = axes[i][j]
            ax.imshow(mask_overlay(bg, masks))
            ax.set_title(f'd={diam} cp={cpt}\n{n_cells} cells', fontsize=8)
            ax.axis('off')

    fig.savefig(str(out_path), dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'  -> {out_path}')
    return results


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(exist_ok=True)

    img = tifffile.imread(args.image)             # (2, H, W)
    img = center_crop(img, args.crop_size)
    iba1 = img[args.iba1_channel]                 # (H, W)
    dapi = img[args.dapi_channel]                 # (H, W)

    # --- Input option 1: IBA1 grayscale only (H, W, 3) with IBA1 in all slots ---
    h, w = iba1.shape
    iba1_3ch = np.zeros((h, w, 3), dtype=img.dtype)
    iba1_3ch[:, :, 0] = iba1

    # --- Input option 2: IBA1 slot0 + DAPI slot1 (H, W, 3) ---
    two_ch = np.zeros((h, w, 3), dtype=img.dtype)
    two_ch[:, :, 0] = iba1
    two_ch[:, :, 1] = dapi

    bg = norm8(iba1)    # IBA1 background for both grids
    print(f'Crop: {h}x{w}  |  Running {len(DIAMETERS)*len(CELLPROB_THRESHOLDS)*2} combos\n')

    model = CellposeModel(gpu=True)

    print('=== IBA1 only ===')
    results_1ch = run_grid(
        model, iba1_3ch, channel_axis=2, flow_threshold=args.flow_threshold,
        bg=bg, label='IBA1 only',
        out_path=out_dir / 'sweep_grid_iba1only.png',
    )

    print('\n=== IBA1 + DAPI ===')
    results_2ch = run_grid(
        model, two_ch, channel_axis=2, flow_threshold=args.flow_threshold,
        bg=bg, label='IBA1+DAPI',
        out_path=out_dir / 'sweep_grid_2ch.png',
    )

    csv_path = out_dir / 'sweep_results.csv'
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['input', 'diameter', 'cellprob_threshold', 'n_cells'])
        for row in results_1ch:
            w.writerow(['iba1only'] + list(row))
        for row in results_2ch:
            w.writerow(['iba1+dapi'] + list(row))
    print(f'\nCSV: {csv_path}')
    print('Done. Compare sweep_grid_iba1only.png vs sweep_grid_2ch.png')


if __name__ == '__main__':
    main()
