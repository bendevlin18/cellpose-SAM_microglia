"""
Test generalization of best parameters across multiple images.

Fixes flow_threshold=1.0, tile_norm_blocksize=100, diameter=150.
Sweeps cellprob_threshold to find the value that best expands masks into processes.

Produces one PNG per image: rows = cellprob_threshold values, single column of overlay panels.
Also saves a side-by-side summary grid across all tested images.

Usage:
    conda run -n cellpose python generalization_sweep.py
    conda run -n cellpose python generalization_sweep.py --n-images 6
"""
import argparse
import numpy as np
import tifffile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from cellpose.models import CellposeModel
from cellpose.plot import mask_overlay


CELLPROB_THRESHOLDS = [-2.0, -1.5, -1.0, -0.5, 0.0]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--image-dir', default='images')
    p.add_argument('--output-dir', default='sweep')
    p.add_argument('--iba1-channel', type=int, default=1)
    p.add_argument('--dapi-channel', type=int, default=0)
    p.add_argument('--diameter', type=float, default=150)
    p.add_argument('--flow-threshold', type=float, default=1.0)
    p.add_argument('--tile-norm-blocksize', type=int, default=100)
    p.add_argument('--n-images', type=int, default=5,
                   help='Number of images to sample (picks one per unique subject prefix)')
    p.add_argument('--crop-size', type=int, default=512)
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


def pick_images(img_dir, n):
    """Pick one image per unique subject prefix, up to n images."""
    all_files = sorted(img_dir.glob('*.tif'))
    seen_subjects = set()
    picked = []
    for f in all_files:
        # Subject prefix is the part before the last _A01_Gxxx suffix
        parts = f.stem.split('_')
        subject = '_'.join(parts[:-2]) if len(parts) >= 3 else f.stem
        if subject not in seen_subjects:
            seen_subjects.add(subject)
            picked.append(f)
        if len(picked) >= n:
            break
    return picked


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(exist_ok=True)
    norm_param = {'normalize': True, 'tile_norm_blocksize': args.tile_norm_blocksize}

    images = pick_images(Path(args.image_dir), args.n_images)
    print(f'Testing on {len(images)} images:')
    for f in images:
        print(f'  {f.name}')
    print(f'\nFixed: d={args.diameter}, ft={args.flow_threshold}, tnb={args.tile_norm_blocksize}')
    print(f'Sweeping cellprob_threshold: {CELLPROB_THRESHOLDS}\n')

    model = CellposeModel(gpu=True)

    n_rows = len(CELLPROB_THRESHOLDS)
    n_cols = len(images)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))
    fig.suptitle(
        f'd={args.diameter}  ft={args.flow_threshold}  tnb={args.tile_norm_blocksize}\n'
        f'rows=cellprob_threshold  cols=image',
        fontsize=11,
    )

    for j, img_path in enumerate(images):
        img = tifffile.imread(str(img_path))
        img = center_crop(img, args.crop_size)
        h, w = img.shape[1], img.shape[2]

        img_hwc = np.zeros((h, w, 3), dtype=img.dtype)
        img_hwc[:, :, 0] = img[args.iba1_channel]
        img_hwc[:, :, 1] = img[args.dapi_channel]
        bg = norm8(img[args.iba1_channel])

        short_name = img_path.stem.replace('Stitch_', '')

        for i, cpt in enumerate(CELLPROB_THRESHOLDS):
            masks, _, _ = model.eval(
                img_hwc,
                diameter=args.diameter,
                channel_axis=2,
                flow_threshold=args.flow_threshold,
                cellprob_threshold=cpt,
                normalize=norm_param,
            )
            n_cells = int(masks.max())
            print(f'  {short_name}  cp={cpt:>5}: {n_cells} cells')

            ax = axes[i][j]
            ax.imshow(mask_overlay(bg, masks))
            if i == 0:
                ax.set_title(short_name, fontsize=7)
            ax.set_ylabel(f'cp={cpt}', fontsize=8) if j == 0 else None
            ax.axis('off')

        print()

    out_path = out_dir / 'generalization_sweep.png'
    fig.savefig(str(out_path), dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'Grid saved: {out_path}')


if __name__ == '__main__':
    main()
