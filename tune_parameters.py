"""
Step 1 of the tuning workflow: parameter grid search on tile-sized crops.

Samples crops from several images × regions and runs a phased parameter grid on
each, using raw model.eval() at tile size (same as what segment_tiled feeds the
model, so results transfer directly to the full pipeline).

Outputs:
    <outdir>/<sample>__<combo>.png   labelled previews, one per (crop, combo)
    <outdir>/summary.csv             one row per (crop, combo) with cell count and
                                     mask-size stats (min/max/mean/median in pixels)

Phased grid (override any individual param with its CLI flag):
    --phase 1  diameter × cellprob_threshold  (defaults: 4×4 = 16 combos)
    --phase 2  flow_threshold × pix_filter    (defaults: 4×4 = 16 combos)
    --phase 3  tile_norm_blocksize × niter    (defaults: 4×4 = 16 combos)

Usage:
    conda run -n cellpose python -u tune_parameters.py                    # phase 1
    conda run -n cellpose python -u tune_parameters.py --phase 2 --diameter 150 --cellprob -2.0
    conda run -n cellpose python -u tune_parameters.py --diameter 100,150,200 --cellprob -2.0,-1.0
"""
import argparse
import csv
import sys
import time
import numpy as np

sys.stdout.reconfigure(line_buffering=True)

import tifffile
from pathlib import Path
from cellpose.models import CellposeModel
from bens_cellpose_utils import preview_cellpose_params


KNOWN_GOOD = {
    'diameter': 150,
    'cellprob_threshold': -2.0,
    'flow_threshold': 1.0,
    'pix_filter': 500,
    'tile_norm_blocksize': 100,
    'niter': None,
}

PHASE_GRIDS = {
    1: {
        'diameter':           [75, 150, 225, 300],
        'cellprob_threshold': [-3.0, -2.0, -1.0, 0.0],
    },
    2: {
        'flow_threshold': [0.4, 1.0, 1.5, 2.0],
        'pix_filter':     [250, 500, 1000, 2000],
    },
    3: {
        'tile_norm_blocksize': [0, 50, 100, 200],
        'niter':               [None, 200, 500, 1000],
    },
}

CSV_COLUMNS = [
    'sample', 'image', 'region', 'crop', 'combo',
    'diameter', 'cellprob_threshold', 'flow_threshold',
    'pix_filter', 'tile_norm_blocksize', 'niter',
    'n_cells', 'size_min', 'size_max', 'size_mean', 'size_median',
    'error',
]


def parse_list(arg, cast):
    if arg is None:
        return None
    out = []
    for part in arg.split(','):
        part = part.strip()
        if part.lower() == 'none':
            out.append(None)
        else:
            out.append(cast(part))
    return out


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--image-dir', default='images')
    p.add_argument('--output-dir', default='tune')
    p.add_argument('--phase', type=int, default=1, choices=[1, 2, 3])
    p.add_argument('--n-images', type=int, default=4)
    p.add_argument('--regions-per-image', type=int, default=2)
    p.add_argument('--crops-per-region', type=int, default=3)
    p.add_argument('--crop-size', type=int, default=448,
                   help='Tile size in px (inner 384 + 2*64 context, matches segment_tiled)')
    p.add_argument('--jitter', type=int, default=200,
                   help='Max random offset (px) of crops around each region centre')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--iba1-channel', type=int, default=1)
    p.add_argument('--dapi-channel', type=int, default=0)
    p.add_argument('--label-fontsize', type=float, default=2)

    # Per-parameter overrides (comma-separated). If set, overrides the phase grid.
    p.add_argument('--diameter', default=None,
                   help='e.g. "150" or "100,150,200"')
    p.add_argument('--cellprob', dest='cellprob_threshold', default=None)
    p.add_argument('--flow', dest='flow_threshold', default=None)
    p.add_argument('--pix-filter', default=None)
    p.add_argument('--tnb', dest='tile_norm_blocksize', default=None)
    p.add_argument('--niter', default=None,
                   help='e.g. "None,200,500"')

    return p.parse_args()


def build_grid(args):
    grid = {k: [v] for k, v in KNOWN_GOOD.items()}
    grid.update(PHASE_GRIDS[args.phase])

    overrides = {
        'diameter':            parse_list(args.diameter, float),
        'cellprob_threshold':  parse_list(args.cellprob_threshold, float),
        'flow_threshold':      parse_list(args.flow_threshold, float),
        'pix_filter':          parse_list(args.pix_filter, int),
        'tile_norm_blocksize': parse_list(args.tile_norm_blocksize, int),
        'niter':               parse_list(args.niter, int),
    }
    for k, v in overrides.items():
        if v is not None:
            grid[k] = v
    return grid


def sample_crops(tif_files, args, rng):
    """Return list of (image_stem, region_idx, crop_idx, crop_hwc) tuples."""
    chosen = rng.choice(len(tif_files), size=min(args.n_images, len(tif_files)),
                        replace=False)
    crops = []
    half = args.crop_size // 2
    for image_idx in chosen:
        tif_path = tif_files[image_idx]
        img = tifffile.imread(str(tif_path))  # (2, H, W)
        if img.ndim != 3 or img.shape[0] < 2:
            print(f'  skip {tif_path.name}: unexpected shape {img.shape}', flush=True)
            continue
        H, W = img.shape[1], img.shape[2]
        lo, hi = half, min(H, W) - half
        if hi <= lo:
            print(f'  skip {tif_path.name}: image too small for crop-size {args.crop_size}', flush=True)
            continue

        for region_idx in range(args.regions_per_image):
            cy = int(rng.integers(lo, hi))
            cx = int(rng.integers(lo, hi))
            for crop_idx in range(args.crops_per_region):
                jy = int(rng.integers(-args.jitter, args.jitter + 1))
                jx = int(rng.integers(-args.jitter, args.jitter + 1))
                y = max(half, min(H - half, cy + jy))
                x = max(half, min(W - half, cx + jx))
                y0, y1 = y - half, y - half + args.crop_size
                x0, x1 = x - half, x - half + args.crop_size

                crop_hwc = np.zeros((args.crop_size, args.crop_size, 3),
                                    dtype=img.dtype)
                crop_hwc[:, :, 0] = img[args.iba1_channel, y0:y1, x0:x1]  # IBA1
                crop_hwc[:, :, 1] = img[args.dapi_channel, y0:y1, x0:x1]  # DAPI
                sample = f'{tif_path.stem}__r{region_idx}_c{crop_idx}'
                crops.append({
                    'sample': sample,
                    'image': tif_path.stem,
                    'region': region_idx,
                    'crop': crop_idx,
                    'origin': (y0, x0),
                    'data': crop_hwc,
                })
    return crops


def main():
    args = parse_args()
    img_dir = Path(args.image_dir)
    out_dir = Path(args.output_dir) / f'phase{args.phase}'
    out_dir.mkdir(parents=True, exist_ok=True)

    tif_files = sorted(img_dir.glob('*.tif'))
    if not tif_files:
        print(f'No .tif files in {img_dir}', flush=True)
        sys.exit(1)

    grid = build_grid(args)
    n_combos = 1
    for v in grid.values():
        n_combos *= len(v)

    print(f'Phase {args.phase}  |  output: {out_dir}')
    print(f'Grid ({n_combos} combos):')
    for k, v in grid.items():
        marker = '*' if len(v) > 1 else ' '
        print(f'  {marker} {k}: {v}')

    rng = np.random.default_rng(args.seed)
    crops = sample_crops(tif_files, args, rng)
    print(f'\nSampled {len(crops)} crops from {args.n_images} images '
          f'({args.regions_per_image} regions × {args.crops_per_region} crops, '
          f'{args.crop_size}px, seed={args.seed})\n')

    model = CellposeModel(gpu=True)

    all_rows = []
    t_start = time.time()
    for ci, crop in enumerate(crops, 1):
        print(f'[{ci}/{len(crops)}] {crop["sample"]}  (origin y={crop["origin"][0]}, x={crop["origin"][1]})')
        rows = preview_cellpose_params(
            model=model,
            reference_image=crop['data'],
            param_grid=grid,
            output_dir=str(out_dir),
            sample_label=crop['sample'],
            save_previews=True,
            label_fontsize=args.label_fontsize,
        )
        for row in rows:
            row['image'] = crop['image']
            row['region'] = crop['region']
            row['crop'] = crop['crop']
            all_rows.append(row)

    summary_path = out_dir / 'summary.csv'
    with summary_path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for row in all_rows:
            w.writerow({k: row.get(k, '') for k in CSV_COLUMNS})

    elapsed = time.time() - t_start
    print(f'\nDone in {elapsed/60:.1f} min.')
    print(f'Previews -> {out_dir}/')
    print(f'Summary  -> {summary_path}')
    print(f'{len(all_rows)} rows ({len(crops)} crops × {n_combos} combos).')


if __name__ == '__main__':
    main()
