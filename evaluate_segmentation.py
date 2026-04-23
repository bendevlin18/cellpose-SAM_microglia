"""
Compute QC metrics from segmentation masks and print a summary table.

Metrics per image:
  - n_cells: total detected cells
  - median_area_px: median cell area in pixels
  - median_diam_px: median equivalent diameter in pixels
  - size_cv: coefficient of variation of cell areas (fragmentation indicator)
  - coverage_pct: fraction of image area covered by masks

Usage:
    conda run -n cellpose python evaluate_segmentation.py [--mask-dir masks]
"""
import argparse
import numpy as np
import tifffile
from pathlib import Path
from scipy import ndimage


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--mask-dir', default='masks')
    p.add_argument('--csv', default='metrics.csv', help='Save metrics to CSV')
    return p.parse_args()


def cell_metrics(masks):
    labels = np.unique(masks)
    labels = labels[labels > 0]
    if len(labels) == 0:
        return 0, np.nan, np.nan, np.nan, 0.0

    areas = np.array([(masks == lbl).sum() for lbl in labels])
    diams = 2 * np.sqrt(areas / np.pi)
    coverage = areas.sum() / masks.size

    return (
        len(labels),
        float(np.median(areas)),
        float(np.median(diams)),
        float(areas.std() / areas.mean()) if areas.mean() > 0 else np.nan,
        float(coverage * 100),
    )


def main():
    args = parse_args()
    mask_dir = Path(args.mask_dir)
    mask_files = sorted(mask_dir.glob('*_mask.tif'))

    if not mask_files:
        print(f'No mask files found in {mask_dir}. Run run_segmentation.py first.')
        return

    header = f"{'Image':<40} {'Cells':>6} {'Med.Area':>9} {'Med.Diam':>9} {'SizeCV':>7} {'Cover%':>7}"
    print(header)
    print('-' * len(header))

    rows = []
    for mask_path in mask_files:
        masks = tifffile.imread(str(mask_path))
        n_cells, med_area, med_diam, size_cv, cover = cell_metrics(masks)
        name = mask_path.stem.replace('_mask', '')
        print(f'{name:<40} {n_cells:>6} {med_area:>9.1f} {med_diam:>9.1f} {size_cv:>7.3f} {cover:>7.2f}')
        rows.append((name, n_cells, med_area, med_diam, size_cv, cover))

    if args.csv:
        import csv
        with open(args.csv, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['image', 'n_cells', 'median_area_px', 'median_diam_px', 'size_cv', 'coverage_pct'])
            w.writerows(rows)
        print(f'\nMetrics saved to {args.csv}')

    # Summary stats
    if rows:
        cell_counts = [r[1] for r in rows]
        print(f'\nCell count summary: mean={np.mean(cell_counts):.0f}, '
              f'min={np.min(cell_counts)}, max={np.max(cell_counts)}')


if __name__ == '__main__':
    main()
