"""
Volume-based post-processing for segmentation masks (NIfTI).
- Remove small connected components by voxel-count or mm^3 thresholds
- Works on 3D binary masks (0/1). Non-binary inputs are binarized by threshold (>0.5).
- Preserves affine and header of input files.
"""

from pathlib import Path
import argparse
import sys
import numpy as np
import nibabel as nib
from scipy import ndimage as ndi

def load_mask(path: Path):
    nii = nib.load(str(path))
    data = nii.get_fdata()
    affine = nii.affine
    header = nii.header
    # binarize
    mask = (data > 0.5).astype(np.uint8)
    return mask, affine, header

def save_mask(mask: np.ndarray, affine, header, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(mask.astype(np.uint8), affine, header=header), str(out_path))

def component_volumes(mask: np.ndarray, voxel_volume_mm3: float):
    """Return labels map and dict: label -> (voxel_count, volume_mm3)."""
    labels, n = ndi.label(mask, structure=np.ones((3,3,3), dtype=np.uint8))
    stats = {}
    for lab in range(1, n+1):
        cnt = int((labels == lab).sum())
        stats[lab] = (cnt, cnt * voxel_volume_mm3)
    return labels, stats

def filter_by_threshold(labels: np.ndarray,
                        stats: dict,
                        min_voxels: int = None,
                        min_mm3: float = None) -> np.ndarray:
    keep = np.zeros_like(labels, dtype=bool)
    for lab, (cnt, vol_mm3) in stats.items():
        ok = False
        if min_voxels is not None:
            ok |= (cnt >= min_voxels)
        if min_mm3 is not None:
            ok |= (vol_mm3 >= min_mm3)
        if ok:
            keep |= (labels == lab)
    return keep.astype(np.uint8)

def main():
    ap = argparse.ArgumentParser(
        description="Remove small components from 3D NIfTI masks by volume thresholds."
    )
    ap.add_argument("--input", required=True, help="Input folder containing .nii/.nii.gz")
    ap.add_argument("--output", required=True, help="Output folder for filtered masks")
    ap.add_argument("--recursive", action="store_true", help="Search input recursively")
    # thresholds (choose one or both)
    ap.add_argument("--min-voxels", type=int, default=None,
                    help="Minimum connected-component size in voxels (e.g., 400)")
    ap.add_argument("--min-mm3", type=float, default=None,
                    help="Minimum connected-component volume in mm^3 (uses header zooms)")
    # convenience preset for phase-specific voxel thresholds (optional)
    ap.add_argument("--phase", choices=["arterial","venous"], default=None,
                    help="Optional preset: arterial=415 vox, venous=450 vox (adjust as needed)")
    args = ap.parse_args()

    if args.min_voxels is None and args.min_mm3 is None and args.phase is None:
        ap.error("Please specify at least one threshold: --min-voxels or --min-mm3 (or use --phase).")

    # apply phase presets if provided (you can change these to your own study's values)
    if args.phase == "arterial" and args.min_voxels is None and args.min_mm3 is None:
        args.min_voxels = 415
    if args.phase == "venous" and args.min_voxels is None and args.min_mm3 is None:
        args.min_voxels = 450

    in_dir = Path(args.input)
    out_dir = Path(args.output)

    patterns = ["*.nii.gz", "*.nii"]
    files = []
    for pat in patterns:
        files += list(in_dir.rglob(pat) if args.recursive else in_dir.glob(pat))
    files = sorted(files)

    if not files:
        print(f"[WARN] No NIfTI files found under: {in_dir}", file=sys.stderr)
        return

    ok = fail = 0
    print(f"[INFO] Found {len(files)} file(s). Output -> {out_dir}")

    for i, fp in enumerate(files, 1):
        rel = fp.relative_to(in_dir)
        outp = out_dir / rel
        try:
            mask, affine, header = load_mask(fp)
            # voxel volume (mm^3) from header zooms; fallback to 1.0 if missing
            zooms = header.get_zooms()
            if len(zooms) < 3:
                voxel_mm3 = 1.0
            else:
                voxel_mm3 = float(zooms[0] * zooms[1] * zooms[2])

            labels, stats = component_volumes(mask, voxel_mm3)
            filtered = filter_by_threshold(labels, stats,
                                           min_voxels=args.min_voxels,
                                           min_mm3=args.min_mm3)
            save_mask(filtered, affine, header, outp)
            ok += 1

            # simple log line with counts
            kept_components = int((ndi.label(filtered)[1]))
            print(f"[{i}/{len(files)}] OK {rel} | kept {kept_components} comp(s) "
                  f"| thr_vox={args.min_voxels} thr_mm3={args.min_mm3} voxel_mm3={voxel_mm3:.3f}")
        except Exception as e:
            fail += 1
            print(f"[{i}/{len(files)}] FAIL {rel} -> {e}", file=sys.stderr)

    print(f"[DONE] Success: {ok}, Failed: {fail}. Outputs in: {out_dir}")

if __name__ == "__main__":
    main()