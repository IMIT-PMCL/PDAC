"""
Batch extract 3D centerlines from vessel masks (NIfTI).
- Binarizes input, optionally keeps largest connected component
- 3D skeletonization (skimage.morphology.skeletonize_3d)
- Preserves input folder structure under output directory
"""

import argparse
from pathlib import Path
import sys
import numpy as np
import nibabel as nib
from skimage.morphology import skeletonize_3d
from scipy import ndimage as ndi

def keep_lcc(mask: np.ndarray) -> np.ndarray:
    lab, n = ndi.label(mask, structure=np.ones((3,3,3), dtype=np.uint8))
    if n <= 1:
        return mask
    counts = np.bincount(lab.ravel())
    counts[0] = 0
    keep = np.argmax(counts)
    return (lab == keep).astype(np.uint8)

def process_one(in_path: Path, out_path: Path, thresh: float, keep_largest: bool) -> None:
    nii = nib.load(str(in_path))
    data = nii.get_fdata()
    mask = (data > thresh).astype(np.uint8)

    if keep_largest:
        mask = keep_lcc(mask)

    center = skeletonize_3d(mask).astype(np.uint8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(center, nii.affine, header=nii.header), str(out_path))

def main():
    p = argparse.ArgumentParser(description="Batch 3D centerline extraction from NIfTI vessel masks.")
    p.add_argument("--input", required=True, help="Input folder containing .nii/.nii.gz masks")
    p.add_argument("--output", required=True, help="Output folder to save centerlines")
    p.add_argument("--recursive", action="store_true", help="Recursively search for masks")
    p.add_argument("--threshold", type=float, default=0.5, help="Binarization threshold (default: 0.5)")
    p.add_argument("--keep-lcc", action="store_true", help="Keep only the largest connected component")
    args = p.parse_args()

    in_dir = Path(args.input)
    out_dir = Path(args.output)

    patterns = ["*.nii.gz", "*.nii"]
    files = []
    for pat in patterns:
        if args.recursive:
            files += list(in_dir.rglob(pat))
        else:
            files += list(in_dir.glob(pat))
    files = sorted(files)

    if not files:
        print(f"[WARN] No NIfTI files found under: {in_dir}", file=sys.stderr)
        return

    print(f"[INFO] Found {len(files)} file(s). Writing outputs to {out_dir}")
    ok, fail = 0, 0

    for i, f in enumerate(files, 1):
        # Mirror input relative path under output
        rel = f.relative_to(in_dir)
        out_path = out_dir / rel
        # keep filename; just write NIfTI (same extension)
        try:
            process_one(f, out_path, args.threshold, args.keep_lcc)
            ok += 1
            print(f"[{i}/{len(files)}] OK: {rel}")
        except Exception as e:
            fail += 1
            print(f"[{i}/{len(files)}] FAIL: {rel} -> {e}", file=sys.stderr)

    print(f"[DONE] Success: {ok}, Failed: {fail}, Output dir: {out_dir}")

if __name__ == "__main__":
    main()