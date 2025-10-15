"""
Quantify tumor–vessel encasement angles along a vessel centerline.

Inputs:
  - vessel_mask.nii.gz: binary mask for a single vessel (e.g., SMA)
  - tumor_mask.nii.gz : binary tumor mask
  - centerline.nii.gz : binary centerline mask for the vessel

Outputs:
  - CSV with per-point angles and a summary row
  - (optional) PNG slices visualizing vessel/tumor overlap per sampled plane
"""

import argparse
from pathlib import Path
import sys
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from scipy.ndimage import map_coordinates, center_of_mass, label
from scipy.spatial.distance import euclidean
from math import acos, degrees
import networkx as nx

# ------------------------- I/O -------------------------

def load_bin(path: Path):
    nii = nib.load(str(path))
    data = nii.get_fdata()
    return (data > 0.5).astype(np.uint8), nii.affine, nii.header

def save_csv(rows, out_csv: Path, header=None):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", encoding="utf-8") as f:
        if header:
            f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(map(str, r)) + "\n")

# ------------------------- centerline ordering -------------------------

def skeleton_to_graph(skel: np.ndarray) -> nx.Graph:
    """Build 26-neighborhood graph from centerline/skeleton voxels."""
    coords = np.argwhere(skel > 0)
    idx = {tuple(c): i for i, c in enumerate(map(tuple, coords))}
    G = nx.Graph()
    for c in coords:
        c = tuple(c)
        i = idx[c]
        G.add_node(i, xyz=c)
        z,y,x = c
        for dz in (-1,0,1):
            for dy in (-1,0,1):
                for dx in (-1,0,1):
                    if dx==dy==dz==0: 
                        continue
                    n = (z+dz, y+dy, x+dx)
                    j = idx.get(n, None)
                    if j is not None:
                        w = np.sqrt(dx*dx + dy*dy + dz*dz)
                        G.add_edge(i, j, weight=w)
    return G

def pick_endpoints(G: nx.Graph):
    start = next(iter(G.nodes))
    # farthest twice → approximate graph diameter endpoints
    lengths = nx.single_source_dijkstra_path_length(G, start, weight='weight')
    a = max(lengths, key=lengths.get)
    lengths = nx.single_source_dijkstra_path_length(G, a, weight='weight')
    b = max(lengths, key=lengths.get)
    return a, b

def ordered_centerline(center_mask: np.ndarray):
    """Return ordered list of voxels along main path."""
    G = skeleton_to_graph(center_mask)
    if len(G) == 0:
        raise ValueError("Empty centerline mask.")
    a, b = pick_endpoints(G)
    path_nodes = nx.shortest_path(G, a, b, weight='weight')
    coords = np.array([G.nodes[n]['xyz'] for n in path_nodes], dtype=float)
    return coords  # (N,3) in voxel indices z,y,x

# ------------------------- geometry helpers -------------------------

def normalize(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms

def local_frames(path_xyz: np.ndarray):
    """Compute tangent / (normal, binormal) per point along path."""
    # forward difference + tail repeat
    tangents = np.diff(path_xyz, axis=0)
    tangents = np.vstack([tangents, tangents[-1]])
    T = normalize(tangents)
    frames = []
    for t in T:
        # pick a non-colinear vector to cross with
        base = np.array([1,0,0]) if not np.allclose(t[:2], 0) else np.array([0,1,0])
        n = np.cross(t, base); n = n / (np.linalg.norm(n) + 1e-8)
        b = np.cross(t, n);    b = b / (np.linalg.norm(b) + 1e-8)
        frames.append((t, n, b))
    return frames  # list of (t, n, b)

def plane_grid(point, n, b, size=256, spacing=1.0):
    """Return 3x(HW) coordinates in voxel space for resampling."""
    half = size // 2
    gx, gy = np.meshgrid(np.arange(-half, half) * spacing,
                         np.arange(-half, half) * spacing, indexing='xy')
    gx = gx.flatten(); gy = gy.flatten()
    coords = (point[:,None] + gx * n[:,None] + gy * b[:,None])
    return coords.reshape(3, size, size)

def reslice(vol, coords3):
    return map_coordinates(vol, [coords3[0], coords3[1], coords3[2]], order=1, mode='nearest')

# ------------------------- angle on 2D slice -------------------------

def boundary_points(region_mask, tumor_mask):
    pts = []
    H, W = region_mask.shape
    for x in range(1, H-1):
        for y in range(1, W-1):
            if region_mask[x,y] and tumor_mask[x,y]:
                nb = region_mask[x-1:x+2, y-1:y+2]
                if np.any(nb == 0):
                    pts.append((x,y))
    return pts

def wrap_angle_deg(region_mask, tumor_mask):
    pts = boundary_points(region_mask, tumor_mask)
    if len(pts) < 2:
        return None
    b, c = pts[0], pts[-1]
    centroid = center_of_mass(region_mask)
    dab = euclidean(centroid, b)
    dac = euclidean(centroid, c)
    dbc = euclidean(b, c)
    denom = max(2 * dab * dac, 1e-6)
    cos_val = (dab**2 + dac**2 - dbc**2) / denom
    cos_val = np.clip(cos_val, -1.0, 1.0)
    raw = degrees(acos(cos_val))
    centroid_in_tumor = tumor_mask[int(centroid[0]), int(centroid[1])] > 0
    return 360 - raw if centroid_in_tumor else raw

def analyze_slice(vessel2d, tumor2d, min_area=10, overlap_ratio=0.5):
    if vessel2d.sum() == 0: return [("no_vessel", None)]
    if tumor2d.sum()  == 0: return [("no_tumor",  None)]
    labeled, n = label(vessel2d > 0)
    results = []
    for k in range(1, n+1):
        region = (labeled == k)
        area = int(region.sum())
        if area < min_area:
            continue
        inter = np.logical_and(region, tumor2d > 0)
        inter_area = int(inter.sum())
        if inter_area == 0:
            results.append(("no_contact", None)); continue
        if inter_area == area:
            results.append(("complete_encasement", 360.0)); continue
        if inter_area > overlap_ratio * area:
            ang = wrap_angle_deg(region, tumor2d)
            results.append(("partial_encasement", ang))
        else:
            results.append(("minor_contact", None))
    return results

# ------------------------- main pipeline -------------------------

def quantify_angles(vessel_path: Path, tumor_path: Path, centerline_path: Path,
                    out_dir: Path, size=256, spacing=1.0,
                    min_area=10, overlap_ratio=0.5, save_png=False):

    vessel, _, _ = load_bin(vessel_path)
    tumor , _, _ = load_bin(tumor_path)
    center, _, _ = load_bin(centerline_path)

    # order the centerline voxels
    path_xyz = ordered_centerline(center)  # (N,3) in z,y,x
    frames = local_frames(path_xyz)

    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [("index","z","y","x","angle_deg","relation")]

    max_angle = 0.0
    for i, (p, frm) in enumerate(zip(path_xyz, frames), start=1):
        t, n, b = frm
        coords3 = plane_grid(p, n, b, size=size, spacing=spacing)
        # map_coordinates expects order (z,y,x)
        vessel2d = reslice(vessel, coords3)
        tumor2d  = reslice(tumor , coords3)

        # binarize slices
        v2 = (vessel2d > 0.5).astype(np.uint8)
        t2 = (tumor2d  > 0.5).astype(np.uint8)

        results = analyze_slice(v2, t2, min_area=min_area, overlap_ratio=overlap_ratio)

        # choose max angle for this plane (if multiple components)
        if results:
            # pick the largest angle among components; None -> -inf
            angle_vals = [r[1] if (r[1] is not None) else -1 for r in results]
            best_idx = int(np.argmax(angle_vals))
            relation, angle = results[best_idx]
        else:
            relation, angle = ("no_region", None)

        if angle is None:
            angle_out = ""
        else:
            angle_out = f"{angle:.4f}"
            max_angle = max(max_angle, angle)

        rows.append((i, int(p[0]), int(p[1]), int(p[2]), angle_out, relation))

        if save_png:
            # simple overlay for QA
            fig = plt.figure()
            plt.imshow(v2, cmap="gray", alpha=0.8)
            plt.imshow(t2, cmap="hot",  alpha=0.35)
            title = f"idx {i} | {relation}"
            if angle is not None:
                title += f" | angle={angle:.2f}°"
            plt.title(title)
            fig.savefig(out_dir / f"slice_{i:04d}.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

    # summary
    rows.append(("summary","","","",f"{max_angle:.4f}","max_angle_deg"))
    save_csv(rows, out_dir / "angles.csv")

    return max_angle

# ------------------------- CLI -------------------------

def main():
    ap = argparse.ArgumentParser(description="Quantify tumor–vessel encasement angles along a centerline.")
    ap.add_argument("--vessel", required=True, help="Path to vessel mask NIfTI (.nii/.nii.gz)")
    ap.add_argument("--tumor",  required=True, help="Path to tumor mask NIfTI (.nii/.nii.gz)")
    ap.add_argument("--centerline", required=True, help="Path to centerline NIfTI (.nii/.nii.gz)")
    ap.add_argument("--out", required=True, help="Output folder")
    ap.add_argument("--size", type=int, default=256, help="Plane size (pixels)")
    ap.add_argument("--spacing", type=float, default=1.0, help="In-plane sampling step in voxel units")
    ap.add_argument("--min-area", type=int, default=10, help="Min vessel component area (px) on slice")
    ap.add_argument("--overlap-ratio", type=float, default=0.5, help="Intersection/region area ratio to call encasement")
    ap.add_argument("--save-png", action="store_true", help="Save per-slice PNG overlays")
    args = ap.parse_args()

    max_ang = quantify_angles(
        vessel_path=Path(args.vessel),
        tumor_path=Path(args.tumor),
        centerline_path=Path(args.centerline),
        out_dir=Path(args.out),
        size=args.size,
        spacing=args.spacing,
        min_area=args.min_area,
        overlap_ratio=args.overlap_ratio,
        save_png=args.save_png
    )
    print(f"[DONE] Max encasement angle = {max_ang:.2f}° | outputs -> {args.out}")

if __name__ == "__main__":
    main()
