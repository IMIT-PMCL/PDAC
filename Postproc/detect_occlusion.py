"""
Detect vessel interruption/occlusion from a vessel mask and its centerline.

Heuristic:
- Order centerline voxels (graph shortest path between two farthest points).
- For each ordered centerline voxel, check if vessel voxels exist within
  a spherical neighborhood of radius R (in voxels). This yields a 0/1 trace.
- If there exists a contiguous zero-run with length >= L, we mark a geometric gap.
- Optional connectivity check: if vessel mask has multiple 3D components and the
  two centerline ends belong to different components while a gap exists -> occlusion.

Outputs:
- Prints classification and gap stats; optionally writes a CSV and a NIfTI gap mask.
"""

from pathlib import Path
import argparse
import numpy as np
import nibabel as nib
from scipy import ndimage as ndi
import networkx as nx

# ---------- I/O ----------
def load_bin(path: Path):
    nii = nib.load(str(path))
    data = (nii.get_fdata() > 0.5).astype(np.uint8)
    return data, nii.affine, nii.header

def save_mask(mask, affine, header, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(mask.astype(np.uint8), affine, header=header), str(out_path))

# ---------- centerline ordering ----------
def skeleton_to_graph(skel: np.ndarray) -> nx.Graph:
    coords = np.argwhere(skel > 0)
    idx = {tuple(c): i for i, c in enumerate(map(tuple, coords))}
    G = nx.Graph()
    for c in coords:
        c = tuple(c); i = idx[c]
        G.add_node(i, xyz=c)
        z,y,x = c
        for dz in (-1,0,1):
            for dy in (-1,0,1):
                for dx in (-1,0,1):
                    if dx==dy==dz==0: 
                        continue
                    n = (z+dz, y+dy, x+dx)
                    j = idx.get(n)
                    if j is not None:
                        w = np.sqrt(dx*dx + dy*dy + dz*dz)
                        G.add_edge(i, j, weight=w)
    return G

def farthest_pair(G: nx.Graph):
    start = next(iter(G.nodes))
    d1 = nx.single_source_dijkstra_path_length(G, start, weight='weight')
    a = max(d1, key=d1.get)
    d2 = nx.single_source_dijkstra_path_length(G, a, weight='weight')
    b = max(d2, key=d2.get)
    return a, b

def ordered_centerline(center: np.ndarray) -> np.ndarray:
    G = skeleton_to_graph(center)
    if len(G) == 0:
        raise ValueError("Empty centerline mask.")
    a, b = farthest_pair(G)
    path_nodes = nx.shortest_path(G, a, b, weight='weight')
    coords = np.array([G.nodes[n]['xyz'] for n in path_nodes], dtype=int)
    return coords  # (N,3) z,y,x

# ---------- occlusion logic ----------
def ball_indices(radius: int):
    """Return offsets within a 3D ball of given radius (in voxels)."""
    r = int(radius)
    z,y,x = np.mgrid[-r:r+1, -r:r+1, -r:r+1]
    mask = (z*z + y*y + x*x) <= (radius*radius + 1e-6)
    offs = np.vstack([z[mask], y[mask], x[mask]]).T
    return offs

def sample_presence_along_path(vessel: np.ndarray, path_xyz: np.ndarray, radius: int):
    offs = ball_indices(radius)
    Z,Y,X = vessel.shape
    trace = []
    for (z,y,x) in path_xyz:
        zz = np.clip(z + offs[:,0], 0, Z-1)
        yy = np.clip(y + offs[:,1], 0, Y-1)
        xx = np.clip(x + offs[:,2], 0, X-1)
        present = int(vessel[zz,yy,xx].any())  # 1 if any vessel voxel within ball
        trace.append(present)
    return np.array(trace, dtype=int)

def longest_zero_run(arr: np.ndarray):
    """Return (max_len, start_idx, end_idx) of longest contiguous zeros."""
    if arr.size == 0:
        return 0, -1, -1
    # Pad with ones to split by zeros cleanly
    padded = np.r_[1, arr, 1]
    zero = (padded == 0).astype(int)
    # find transitions 1->0 and 0->1
    starts = np.where((zero[1:] == 1) & (zero[:-1] == 0))[0]
    ends   = np.where((zero[1:] == 0) & (zero[:-1] == 1))[0] - 1
    if starts.size == 0 or ends.size == 0:
        return 0, -1, -1
    lengths = ends - starts + 1
    k = int(np.argmax(lengths))
    return int(lengths[k]), int(starts[k]), int(ends[k])

def endpoints_components(vessel: np.ndarray, path_xyz: np.ndarray, radius: int):
    """Return connected-component labels around two endpoints."""
    labels, _ = ndi.label(vessel, structure=np.ones((3,3,3), dtype=np.uint8))
    offs = ball_indices(radius)
    Z,Y,X = vessel.shape

    def comp_at(pt):
        z,y,x = pt
        zz = np.clip(z + offs[:,0], 0, Z-1)
        yy = np.clip(y + offs[:,1], 0, Y-1)
        xx = np.clip(x + offs[:,2], 0, X-1)
        labs = labels[zz,yy,xx]
        labs = labs[labs > 0]
        if labs.size == 0:
            return 0
        # mode
        vals, counts = np.unique(labs, return_counts=True)
        return int(vals[np.argmax(counts)])

    c1 = comp_at(path_xyz[0])
    c2 = comp_at(path_xyz[-1])
    return c1, c2

def detect_occlusion(vessel_mask: np.ndarray,
                     centerline_mask: np.ndarray,
                     neighborhood_radius: int = 2,
                     min_gap_len: int = 5,
                     require_disconnected_components: bool = True):
    """
    Returns:
      dict with keys:
        - is_gap (bool)
        - gap_len, gap_start, gap_end
        - components_differ (bool)
        - is_occlusion (bool)  # final decision
        - trace (0/1 array)
        - path_xyz (Nx3)
    """
    path_xyz = ordered_centerline(centerline_mask)
    trace = sample_presence_along_path(vessel_mask, path_xyz, neighborhood_radius)
    gap_len, g0, g1 = longest_zero_run(trace)
    is_gap = gap_len >= int(min_gap_len)

    comp_diff = False
    if require_disconnected_components:
        c1, c2 = endpoints_components(vessel_mask, path_xyz, neighborhood_radius)
        comp_diff = (c1 != 0 and c2 != 0 and c1 != c2)

    is_occ = is_gap and (comp_diff if require_disconnected_components else True)

    return {
        "is_gap": bool(is_gap),
        "gap_len": int(gap_len),
        "gap_start": int(g0),
        "gap_end": int(g1),
        "components_differ": bool(comp_diff),
        "is_occlusion": bool(is_occ),
        "trace": trace,
        "path_xyz": path_xyz
    }

# ---------- CLI ----------
def main():
    import argparse, csv
    ap = argparse.ArgumentParser(description="Detect vessel interruption/occlusion from mask + centerline.")
    ap.add_argument("--vessel", required=True, help="Vessel mask NIfTI (.nii/.nii.gz)")
    ap.add_argument("--centerline", required=True, help="Centerline NIfTI (.nii/.nii.gz)")
    ap.add_argument("--radius", type=int, default=2, help="Neighborhood radius (voxels)")
    ap.add_argument("--min-gap", type=int, default=5, help="Minimum consecutive-missing length (voxels)")
    ap.add_argument("--no-comp-check", action="store_true", help="Do NOT require start/end in different components")
    ap.add_argument("--out-dir", default=None, help="Optional folder to save gap mask & CSV")
    args = ap.parse_args()

    vessel, aff, hdr = load_bin(Path(args.vessel))
    center, _, _ = load_bin(Path(args.centerline))

    res = detect_occlusion(
        vessel_mask=vessel,
        centerline_mask=center,
        neighborhood_radius=args.radius,
        min_gap_len=args.min_gap,
        require_disconnected_components=not args.no_comp_check
    )

    print(f"is_gap={res['is_gap']} gap_len={res['gap_len']} "
          f"gap_range=[{res['gap_start']},{res['gap_end']}] "
          f"components_differ={res['components_differ']} "
          f"is_occlusion={res['is_occlusion']}")

    if args.out_dir:
        out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        gap_mask = np.zeros_like(vessel, dtype=np.uint8)
        if res["is_gap"]:
            pts = res["path_xyz"][res["gap_start"]:res["gap_end"]+1]
            for (z,y,x) in pts:
                gap_mask[z,y,x] = 1
        save_mask(gap_mask, aff, hdr, out_dir / "gap_segment.nii.gz")
        with open(out_dir / "trace.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["index","present"])
            for i, v in enumerate(res["trace"], 1):
                w.writerow([i, int(v)])

if __name__ == "__main__":
    main()
