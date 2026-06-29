"""
Assemble and plot Bx on three orthogonal slices (XY, YZ, ZX) from per-proc
iPIC3D HDF tiles. MPI-parallel over files; each rank contributes only to the
three target planes, which are SUM-reduced on rank 0.
"""

import numpy as np
from mpi4py import MPI
import os, glob, h5py, argparse
import matplotlib.pyplot as plt

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# --------------------------- Arguments ---------------------------
parser = argparse.ArgumentParser(description="Plot Bx on XY/YZ/ZX slices from iPIC3D output")
parser.add_argument("dir_data",   type=str)
parser.add_argument("time_cycle", type=str)
parser.add_argument("xlen", type=int)
parser.add_argument("ylen", type=int)
parser.add_argument("zlen", type=int)
parser.add_argument("xmin", type=float)        ###! physical box extents (e.g. from .inp: 0..Lx)
parser.add_argument("xmax", type=float)
parser.add_argument("ymin", type=float)
parser.add_argument("ymax", type=float)
parser.add_argument("zmin", type=float)
parser.add_argument("zmax", type=float)
args = parser.parse_args()

dir_data   = args.dir_data
time_cycle = args.time_cycle
XLEN, YLEN, ZLEN = args.xlen, args.ylen, args.zlen
xmin, xmax = args.xmin, args.xmax              ###! used only for axis labelling, not assembly
ymin, ymax = args.ymin, args.ymax
zmin, zmax = args.zmin, args.zmax
num_expected_files = XLEN * YLEN * ZLEN
SHARED_PLANES = True

# --------------------------- Helpers ---------------------------
def proc_id_from_filename(fp: str) -> int:
    base = os.path.basename(fp)              # expects procNNN.hdf
    return int(base.replace("proc", "").replace(".hdf", ""))

def get_Bx(f: h5py.File, cycle: str) -> np.ndarray:
    return np.array(f[f"fields/Bx/{cycle}"])  # full 3D tile (nx,ny,nz)

def mapping_candidates():
    """proc_id -> (i,j,k). Candidate 'A' == iPIC3D's standard MPI_Cart
    ordering id = i*(YLEN*ZLEN) + j*ZLEN + k (k fastest, then j, then i).
    ⚠️ If plots look wrong, confirm ordering in VCtopology3D.cpp — the
    auto-detector only picks the LEAST-BAD of these six, never proves correctness."""
    def A(p): k=p%ZLEN; t=p//ZLEN; j=t%YLEN; i=t//YLEN; return i,j,k   # k,j,i
    def B(p): j=p%YLEN; t=p//YLEN; k=t%ZLEN; i=t//ZLEN; return i,j,k   # j,k,i
    def C(p): k=p%ZLEN; t=p//ZLEN; i=t%XLEN; j=t//XLEN; return i,j,k   # k,i,j
    def D(p): i=p%XLEN; t=p//XLEN; j=t%YLEN; k=t//YLEN; return i,j,k   # i,j,k
    def E(p): j=p%YLEN; t=p//YLEN; i=t%XLEN; k=t//XLEN; return i,j,k   # j,i,k
    def F(p): i=p%XLEN; t=p//XLEN; k=t%ZLEN; j=t//ZLEN; return i,j,k   # i,k,j
    return [("A",A),("B",B),("C",C),("D",D),("E",E),("F",F)]

def axis_layout(idx, ntile, nsplit):
    """For split index `idx` along one axis, return (g0, crop, n_used):
       g0     = global index where this tile's (possibly cropped) data starts
       crop   = how many leading planes to drop (shared-boundary handling)
       n_used = number of planes actually placed
    This tiles the global axis with NO gaps and NO overlaps."""
    if SHARED_PLANES:
        ncell = ntile - 1
        crop  = 0 if idx == 0 else 1          # drop duplicated lower node
        return idx * ncell + crop, crop, ntile - crop
    else:
        return idx * ntile, 0, ntile          # cell-centered: contiguous, no crop

def global_size(ntile, nsplit):
    return nsplit * (ntile - 1) + 1 if SHARED_PLANES else nsplit * ntile

# --------------------------- File discovery (rank 0) ---------------------------
if rank == 0:
    all_files = sorted(glob.glob(os.path.join(dir_data, "proc*.hdf")))
    if len(all_files) == 0:
        raise RuntimeError(f"No proc*.hdf files in: {dir_data}")
    if len(all_files) != num_expected_files:
        print(f"WARNING: expected {num_expected_files} files, found {len(all_files)}.")
else:
    all_files = None
all_files = comm.bcast(all_files, root=0)

local_files = all_files[rank::size]           # chunked distribution across ranks

# --------------------------- Probe tile shape (rank 0 -> all) ---------------------------
if rank == 0:
    with h5py.File(all_files[0], "r") as f:
        tile_shape = get_Bx(f, time_cycle).shape
else:
    tile_shape = None
nx_tile, ny_tile, nz_tile = comm.bcast(tile_shape, root=0)

nx_global = global_size(nx_tile, XLEN)
ny_global = global_size(ny_tile, YLEN)
nz_global = global_size(nz_tile, ZLEN)

###! Slice positions (global indices). Default = domain mid-planes.
###! ⚠️ For a double-Harris sheet the current layers are NOT at the centre;
###! set these to the sheet/X-line locations if you want the reconnection region.
IX = nx_global // 2     # YZ plane lives at this x
IY = ny_global // 2     # ZX plane lives at this y
IZ = nz_global // 2     # XY plane lives at this z

# --------------------------- Choose mapping (rank 0, 3D occupancy) ---------------------------
if rank == 0:
    proc_ids = [proc_id_from_filename(fp) for fp in all_files]
    best_name, best_score = None, None
    for name, fn in mapping_candidates():
        Occ = np.zeros((nx_global, ny_global, nz_global), dtype=np.int32)
        bad = False
        for pid in proc_ids:
            i, j, k = fn(pid)
            if not (0 <= i < XLEN and 0 <= j < YLEN and 0 <= k < ZLEN):
                bad = True; break
            gx, _, nxu = axis_layout(i, nx_tile, XLEN)
            gy, _, nyu = axis_layout(j, ny_tile, YLEN)
            gz, _, nzu = axis_layout(k, nz_tile, ZLEN)
            Occ[gx:gx+nxu, gy:gy+nyu, gz:gz+nzu] += 1
        if bad:
            continue
        gaps     = int(np.count_nonzero(Occ == 0))
        overlaps = int(np.count_nonzero(Occ > 1))
        score    = gaps * 10 + overlaps * 2 + max(0, int(Occ.max()) - 1) * 1000
        if best_score is None or score < best_score:
            best_name, best_score = name, score
    if best_name is None:
        raise RuntimeError("No valid proc->(i,j,k) mapping found.")
else:
    best_name = None
best_name = comm.bcast(best_name, root=0)
rank_to_ijk = dict(mapping_candidates())[best_name]

# --------------------------- Local plane buffers ---------------------------
# XY[x,y] @ z=IZ ; YZ[y,z] @ x=IX ; ZX[x,z] @ y=IY
loc_XY = np.zeros((nx_global, ny_global), dtype=np.float64)
loc_YZ = np.zeros((ny_global, nz_global), dtype=np.float64)
loc_ZX = np.zeros((nx_global, nz_global), dtype=np.float64)
occ_XY = np.zeros((nx_global, ny_global), dtype=np.int32)
occ_YZ = np.zeros((ny_global, nz_global), dtype=np.int32)
occ_ZX = np.zeros((nx_global, nz_global), dtype=np.int32)

# --------------------------- Assemble (each rank) ---------------------------
for fp in local_files:
    i, j, k = rank_to_ijk(proc_id_from_filename(fp))

    gx, cx, nxu = axis_layout(i, nx_tile, XLEN)
    gy, cy, nyu = axis_layout(j, ny_tile, YLEN)
    gz, cz, nzu = axis_layout(k, nz_tile, ZLEN)

    # Does this tile touch any of the three target planes? If not, skip the read.
    hit_XY = gz <= IZ < gz + nzu
    hit_YZ = gx <= IX < gx + nxu
    hit_ZX = gy <= IY < gy + nyu
    if not (hit_XY or hit_YZ or hit_ZX):
        continue

    with h5py.File(fp, "r") as f:
        tile = get_Bx(f, time_cycle)[cx:, cy:, cz:]   # crop shared lower planes

    if hit_XY:
        slab = tile[:, :, IZ - gz]                    # (nxu, nyu)
        loc_XY[gx:gx+nxu, gy:gy+nyu] = slab
        occ_XY[gx:gx+nxu, gy:gy+nyu] += 1
    if hit_YZ:
        slab = tile[IX - gx, :, :]                    # (nyu, nzu)
        loc_YZ[gy:gy+nyu, gz:gz+nzu] = slab
        occ_YZ[gy:gy+nyu, gz:gz+nzu] += 1
    if hit_ZX:
        slab = tile[:, IY - gy, :]                    # (nxu, nzu)
        loc_ZX[gx:gx+nxu, gz:gz+nzu] = slab
        occ_ZX[gx:gx+nxu, gz:gz+nzu] += 1

# --------------------------- Reduce to root (SUM: tiles are non-overlapping) ---------------------------
XY = YZ = ZX = oXY = oYZ = oZX = None
if rank == 0:
    XY  = np.zeros_like(loc_XY); YZ  = np.zeros_like(loc_YZ); ZX  = np.zeros_like(loc_ZX)
    oXY = np.zeros_like(occ_XY); oYZ = np.zeros_like(occ_YZ); oZX = np.zeros_like(occ_ZX)

comm.Reduce(loc_XY, XY, op=MPI.SUM, root=0)
comm.Reduce(loc_YZ, YZ, op=MPI.SUM, root=0)
comm.Reduce(loc_ZX, ZX, op=MPI.SUM, root=0)
comm.Reduce(occ_XY, oXY, op=MPI.SUM, root=0)
comm.Reduce(occ_YZ, oYZ, op=MPI.SUM, root=0)
comm.Reduce(occ_ZX, oZX, op=MPI.SUM, root=0)


# --------------------------- Plot (root) ---------------------------
if rank == 0:
    for nm, o in (("XY", oXY), ("ZY", oYZ), ("ZX", oZX)):
        if o.min() == 0:
            print(f"WARNING [{nm}]: gaps remain (Occ.min=0) — check mapping / SHARED_PLANES.")
        if o.max() > 1:
            print(f"WARNING [{nm}]: overlaps remain (Occ.max={int(o.max())}) — check SHARED_PLANES.")

    def show(ax, data, title, xlabel, ylabel, extent):
        # data passed as [horizontal, vertical]; transpose so rows=vertical for imshow.
        # extent = (h_min, h_max, v_min, v_max) matches the horizontal/vertical axes.
        vmax = np.max(np.abs(data)) or 1.0          # symmetric diverging scale for signed Bx
        im = ax.imshow(data.T, origin="lower", cmap="seismic", aspect="auto",
                       vmin=-vmax, vmax=vmax, extent=extent)
        ax.set_title(title, fontsize=16)
        ax.set_xlabel(xlabel, fontsize=14); ax.set_ylabel(ylabel, fontsize=14)
        plt.colorbar(im, ax=ax)

    # Physical slice positions for titles (map global index -> coordinate)
    x_at = xmin + (xmax - xmin) * IX / (nx_global - 1)
    y_at = ymin + (ymax - ymin) * IY / (ny_global - 1)
    z_at = zmin + (zmax - zmin) * IZ / (nz_global - 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), dpi=250)

    # XY: horizontal=X, vertical=Y. Buffer XY is [x,y] -> show() transposes to [y,x]. OK.
    show(axes[0], XY, f"Bx  XY  (z={z_at:.2f})", "X", "Y",
         extent=(xmin, xmax, ymin, ymax))

    # ZY: horizontal=Z, vertical=Y. Buffer YZ is [y,z]; show() transposes [y,z]->[z,y],
    # giving rows=y (vertical), cols=z (horizontal) — exactly Z-horizontal, Y-vertical.
    show(axes[1], YZ, f"Bx  ZY  (x={x_at:.2f})", "Z", "Y",
         extent=(zmin, zmax, ymin, ymax))

    # ZX: horizontal=X, vertical=Z. Buffer ZX is [x,z] -> show() transposes to [z,x]. OK.
    show(axes[2], ZX, f"Bx  ZX  (y={y_at:.2f})", "X", "Z",
         extent=(xmin, xmax, zmin, zmax))

    fig.tight_layout()
    fig.savefig(os.path.join(dir_data, f"Bx_{time_cycle}.png"))
    plt.close(fig)