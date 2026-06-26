
"""
Assemble global Bx/By/Bz (2D slice k=0) from per-proc iPIC3D HDF tiles,
handling shared boundary/ghost layers to avoid double counting seams.

Assumptions (consistent with B*_data shape (21,21,2)):
- The first two dimensions are nodal (or nodal-like) with one extra point:
    nx_tile = nx_cell + 1, ny_tile = ny_cell + 1
  so nx_cell = nx_tile - 1, ny_cell = ny_tile - 1.
- Adjacent ranks share one boundary plane. We drop the "lower" shared plane
  for ranks that are not on the global low edge, i.e. drop index 0 in that
  direction when i>0 and/or j>0.

If your output instead includes true ghost layers of width >1, you can set
GHOST=1/2/... and crop [GHOST:-GHOST] etc. See section (B).
"""

import numpy as np
from mpi4py import MPI
import os, glob, h5py, argparse
import matplotlib.pyplot as plt

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

###* =================================================================== *###

###* Argument parser
parser = argparse.ArgumentParser(description= "Plot 2D charge density from iPIC3D output data")

parser.add_argument("dir_data",   type=str, help="Directory where proc.hdf files are stored, e.g., './data_reconnection/'")
parser.add_argument("time_cycle", type=str, help="Time cycle to plot, e.g., 'cycle_100'")

parser.add_argument("xlen", type=int, help="Number of MPI processes along X (must match simulation)")
parser.add_argument("ylen", type=int, help="Number of MPI processes along Y (must match simulation)")
parser.add_argument("zlen", type=int, help="Number of MPI processes along Z (must match simulation)")

args = parser.parse_args()

###* Directory where proc.hdf files are saved (plots are saved to the same directory)
dir_data = args.dir_data

###* Time cycle when data is to plotted 
time_cycle = args.time_cycle

###! MPI topology (must match simulation)
XLEN, YLEN, ZLEN = args.xlen, args.ylen, args.zlen
num_expected_files = XLEN * YLEN * ZLEN

# ----------------------------- Helpers -----------------------------
def proc_id_from_filename(fp: str) -> int:
    base = os.path.basename(fp)
    # expects procNNN.hdf
    return int(base.replace("proc", "").replace(".hdf", ""))

def get_dataset(f: h5py.File, field: str, cycle: str) -> np.ndarray:
    # expects fields/<field>/<cycle>
    return np.array(f[f"fields/{field}/{cycle}"])

def mapping_candidates():
    """
    Return common proc_id -> (i,j,k) orderings.
    The right one depends on how iPIC3D numbers ranks in the output filename.

    Candidates:
      A: id = (i*YLEN + j)*ZLEN + k    (k fastest, then j, then i)
      B: id = (i*ZLEN + k)*YLEN + j    (j fastest, then k, then i)
      C: id = (j*XLEN + i)*ZLEN + k    (k fastest, then i, then j)  [swap i/j]
      D: id = (k*YLEN + j)*XLEN + i    (i fastest, then j, then k)
      E: id = (k*XLEN + i)*YLEN + j    (j fastest, then i, then k)
      F: id = (j*ZLEN + k)*XLEN + i    (i fastest, then k, then j)
    """
    def A(pid):  # (i,j,k) with k fastest
        k = pid % ZLEN
        t = pid // ZLEN
        j = t % YLEN
        i = t // YLEN
        return i, j, k

    def B(pid):  # j fastest
        j = pid % YLEN
        t = pid // YLEN
        k = t % ZLEN
        i = t // ZLEN
        return i, j, k

    def C(pid):  # swap i/j relative to A
        k = pid % ZLEN
        t = pid // ZLEN
        i = t % XLEN
        j = t // XLEN
        return i, j, k

    def D(pid):  # i fastest
        i = pid % XLEN
        t = pid // XLEN
        j = t % YLEN
        k = t // YLEN
        return i, j, k

    def E(pid):  # j fastest, then i, then k
        j = pid % YLEN
        t = pid // YLEN
        i = t % XLEN
        k = t // XLEN
        return i, j, k

    def F(pid):  # i fastest, then k, then j
        i = pid % XLEN
        t = pid // XLEN
        k = t % ZLEN
        j = t // ZLEN
        return i, j, k

    return [("A", A), ("B", B), ("C", C), ("D", D), ("E", E), ("F", F)]

def score_occupancy(Occ: np.ndarray):
    gaps = int(np.count_nonzero(Occ == 0))
    overlaps = int(np.count_nonzero(Occ > 1))
    maxv = int(Occ.max())
    # weighted score: gaps matter most, then overlaps, then peak overlap
    score = gaps * 10 + overlaps * 2 + max(0, maxv - 1) * 1000
    return score, gaps, overlaps, maxv

# ----------------------------- File discovery -----------------------------
if rank == 0:
    all_hdf_files = sorted(glob.glob(os.path.join(dir_data, "proc*.hdf")))
    if len(all_hdf_files) == 0:
        raise RuntimeError(f"No proc*.hdf files found in: {dir_data}")
    if len(all_hdf_files) != num_expected_files:
        print(f"WARNING: expected {num_expected_files} files but found {len(all_hdf_files)}.")
else:
    all_hdf_files = None

all_hdf_files = comm.bcast(all_hdf_files, root=0)

# Chunk distribution
local_files = all_hdf_files[rank::size]
if rank == 0:
    print(f"Processing {len(all_hdf_files)} files with {size} MPI tasks")

# ----------------------------- Probe tile shape -----------------------------
tile_shape = None
if local_files:
    with h5py.File(local_files[0], "r") as f:
        tile_shape = get_dataset(f, "Bx", time_cycle).shape  # (nx, ny, nzslab)

tile_shape_all = comm.gather(tile_shape, root=0)

if rank == 0:
    tile_shape = next(s for s in tile_shape_all if s is not None)
    nx_tile, ny_tile, nz_tile = tile_shape
    print("Detected tile shape:", tile_shape)
else:
    nx_tile = ny_tile = nz_tile = None

nx_tile, ny_tile, nz_tile = comm.bcast((nx_tile, ny_tile, nz_tile), root=0)

# Select a k-slab to plot (use 0 by default; you can change here if desired)
KSLAB = 0
if not (0 <= KSLAB < nz_tile):
    raise ValueError(f"KSLAB={KSLAB} out of bounds for nz_tile={nz_tile}")

# ----------------------------- Interpret tile layout -----------------------------
# Nodal-like with shared boundary planes:
nx_cell = nx_tile - 1
ny_cell = ny_tile - 1

# Global nodal sizes:
nx_global = XLEN * nx_cell + 1
ny_global = YLEN * ny_cell + 1

# ----------------------------- Choose best mapping (rank 0) -----------------------------
best_name = None
best_map = None
best_stats = None

if rank == 0:
    # Use filenames to compute occupancy only (no HDF reads needed here)
    proc_ids = [proc_id_from_filename(fp) for fp in all_hdf_files]

    for name, fn in mapping_candidates():
        Occ = np.zeros((nx_global, ny_global), dtype=np.int32)

        for pid in proc_ids:
            i, j, k = fn(pid)

            # Keep only a single z-slab's worth of files for mapping validation
            # because we are assembling a 2D plot (k fixed). If ZLEN>1, we only
            # consider those with k==0 (same as KSLAB selection).
            if k != 0:
                continue

            # Sanity: reject invalid indices
            if not (0 <= i < XLEN and 0 <= j < YLEN):
                # This mapping is incompatible
                Occ[:] = -1
                break

            # Crop shared boundary planes to avoid double counting
            xs = 0 if i == 0 else 1
            ys = 0 if j == 0 else 1

            nx_use = nx_tile - xs
            ny_use = ny_tile - ys

            x0 = i * nx_cell
            y0 = j * ny_cell

            Occ[x0:x0 + nx_use, y0:y0 + ny_use] += 1

        if Occ[0, 0] < 0:
            continue

        score, gaps, overlaps, maxv = score_occupancy(Occ)
        if best_stats is None or score < best_stats[0]:
            best_name = name
            best_map = fn
            best_stats = (score, gaps, overlaps, maxv)

    if best_map is None:
        raise RuntimeError("Could not determine a valid proc->(i,j,k) mapping.")

    print(f"Selected mapping: {best_name} with score={best_stats[0]}, gaps={best_stats[1]}, overlaps={best_stats[2]}, Occ.max={best_stats[3]}")

best_name = comm.bcast(best_name, root=0)

# Broadcast chosen mapping name; re-instantiate the same mapping on all ranks
maps = {name: fn for name, fn in mapping_candidates()}
rank_to_ijk = maps[best_name]

# ----------------------------- Assemble on each rank -----------------------------
local_Bx = np.zeros((nx_global, ny_global), dtype=np.float64)
local_By = np.zeros((nx_global, ny_global), dtype=np.float64)
local_Bz = np.zeros((nx_global, ny_global), dtype=np.float64)

local_occ = np.zeros((nx_global, ny_global), dtype=np.int32)

for fp in local_files:
    pid = proc_id_from_filename(fp)
    i, j, k = rank_to_ijk(pid)

    # Only assemble one z-slab for the 2D plot
    if k != 0:
        continue

    with h5py.File(fp, "r") as f:
        Bx_data = get_dataset(f, "Bx", time_cycle)[:, :, KSLAB]
        By_data = get_dataset(f, "By", time_cycle)[:, :, KSLAB]
        Bz_data = get_dataset(f, "Bz", time_cycle)[:, :, KSLAB]

    # Crop shared boundary planes:
    xs = 0 if i == 0 else 1
    ys = 0 if j == 0 else 1

    Bx_tile = Bx_data[xs:, ys:]
    By_tile = By_data[xs:, ys:]
    Bz_tile = Bz_data[xs:, ys:]

    x0 = i * nx_cell
    y0 = j * ny_cell

    nx_use, ny_use = Bx_tile.shape

    local_Bx[x0:x0 + nx_use, y0:y0 + ny_use] = Bx_tile
    local_By[x0:x0 + nx_use, y0:y0 + ny_use] = By_tile
    local_Bz[x0:x0 + nx_use, y0:y0 + ny_use] = Bz_tile

    local_occ[x0:x0 + nx_use, y0:y0 + ny_use] += 1

# ----------------------------- Reduce to root -----------------------------
Bx = By = Bz = Occ = None
if rank == 0:
    Bx  = np.zeros((nx_global, ny_global), dtype=np.float64)
    By  = np.zeros((nx_global, ny_global), dtype=np.float64)
    Bz  = np.zeros((nx_global, ny_global), dtype=np.float64)
    Occ = np.zeros((nx_global, ny_global), dtype=np.int32)

comm.Reduce(local_Bx, Bx, op=MPI.MAX, root=0)
comm.Reduce(local_By, By, op=MPI.MAX, root=0)
comm.Reduce(local_Bz, Bz, op=MPI.MAX, root=0)
comm.Reduce(local_occ, Occ, op=MPI.SUM, root=0)

# ----------------------------- Plot on root -----------------------------
if rank == 0:
    print("Occ min/max:", int(Occ.min()), int(Occ.max()))
    if Occ.min() == 0:
        print("WARNING: gaps remain. This usually indicates Z-slab selection mismatch or an unexpected dump layout.")
    if Occ.max() > 1:
        print("WARNING: overlaps remain. This usually indicates more than one shared/ghost layer or inconsistent tile size assumptions.")

    print("Bx min/max:", float(Bx.min()), float(Bx.max()))
    print("By min/max:", float(By.min()), float(By.max()))
    print("Bz min/max:", float(Bz.min()), float(Bz.max()))

    # 2D plots
    fig = plt.figure(figsize=(14, 4), dpi=250)

    plt.subplot(1, 3, 1)
    plt.imshow(Bx.T, origin="lower", cmap="seismic", aspect="auto")
    plt.xlabel("X", fontsize=16); plt.ylabel("Y", fontsize=16)
    plt.title("Bx", fontsize=18); plt.colorbar()

    plt.subplot(1, 3, 2)
    plt.imshow(By.T, origin="lower", cmap="seismic", aspect="auto")
    plt.xlabel("X", fontsize=16); plt.ylabel("Y", fontsize=16)
    plt.title("By", fontsize=18); plt.colorbar()

    plt.subplot(1, 3, 3)
    plt.imshow(Bz.T, origin="lower", cmap="seismic", aspect="auto")
    plt.xlabel("X", fontsize=16); plt.ylabel("Y", fontsize=16)
    plt.title("Bz", fontsize=18); plt.colorbar()

    fig.tight_layout()
    plt.savefig(os.path.join(dir_data, f"B_field_{time_cycle}.png"))
    plt.close()

    # 1D cuts: along X for multiple Y (use quarters)
    ycuts = [ny_global // 4, ny_global // 2, (3 * ny_global) // 4]
    x = np.arange(nx_global)

    plt.figure(figsize=(7, 4), dpi=200)
    for y0 in ycuts:
        plt.plot(x, Bx[:, y0], lw=2, label=f"Y={y0}")
    plt.xlabel("X index"); plt.ylabel("Bx")
    plt.title(f"Bx 1D cuts ({time_cycle})")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(dir_data, f"Bx_cutX_{time_cycle}.png"))
    plt.close()

    # 1D cuts: along Y for multiple X
    xcuts = [nx_global // 4, nx_global // 2, (3 * nx_global) // 4]
    y = np.arange(ny_global)

    plt.figure(figsize=(7, 4), dpi=200)
    for x0 in xcuts:
        plt.plot(y, By[x0, :], lw=2, label=f"X={x0}")
    plt.xlabel("Y index"); plt.ylabel("By")
    plt.title(f"By 1D cuts ({time_cycle})")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(dir_data, f"By_cutY_{time_cycle}.png"))
    plt.close()