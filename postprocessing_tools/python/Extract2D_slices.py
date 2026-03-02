"""
Created on Wed Feb 25 14:13 2026

@author: Pranab JD, ChatGPT

Description: Assemble and SAVE an arbitrary 2D slice of FIELDS from per-proc iPIC3D HDF tiles. 
             Writes one ASCII .txt per time step (rank 0 only).

Usage:  # Directory where proc.hdf files are saved
        folder=/scratch/project_465002078/pranab/RMR/sigma_1/domain_128_256_128/Bz_0

        xlen=64; ylen=2; zlen=64

        quantity=By
        axis=y
        index=192
        outdir="${folder}/slices_${quantity}_${axis}_CS1"

        time_cycle=cycle_500
        
        srun python3 ../postprocessing_tools/python/Extract2D_slices.py \
            "$folder" \
            "$time_cycle" \
            "$xlen" "$ylen" "$zlen" \
            "$quantity" "$axis" "$index" \
            "${outdir}"
"""

import os, glob, argparse
import numpy as np
import h5py
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# ----------------------------- Helpers -----------------------------
def proc_id_from_filename(fp: str) -> int:
    base = os.path.basename(fp)
    return int(base.replace("proc", "").replace(".hdf", ""))

def dset_path(quantity: str, cycle: str, species: int | None = None) -> str:
    if kind == "fields":
        return f"fields/{quantity}/{cycle}"
    else:
        if species is None:
            raise ValueError("species required for moments")
        return f"moments/species_{species}/{quantity}/{cycle}"

def read_slab_2d(f: h5py.File, quantity: str, cycle: str, axis: str,
                loc_i: int | None = None, loc_j: int | None = None, loc_k: int | None = None,
                species_sel=None) -> np.ndarray:
    """
    Return a 2D slab (float64) for either fields or moments (summed over species_sel).
    axis selects which index is fixed.
    """
    if kind == "fields":
        d = f[dset_path(quantity, cycle)]
        if axis == "z":
            return np.array(d[:, :, loc_k], dtype=np.float64)
        elif axis == "y":
            return np.array(d[:, loc_j, :], dtype=np.float64)
        else:  # axis == "x"
            return np.array(d[loc_i, :, :], dtype=np.float64)

    else:  # moments
        slab_sum = None
        for s in species_sel:
            d = f[dset_path(quantity, cycle, species=s)]
            if axis == "z":
                arr = np.array(d[:, :, loc_k], dtype=np.float64)
            elif axis == "y":
                arr = np.array(d[:, loc_j, :], dtype=np.float64)
            else:
                arr = np.array(d[loc_i, :, :], dtype=np.float64)
            slab_sum = arr if slab_sum is None else slab_sum + arr
        return slab_sum

def mapping_candidates(XLEN, YLEN, ZLEN):
    def A(pid):  # id = (i*YLEN + j)*ZLEN + k
        k = pid % ZLEN
        t = pid // ZLEN
        j = t % YLEN
        i = t // YLEN
        return i, j, k

    def B(pid):  # id = (i*ZLEN + k)*YLEN + j
        j = pid % YLEN
        t = pid // YLEN
        k = t % ZLEN
        i = t // ZLEN
        return i, j, k

    def C(pid):  # swap i/j vs A
        k = pid % ZLEN
        t = pid // ZLEN
        i = t % XLEN
        j = t // XLEN
        return i, j, k

    def D(pid):  # id = (k*YLEN + j)*XLEN + i
        i = pid % XLEN
        t = pid // XLEN
        j = t % YLEN
        k = t // YLEN
        return i, j, k

    def E(pid):  # id = (k*XLEN + i)*YLEN + j
        j = pid % YLEN
        t = pid // YLEN
        i = t % XLEN
        k = t // XLEN
        return i, j, k

    def F(pid):  # id = (j*ZLEN + k)*XLEN + i
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
    score = gaps * 10 + overlaps * 2 + max(0, maxv - 1) * 1000
    return score, gaps, overlaps, maxv

# ----------------------------- Args -----------------------------
parser = argparse.ArgumentParser(description="Save 2D slice of iPIC3D fields from proc*.hdf as .txt")

FIELD_Q = {"Ex","Ey","Ez","Bx","By","Bz"}
MOMENT_Q = {"rho","Jx","Jy","Jz"}

parser.add_argument("dir_data",   type=str, help="Directory containing proc*.hdf")
parser.add_argument("time_cycle", type=str, help="Cycle group name, e.g. cycle_19500")

parser.add_argument("xlen", type=int, help="Simulation XLEN")
parser.add_argument("ylen", type=int, help="Simulation YLEN")
parser.add_argument("zlen", type=int, help="Simulation ZLEN")

parser.add_argument("quantity", type=str, help="Field quantity, e.g. Ex, Ey, Ez, Bx, By, Bz")
parser.add_argument("--species", default="all", help="For moments only: 'all' or comma list like '0,1,3'")
parser.add_argument("axis", type=str, choices=["x","y","z"], help="Slice axis")
parser.add_argument("index", type=int, help="Global nodal index along axis")

parser.add_argument("out_dir", type=str, help="Output directory for .txt files")

parser.add_argument("--floatfmt", type=str, default="%.8e", help="np.savetxt float format")
parser.add_argument("--transpose", action="store_true",
                    help="Write transposed array (matches your imshow(out_plane.T) convention).")

args = parser.parse_args()

quantity = args.quantity

if quantity in FIELD_Q:
    kind = "fields"
elif quantity in MOMENT_Q:
    kind = "moments"
else:
    raise ValueError(
        f"Unknown quantity '{quantity}'. "
        f"Fields: {sorted(FIELD_Q)} | Moments: {sorted(MOMENT_Q)}")

dir_data   = args.dir_data
time_cycle = args.time_cycle
XLEN, YLEN, ZLEN = args.xlen, args.ylen, args.zlen
axis    = args.axis.lower()
slice_g = args.index
quantity = args.quantity

# ----------------------------- Discover files -----------------------------
if rank == 0:
    all_files = sorted(glob.glob(os.path.join(dir_data, "proc*.hdf")))
    if not all_files:
        raise RuntimeError(f"No proc*.hdf found in {dir_data}")
else:
    all_files = None

all_files = comm.bcast(all_files, root=0)
local_files = all_files[rank::size]

# ----------------------------- Probe tile shape -----------------------------
tile_shape = None
if local_files:
    with h5py.File(local_files[0], "r") as f:

        if kind == "fields":
            path = dset_path(quantity, time_cycle)
            if path not in f:
                raise RuntimeError(f"Dataset not found: /{path}")
            tile_shape = tuple(f[path].shape)
            species_sel = None

        else:  # moments
            # discover available species
            species_avail = []
            for k in f["moments"].keys():
                if k.startswith("species_"):
                    species_avail.append(int(k.split("_")[1]))
            species_avail = sorted(species_avail)

            if not species_avail:
                raise RuntimeError("No species found under /moments")

            if args.species.strip().lower() == "all":
                species_sel = species_avail
            else:
                species_sel = [int(x) for x in args.species.split(",") if x.strip() != ""]
                missing = sorted(set(species_sel) - set(species_avail))
                if missing:
                    raise RuntimeError(f"Requested species not present: {missing}. Available: {species_avail}")

            s0 = species_sel[0]
            path = dset_path(quantity, time_cycle, species=s0)
            if path not in f:
                raise RuntimeError(f"Dataset not found: /{path}")
            tile_shape = tuple(f[path].shape)

tile_shape_all = comm.gather(tile_shape, root=0)

if rank == 0:
    tile_shape = next(s for s in tile_shape_all if s is not None)
else:
    tile_shape = None

tile_shape = comm.bcast(tile_shape, root=0)
species_sel = comm.bcast(species_sel if rank == 0 else None, root=0)
nx_tile, ny_tile, nz_tile = tile_shape

# ----------------------------- Global sizing -----------------------------
nx_cell = nx_tile - 1
ny_cell = ny_tile - 1
nz_cell = nz_tile - 1

nx_global = XLEN * nx_cell + 1
ny_global = YLEN * ny_cell + 1
nz_global = ZLEN * nz_cell + 1

global_len = {"x": nx_global, "y": ny_global, "z": nz_global}[axis]
if not (0 <= slice_g < global_len):
    raise ValueError(f"slice index {slice_g} out of range for axis {axis} (0..{global_len-1})")

if axis == "z":
    plane_shape = (nx_global, ny_global)
elif axis == "y":
    plane_shape = (nx_global, nz_global)
else:
    plane_shape = (ny_global, nz_global)

# ----------------------------- Infer proc->(i,j,k) mapping -----------------------------
best_name = None
if rank == 0:
    proc_ids = [proc_id_from_filename(fp) for fp in all_files]
    best_stats = None

    for name, fn in mapping_candidates(XLEN, YLEN, ZLEN):
        Occ = np.zeros(plane_shape, dtype=np.int32)
        bad = False

        for pid in proc_ids:
            i, j, k = fn(pid)
            if not (0 <= i < XLEN and 0 <= j < YLEN and 0 <= k < ZLEN):
                bad = True
                break

            xs = 0 if i == 0 else 1
            ys = 0 if j == 0 else 1
            zs = 0 if k == 0 else 1

            x0 = i * nx_cell
            y0 = j * ny_cell
            z0 = k * nz_cell

            gx0 = x0 + xs
            gy0 = y0 + ys
            gz0 = z0 + zs

            nx_use = nx_tile - xs
            ny_use = ny_tile - ys
            nz_use = nz_tile - zs

            if axis == "z":
                if not (gz0 <= slice_g < gz0 + nz_use):
                    continue
                Occ[gx0:gx0+nx_use, gy0:gy0+ny_use] += 1
            elif axis == "y":
                if not (gy0 <= slice_g < gy0 + ny_use):
                    continue
                Occ[gx0:gx0+nx_use, gz0:gz0+nz_use] += 1
            else:
                if not (gx0 <= slice_g < gx0 + nx_use):
                    continue
                Occ[gy0:gy0+ny_use, gz0:gz0+nz_use] += 1

        if bad:
            continue

        score, gaps, overlaps, maxv = score_occupancy(Occ)
        if best_stats is None or score < best_stats[0]:
            best_stats = (score, gaps, overlaps, maxv)
            best_name = name

    if best_name is None:
        raise RuntimeError("Could not infer proc->(i,j,k) mapping from filenames.")

best_name = comm.bcast(best_name, root=0)
maps = {n: fn for n, fn in mapping_candidates(XLEN, YLEN, ZLEN)}
pid_to_ijk = maps[best_name]

# ----------------------------- Assemble requested plane -----------------------------
local_plane = np.zeros(plane_shape, dtype=np.float64)
local_occ   = np.zeros(plane_shape, dtype=np.int32)

for fp in local_files:
    pid = proc_id_from_filename(fp)
    i, j, k = pid_to_ijk(pid)

    xs = 0 if i == 0 else 1
    ys = 0 if j == 0 else 1
    zs = 0 if k == 0 else 1

    x0 = i * nx_cell
    y0 = j * ny_cell
    z0 = k * nz_cell

    gx0 = x0 + xs
    gy0 = y0 + ys
    gz0 = z0 + zs

    nx_use = nx_tile - xs
    ny_use = ny_tile - ys
    nz_use = nz_tile - zs

    if axis == "z":
        if not (gz0 <= slice_g < gz0 + nz_use):
            continue
        loc_k_orig = (slice_g - gz0) + zs

        with h5py.File(fp, "r") as f:
            slab = read_slab_2d(f, quantity, time_cycle, axis="z",
                                loc_k=loc_k_orig, species_sel=species_sel)

        slab = slab[xs:, ys:]  # -> (nx_use, ny_use)
        local_plane[gx0:gx0+nx_use, gy0:gy0+ny_use] += slab
        local_occ[gx0:gx0+nx_use, gy0:gy0+ny_use]   += 1

    elif axis == "y":
        if not (gy0 <= slice_g < gy0 + ny_use):
            continue
        loc_j_orig = (slice_g - gy0) + ys

        with h5py.File(fp, "r") as f:
            slab = read_slab_2d(f, quantity, time_cycle, axis="y",
                                loc_j=loc_j_orig, species_sel=species_sel)

        slab = slab[xs:, zs:]  # -> (nx_use, nz_use)
        local_plane[gx0:gx0+nx_use, gz0:gz0+nz_use] += slab
        local_occ[gx0:gx0+nx_use, gz0:gz0+nz_use]   += 1

    else:  # axis == "x"
        if not (gx0 <= slice_g < gx0 + nx_use):
            continue
        loc_i_orig = (slice_g - gx0) + xs

        with h5py.File(fp, "r") as f:
            slab = read_slab_2d(f, quantity, time_cycle, axis="x",
                                loc_i=loc_i_orig, species_sel=species_sel)

        slab = slab[ys:, zs:]  # -> (ny_use, nz_use)
        local_plane[gy0:gy0+ny_use, gz0:gz0+nz_use] += slab
        local_occ[gy0:gy0+ny_use, gz0:gz0+nz_use]   += 1

# ----------------------------- Reduce to root -----------------------------
plane = None
occ = None
if rank == 0:
    plane = np.zeros(plane_shape, dtype=np.float64)
    occ   = np.zeros(plane_shape, dtype=np.int32)

comm.Reduce(local_plane, plane, op=MPI.SUM, root=0)
comm.Reduce(local_occ,   occ,   op=MPI.SUM, root=0)

# ----------------------------- Save on root -----------------------------
if rank == 0:
    omin, omax = int(occ.min()), int(occ.max())
    if omin == 0:
        print("WARNING: gaps remain (occ==0). Check slice index, mapping, or ghost-layer assumptions.")
    if omax > 1:
        print("WARNING: overlaps (occ>1). Might indicate >1 ghost layer or different centering.")

    out_plane = np.zeros_like(plane)
    mask = (occ > 0)
    out_plane[mask] = plane[mask] / occ[mask]

    os.makedirs(args.out_dir, exist_ok=True)
    out_txt = os.path.join(args.out_dir, f"{quantity}_slice_{axis}{slice_g}_{time_cycle}.txt")

    A = out_plane.T if args.transpose else out_plane

    header = (
        f"{quantity} 2D slice from iPIC3D per-proc tiles\n"
        f"dir={dir_data}\n"
        f"time_cycle={time_cycle}\n"
        f"axis={axis} index={slice_g}\n"
        f"shape_written={A.shape} (transpose={args.transpose})\n"
        f"occ_min={omin} occ_max={omax}\n"
    )

    np.savetxt(out_txt, A, fmt=args.floatfmt, header=header)
    print("Wrote", out_txt)