"""
Created on Sun Mar 22 2026

@author: Pranab JD, ChatGPT

Description: Define CS width as 0.5 times the maximum of Jz (for each sheet individually).
             The width of the 2 current sheets are written in terms of Y-indices to 
             "CS_bounds.txt". These indices can be used to extract data along the CSs.

Usage:  srun python3 Extract_3D_CS.py \
            <dir_data> <outdir> <quantity> \
            <t_start> <t_end> <t_step> \
            <xlen> <ylen> <zlen> \
            [--species all] [--cs_threshold 0.5] [--dtype float32]
"""

import os, re, glob
import numpy as np
import argparse, h5py
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

###! ============================================================
###! Helper Functions
###! ============================================================

def proc_id_from_filename(fp: str) -> int:
    base = os.path.basename(fp)
    return int(base.replace("proc", "").replace(".hdf", ""))

def mapping_candidates(XLEN, YLEN, ZLEN):
    def A(pid):
        k = pid % ZLEN
        t = pid // ZLEN
        j = t % YLEN
        i = t // YLEN
        return i, j, k

    def B(pid):
        j = pid % YLEN
        t = pid // YLEN
        k = t % ZLEN
        i = t // ZLEN
        return i, j, k

    def C(pid):
        k = pid % ZLEN
        t = pid // ZLEN
        i = t % XLEN
        j = t // XLEN
        return i, j, k

    def D(pid):
        i = pid % XLEN
        t = pid // XLEN
        j = t % YLEN
        k = t // YLEN
        return i, j, k

    def E(pid):
        j = pid % YLEN
        t = pid // YLEN
        i = t % XLEN
        k = t // XLEN
        return i, j, k

    def F(pid):
        i = pid % XLEN
        t = pid // XLEN
        k = t % ZLEN
        j = t // ZLEN
        return i, j, k

    return [("A", A), ("B", B), ("C", C), ("D", D), ("E", E), ("F", F)]

def score_occupancy(occ: np.ndarray):
    gaps = int(np.count_nonzero(occ == 0))
    overlaps = int(np.count_nonzero(occ > 1))
    maxv = int(occ.max()) if occ.size else 0
    score = gaps * 10 + overlaps * 2 + max(0, maxv - 1) * 1000
    return score, gaps, overlaps, maxv

def discover_species(f: h5py.File):
    """Return sorted list of species indices under /moments/species_*."""
    if "moments" not in f:
        return []
    out = []
    for k in f["moments"].keys():
        m = re.match(r"species_(\d+)$", k)
        if m:
            out.append(int(m.group(1)))
    return sorted(out)

def dset_path(f: h5py.File, quantity: str, cycle: str, species: int | None = None) -> str | None:
    p_flat = f"moments/{quantity}/{cycle}"
    if p_flat in f:
        return p_flat

    if species is not None:
        p_species = f"moments/species_{species}/{quantity}/{cycle}"
        if p_species in f:
            return p_species

    return None

def cycle_to_number(cycle: str) -> str:
    m = re.search(r"(\d+)$", cycle)
    return m.group(1) if m else cycle

def contiguous_block_containing(indices: np.ndarray, peak_idx: int):
    """
    Given sorted 1D indices and a peak index, return the contiguous block
    containing the peak.
    """
    if indices.size == 0:
        raise RuntimeError(f"No indices found above threshold for peak {peak_idx}.")

    split_points = np.where(np.diff(indices) != 1)[0] + 1
    blocks = np.split(indices, split_points)

    for block in blocks:
        if block.size and (block[0] <= peak_idx <= block[-1]):
            return int(block.min()), int(block.max())

    raise RuntimeError(f"Could not find contiguous block containing peak index {peak_idx}.")


###! ============================================================
###! Input arguments
###! ============================================================

parser = argparse.ArgumentParser(description="Extract global current-sheet bounds from iPIC3D proc*.hdf tiles")

parser.add_argument("dir_data", type=str, help="Directory containing proc*.hdf")
parser.add_argument("outdir",   type=str, help="Output directory")
parser.add_argument("quantity", type=str, choices=["rho", "Jx", "Jy", "Jz"],
                    help="Moments quantity to use")

parser.add_argument("t_start", type=int)
parser.add_argument("t_end",   type=int)
parser.add_argument("t_step",  type=int)

parser.add_argument("xlen", type=int, help="Simulation XLEN")
parser.add_argument("ylen", type=int, help="Simulation YLEN")
parser.add_argument("zlen", type=int, help="Simulation ZLEN")

parser.add_argument("--species", type=str, default="all",
                    help="Species selection: 'all' or comma list like '0,1,3'")
parser.add_argument("--cs_threshold", type=float, default=0.5,
                    help="Threshold fraction of Jmax used to define CS thickness")
parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"],
                    help="Working dtype for postprocessing arrays")

args = parser.parse_args()

dir_data = args.dir_data
outdir = args.outdir
quantity = args.quantity
XLEN, YLEN, ZLEN = args.xlen, args.ylen, args.zlen
cs_threshold = args.cs_threshold

work_dtype = np.float32 if args.dtype == "float32" else np.float64
mpi_dtype = MPI.FLOAT if args.dtype == "float32" else MPI.DOUBLE


###! ============================================================
###! Discover files
###! ============================================================

if rank == 0:
    all_files = sorted(glob.glob(os.path.join(dir_data, "proc*.hdf")))
    if not all_files:
        raise RuntimeError(f"No proc*.hdf found in {dir_data}")
else:
    all_files = None

all_files = comm.bcast(all_files, root=0)
local_files = all_files[rank::size]


###! ============================================================
###! Probe one valid local file to get tile shape and species
###! ============================================================

tile_shape = None
species_avail = None
flat_layout_found = False

if local_files:
    for fp in local_files:
        try:
            with h5py.File(fp, "r") as f:

                probe_cycle = f"cycle_{args.t_start}"

                # First try flat layout
                p0 = dset_path(f, quantity, probe_cycle, species=None)
                if p0 is not None:
                    tile_shape = tuple(f[p0].shape)
                    species_avail = None
                    flat_layout_found = True
                    break

                # Otherwise try species layout
                species_try = discover_species(f)
                if not species_try:
                    continue

                s0 = species_try[0]
                p0 = dset_path(f, quantity, probe_cycle, species=s0)
                if p0 is None:
                    continue

                tile_shape = tuple(f[p0].shape)
                species_avail = species_try
                break

        except Exception:
            if rank == 0:
                print(f"Probe failed for {fp}: {e}", flush=True)
            continue

tile_shape_all = comm.gather(tile_shape, root=0)
species_all = comm.gather(species_avail, root=0)
flat_all = comm.gather(flat_layout_found, root=0)

if rank == 0:
    valid_shapes = [s for s in tile_shape_all if s is not None]

    if not valid_shapes:
        raise RuntimeError(f"No valid dataset found for quantity='{quantity}' "
                            f"in any proc*.hdf under {dir_data}" )

    tile_shape = valid_shapes[0]
    flat_layout = any(flat_all)

    if flat_layout:
        species_avail = None
    else:
        valid_species = [s for s in species_all if s is not None]
        if not valid_species:
            raise RuntimeError("Could not determine species list from any valid file.")
        sp_lists = [set(s) for s in valid_species]
        species_avail = sorted(set.intersection(*sp_lists)) if sp_lists else []
else:
    tile_shape = None
    species_avail = None
    flat_layout = None

tile_shape = comm.bcast(tile_shape, root=0)
species_avail = comm.bcast(species_avail, root=0)
flat_layout = comm.bcast(flat_layout, root=0)

nx_tile, ny_tile, nz_tile = tile_shape

# Species selection
if flat_layout:
    species_sel = None
    if rank == 0:
        print("Using total Jz")
else:
    if args.species.strip().lower() == "all":
        species_sel = species_avail
    else:
        species_sel = [int(x) for x in args.species.split(",") if x.strip()]
        missing = sorted(set(species_sel) - set(species_avail))
        if missing:
            raise RuntimeError(f"Requested species not present: {missing}. Available: {species_avail}")

    if rank == 0:
        print("Using Jz over species:", species_sel)

# Shared nodal boundaries
nx_cell = nx_tile - 1
ny_cell = ny_tile - 1
nz_cell = nz_tile - 1

nx_global = XLEN * nx_cell + 1
ny_global = YLEN * ny_cell + 1
nz_global = ZLEN * nz_cell + 1


###! ============================================================
###! Infer proc -> (i,j,k) mapping globally (independent of slice)
###! ============================================================

best_name = None

if rank == 0:
    proc_ids = [proc_id_from_filename(fp) for fp in all_files]
    best_stats = None

    for name, fn in mapping_candidates(XLEN, YLEN, ZLEN):
        occ3d = np.zeros((nx_global, ny_global, nz_global), dtype=np.int8)
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

            occ3d[gx0:gx0+nx_use, gy0:gy0+ny_use, gz0:gz0+nz_use] += 1

        if bad:
            continue

        score, gaps, overlaps, maxv = score_occupancy(occ3d)
        if best_stats is None or score < best_stats[0]:
            best_stats = (score, gaps, overlaps, maxv)
            best_name = name

    if best_name is None:
        raise RuntimeError("Could not infer proc->(i,j,k) mapping from filenames.")

best_name = comm.bcast(best_name, root=0)
maps = {n: fn for n, fn in mapping_candidates(XLEN, YLEN, ZLEN)}
pid_to_ijk = maps[best_name]


###! ============================================================
###! Build J_yz(y,z) = sum_x |J(x,y,z)| for one time cycle
###! ============================================================

def assemble_Jyz_for_cycle(cycle_name: str):
    local_Jyz = np.zeros((ny_global, nz_global), dtype=work_dtype)
    found_any = False

    for fp in local_files:
        pid = proc_id_from_filename(fp)
        i, j, k = pid_to_ijk(pid)

        xs = 0 if i == 0 else 1
        ys = 0 if j == 0 else 1
        zs = 0 if k == 0 else 1

        x0 = i * nx_cell
        y0 = j * ny_cell
        z0 = k * nz_cell

        gy0 = y0 + ys
        gz0 = z0 + zs

        ny_use = ny_tile - ys
        nz_use = nz_tile - zs

        try:
            with h5py.File(fp, "r") as f:

                if flat_layout:
                    path = dset_path(f, quantity, cycle_name, species=None)
                    if path is None:
                        raise RuntimeError(f"Missing dataset for '{quantity}/{cycle_name}' in file '{fp}'")
                    tile_sum = np.asarray(f[path][...], dtype=work_dtype)

                else:
                    tile_sum = None
                    for s in species_sel:
                        path = dset_path(f, quantity, cycle_name, species=s)
                        if path is None:
                            raise RuntimeError(f"Missing dataset for species {s}, quantity '{quantity}', cycle '{cycle_name}' in file '{fp}'")
                        arr = np.asarray(f[path][...], dtype=work_dtype)
                        tile_sum = arr if tile_sum is None else (tile_sum + arr)

                if tile_sum is None:
                    continue

                found_any = True

                tile_sum = tile_sum[xs:, ys:, zs:]
                tile_yz = np.sum(np.abs(tile_sum), axis=0)
                local_Jyz[gy0:gy0+ny_use, gz0:gz0+nz_use] += tile_yz

        except Exception:
            continue

    found_any_global = comm.allreduce(found_any, op=MPI.LOR)

    if not found_any_global:
        return None

    Jyz_global = np.zeros((ny_global, nz_global), dtype=work_dtype) if rank == 0 else None
    comm.Reduce([local_Jyz, mpi_dtype], [Jyz_global, mpi_dtype], op=MPI.SUM, root=0)

    return Jyz_global


###! ============================================================
###! Main extraction
###! ============================================================

t_start = args.t_start
t_end   = args.t_end
t_step  = args.t_step

for t in range(t_start, t_end + 1, t_step):

    time_cycle = f"cycle_{t}"

    if rank == 0:
        print(f"\nProcessing {time_cycle}", flush=True)

    Jyz_global = assemble_Jyz_for_cycle(time_cycle)

    if Jyz_global is None:
        if rank == 0:
            print(f"Skipping {time_cycle}: dataset not found", flush=True)
        continue

    if rank == 0:

        yL1_global = float("inf")
        yL2_global = -float("inf")
        yU1_global = float("inf")
        yU2_global = -float("inf")

        mid = ny_global // 2

        for z_idx in range(nz_global):
            J_list = Jyz_global[:, z_idx]

            if np.all(J_list == 0):
                continue

            lower = J_list[:mid]
            if np.all(lower == 0):
                continue
            idx_lower = int(np.argmax(lower))
            J_max_lower = lower[idx_lower]

            upper = J_list[mid:]
            if upper.size == 0 or np.all(upper == 0):
                continue
            idx_upper_local = int(np.argmax(upper))
            idx_upper = idx_upper_local + mid
            J_max_upper = J_list[idx_upper]

            mask_lower = lower >= cs_threshold * J_max_lower
            indices_lower = np.where(mask_lower)[0]
            yL1, yL2 = contiguous_block_containing(indices_lower, idx_lower)

            mask_upper = upper >= cs_threshold * J_max_upper
            indices_upper = np.where(mask_upper)[0] + mid
            yU1, yU2 = contiguous_block_containing(indices_upper, idx_upper)

            yL1_global = min(yL1_global, yL1)
            yL2_global = max(yL2_global, yL2)
            yU1_global = min(yU1_global, yU1)
            yU2_global = max(yU2_global, yU2)

        if not np.isfinite(yL1_global) or not np.isfinite(yU1_global):
            print(f"Skipping {time_cycle}: no valid CS found", flush=True)
            continue

        os.makedirs(outdir, exist_ok=True)
        outfile = os.path.join(outdir, "CS_bounds.txt")

        mode = "a"
        write_header = not os.path.exists(outfile)

        with open(outfile, mode) as f:
            if write_header:
                f.write("Cycle Y_lower_min Y_lower_max Y_upper_min Y_upper_max\n")

            f.write(f"{t} {yL1_global} {yL2_global} {yU1_global} {yU2_global}\n")

        print(f"Wrote CS bounds for {time_cycle}", flush=True)