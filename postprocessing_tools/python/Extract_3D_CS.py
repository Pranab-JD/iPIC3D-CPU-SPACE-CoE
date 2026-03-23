"""
Created on Sun Mar 22 18:23 2026

@author: Pranab JD, ChatGPT

Description: Define CS width as 0.5 times the maximum of Jz (for each sheet individually).
             The width of the 2 current sheets are written in terms of Y-indices to 
             "CS_bounds.txt". These indices can be used to extract data along the CSs.
"""

import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os, glob, argparse, re, h5py
from mpl_toolkits.axes_grid1 import make_axes_locatable

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

###? ================================================================================= ?###

###! Helpers

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

def score_occupancy(Occ: np.ndarray):
    gaps = int(np.count_nonzero(Occ == 0))
    overlaps = int(np.count_nonzero(Occ > 1))
    maxv = int(Occ.max())
    score = gaps * 10 + overlaps * 2 + max(0, maxv - 1) * 1000
    return score, gaps, overlaps, maxv

def discover_species(f: h5py.File):
    """Return sorted list of species indices available under /moments/species_*."""
    if "moments" not in f:
        return []
    out = []
    for k in f["moments"].keys():
        m = re.match(r"species_(\d+)$", k)
        if m:
            out.append(int(m.group(1)))
    return sorted(out)

def dset_path(species: int, quantity: str, cycle: str) -> str:
    return f"moments/species_{species}/{quantity}/{cycle}"

def cycle_to_number(cycle: str) -> str:
    # "cycle_19500" -> "19500"
    m = re.search(r"(\d+)$", cycle)
    return m.group(1) if m else cycle

def assemble_Jyz_for_cycle(cycle_name):
    local_Jyz = np.zeros((ny_global, nz_global), dtype=np.float64)

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

        with h5py.File(fp, "r") as f:
            tile_sum = None
            for s in species_sel:
                path = dset_path(s, quantity, cycle_name)
                arr = np.array(f[path][:, :, :], dtype=np.float64)
                tile_sum = arr if tile_sum is None else (tile_sum + arr)

        # remove duplicated nodal boundaries
        tile_sum = tile_sum[xs:, ys:, zs:]   # shape = (nx_use, ny_use, nz_use)

        # sum over x -> gives (ny_use, nz_use)
        tile_yz = np.sum(np.abs(tile_sum), axis=0)

        # accumulate into global yz array
        local_Jyz[gy0:gy0+ny_use, gz0:gz0+nz_use] += tile_yz

    Jyz_global = np.zeros((ny_global, nz_global), dtype=np.float64) if rank == 0 else None
    comm.Reduce(local_Jyz, Jyz_global, op=MPI.SUM, root=0)

    return Jyz_global

###? ================================================================================= ?###

###! Input arguments

parser = argparse.ArgumentParser(description="Plot 2D slice of summed moments (rho/Jx/Jy/Jz) from iPIC3D proc*.hdf tiles")

parser.add_argument("dir_data",   type=str, help="Directory containing proc*.hdf")
parser.add_argument("outdir", type=str, default="", help="Output PNG files")

parser.add_argument("quantity", type=str, choices=["rho","Jx","Jy","Jz"],
                    help="Which moments quantity to plot (summed over species)")
parser.add_argument("time_cycle", type=str, help="Cycle group name, e.g. cycle_19500")

parser.add_argument("xlen", type=int, help="Simulation XLEN")
parser.add_argument("ylen", type=int, help="Simulation YLEN")
parser.add_argument("zlen", type=int, help="Simulation ZLEN")

parser.add_argument("--species", type=str, default="all",
                    help="Species selection: 'all' or comma list like '0,1,3'")

args = parser.parse_args()

dir_data   = args.dir_data
time_cycle = args.time_cycle
XLEN, YLEN, ZLEN = args.xlen, args.ylen, args.zlen
quantity = args.quantity

###? ================================================================================= ?###

###* Discover files
if rank == 0:
    all_files = sorted(glob.glob(os.path.join(dir_data, "proc*.hdf")))
    if not all_files:
        raise RuntimeError(f"No proc*.hdf found in {dir_data}")
else:
    all_files = None

all_files = comm.bcast(all_files, root=0)
local_files = all_files[rank::size]

###* Probe tile shape + species list
tile_shape = None
species_avail = None

if local_files:
    for fp in local_files:
        try:
            with h5py.File(fp, "r") as f:

                if "moments" not in f:
                    print(f"[rank {rank}] no /moments in {fp}")
                    continue

                species_try = discover_species(f)

                if not species_try:
                    continue

                s0 = species_try[0]
                path0 = dset_path(s0, quantity, time_cycle)

                if path0 not in f:
                    print(f"[rank {rank}] missing dataset {path0} in {fp}")
                    continue

                species_avail = species_try
                tile_shape = tuple(f[path0].shape)
                break

        except Exception as e:
            print(f"[rank {rank}] failed reading {fp}: {e}")
            continue

tile_shape_all = comm.gather(tile_shape, root=0)
species_all    = comm.gather(species_avail, root=0)

if rank == 0:

    valid_shapes = [s for s in tile_shape_all if s is not None]
    valid_species = [s for s in species_all if s is not None]

    if not valid_shapes:
        raise RuntimeError(f"No valid dataset found for quantity='{quantity}' and time_cycle='{time_cycle}' "
                            f"in any proc*.hdf file under {dir_data}")

    tile_shape = valid_shapes[0]

    if not valid_species:
        raise RuntimeError(f"No valid species list found for quantity='{quantity}' and time_cycle='{time_cycle}'")

    sp_lists = [set(s) for s in valid_species]
    species_avail = sorted(set.intersection(*sp_lists)) if sp_lists else []
    # intersect species lists across ranks (should be identical; this is just safe)
    sp_lists = [set(s) for s in species_all if s is not None]
    species_avail = sorted(set.intersection(*sp_lists)) if sp_lists else []

    print("Writing CS thickness based on ", quantity, " summed over species:", species_avail)

else:
    species_avail = None

tile_shape = comm.bcast(tile_shape, root=0)
species_avail = comm.bcast(species_avail, root=0)

nx_tile, ny_tile, nz_tile = tile_shape

# Species selection
if args.species.strip().lower() == "all":
    species_sel = species_avail
else:
    species_sel = [int(x) for x in args.species.split(",") if x.strip() != ""]
    missing = sorted(set(species_sel) - set(species_avail))
    if missing:
        raise RuntimeError(f"Requested species not present: {missing}. Available: {species_avail}")

# Nodal-like shared boundaries
nx_cell = nx_tile - 1
ny_cell = ny_tile - 1
nz_cell = nz_tile - 1

nx_global = XLEN * nx_cell + 1
ny_global = YLEN * ny_cell + 1
nz_global = ZLEN * nz_cell + 1

###! Infer proc->(i,j,k) mapping
best_name = None
if rank == 0:
    proc_ids = [proc_id_from_filename(fp) for fp in all_files]
    best_stats = None

    for name, fn in mapping_candidates(XLEN, YLEN, ZLEN):
        Occ = np.zeros((nx_global, ny_global, nz_global), dtype=np.int32)
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

            Occ[gx0:gx0+nx_use, gy0:gy0+ny_use, gz0:gz0+nz_use] += 1

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

###? ================================================================================= ?###

###! Assemble the CS on the root MPI 

Jyz_global = assemble_Jyz_for_cycle(time_cycle)

if rank == 0:
    yL1_global = float("inf")
    yL2_global = -float("inf")
    yU1_global = float("inf")
    yU2_global = -float("inf")

    for slice_g in range(nz_global):
        J_list = Jyz_global[:, slice_g]

        mid = ny_global // 2

        # lower CS
        idx_lower = np.argmax(J_list[:mid])
        J_max_lower = J_list[idx_lower]

        # upper CS
        idx_upper_local = np.argmax(J_list[mid:])
        idx_upper = idx_upper_local + mid
        J_max_upper = J_list[idx_upper]

        cs_threshold = 0.5              #! 50% of the maximum current along Z (Jz)

        # lower bounds
        mask_lower = J_list[:mid] >= cs_threshold * J_max_lower
        indices_lower = np.where(mask_lower)[0]
        blocks_lower = np.split(indices_lower, np.where(np.diff(indices_lower) != 1)[0] + 1)
        yL1, yL2 = next((b.min(), b.max()) for b in blocks_lower if idx_lower in b)

        # upper bounds
        mask_upper = J_list[mid:] >= cs_threshold * J_max_upper
        indices_upper = np.where(mask_upper)[0] + mid
        blocks_upper = np.split(indices_upper, np.where(np.diff(indices_upper) != 1)[0] + 1)
        yU1, yU2 = next((b.min(), b.max()) for b in blocks_upper if idx_upper in b)

        yL1_global = min(yL1_global, yL1)
        yL2_global = max(yL2_global, yL2)
        yU1_global = min(yU1_global, yU1)
        yU2_global = max(yU2_global, yU2)

###? ================================================================================= ?###

###! Write CS bounds to a file 

if rank == 0:
    if args.outdir.strip():
        outdir = args.outdir
    else:
        outdir = "."

    cyc = cycle_to_number(time_cycle)
    os.makedirs(outdir, exist_ok=True)
    outfile = os.path.join(outdir, "CS_bounds.txt")

    # write header only once
    write_header = not os.path.exists(outfile)
    with open(outfile, "a") as f:
        if write_header:
            f.write("Cycle       Y_lower_min     Y_lower_max     Y_upper_min     Y_upper_max\n")

        f.write(f"{cyc}     {yL1_global}        {yL2_global}        {yU1_global}        {yU2_global}\n")

    print("Wrote CS bounds at ", time_cycle, " to :", outfile)
    print()

###? ================================================================================= ?###

    ###! Plot figure

    # fig, ax = plt.subplots(figsize=(6.5, 8.0), dpi=250)

    # im = ax.imshow(out_plane.T, origin="lower", aspect="auto", cmap=args.cmap, vmin=vmin, vmax=vmax)

    # ###* Upper CS
    # ax.axhline(y=yU1, color="k", linestyle="--", linewidth=1.5)
    # ax.axhline(y=yU2, color="k", linestyle="--", linewidth=1.5)

    # ###* Lower CS
    # ax.axhline(y=yL1, color="k", linestyle="--", linewidth=1.5)
    # ax.axhline(y=yL2, color="k", linestyle="--", linewidth=1.5)

    # ax.set_xticks(x_pos)
    # ax.set_xticklabels([f"{v:.0f}" for v in x_lab])
    # ax.set_yticks(y_pos)
    # ax.set_yticklabels([f"{v:.0f}" for v in y_lab])

    # ax.tick_params(axis='x', which='major', labelsize=16, length=8)
    # ax.tick_params(axis='y', which='major', labelsize=16, length=8)

    # ax.set_xlabel("X", fontsize=16)
    # ax.set_ylabel("Y", fontsize=16)

    # ax.set_title(f"{cycle_number} $\\omega^{{-1}}$", fontsize=18)

    # ###* Colorbar matched to image height
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="5%", pad=0.10)
    # cbar = fig.colorbar(im, cax=cax)
    # cbar.set_label(f"{quantity}  ({axis}={slice_g})", fontsize=18)
    # cbar.ax.tick_params(labelsize=16)
    # cbar.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter(f"%.{args.cb_decimals}f"))
    # plt.tight_layout()

    # ###* Save plots in the user-specified folder
    # if args.outdir.strip():
    #     outdir = args.outdir
    # else:
    #     outdir = "."

    # os.makedirs(outdir, exist_ok=True)
    # out_png = os.path.join(outdir, f"{quantity}_slice_{axis}{slice_g}_{time_cycle}.png")

    # plt.savefig(out_png)
    # plt.close()
    # print("Wrote:", out_png)
    # print()

###? ================================================================================= ?###