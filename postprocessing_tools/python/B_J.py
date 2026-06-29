"""
    SCRIPT="../postprocessing_tools/python/B_J.py"
    DATA_DIR="/scratch/project_465003132/test_xy/"
    OUT_DIR="/scratch/project_465003132/test_xy/plots"

    XLEN=32; YLEN=2; ZLEN=32
    xmin=0; xmax=18
    ymin=0; ymax=36
    zmin=0; zmax=18

    srun python3 "$SCRIPT" "$DATA_DIR" "$XLEN" "$YLEN" "$ZLEN" \
        $xmin $xmax $ymin $ymax $zmin $zmax \
        --cycle-start 0 \
        --cycle-end 5000 \
        --cycle-step 100 \
        --outdir "$OUT_DIR"

    NOTES: 
    1. Slices are ALWAYS taken at the global midplane:
        XY plane  -> z = nz_global // 2
        ZY plane  -> x = nx_global // 2

    2. J layout is auto-detected from the first file:
        total      -> moments/Jz/<cycle>                (single dataset; --species ignored)
        per-species-> moments/species_<sp>/Jz/<cycle>   (summed over --species)
"""

import os
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
os.environ.setdefault("MPLBACKEND", "Agg")

import glob
import argparse
from datetime import datetime

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpi4py import MPI


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

start_time = datetime.now()

comm.Barrier()
if rank == 0:
    print("All ranks passed first barrier", flush=True)


def proc_id_from_filename(fp):
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

    return {"A": A, "B": B, "C": C, "D": D, "E": E, "F": F}


def choose_mapping(files, XLEN, YLEN, ZLEN):
    proc_ids = [proc_id_from_filename(fp) for fp in files]
    maps = mapping_candidates(XLEN, YLEN, ZLEN)

    best_name = None
    best_score = None

    for name, fn in maps.items():
        occ = np.zeros((XLEN, YLEN, ZLEN), dtype=np.int32)
        valid = True

        for pid in proc_ids:
            i, j, k = fn(pid)

            if not (0 <= i < XLEN and 0 <= j < YLEN and 0 <= k < ZLEN):
                valid = False
                break

            occ[i, j, k] += 1

        if not valid:
            continue

        gaps = int(np.count_nonzero(occ == 0))
        overlaps = int(np.count_nonzero(occ > 1))
        score = 10 * gaps + 100 * overlaps

        if best_score is None or score < best_score:
            best_score = score
            best_name = name

    if best_name is None:
        raise RuntimeError("Could not determine proc -> (i,j,k) mapping.")

    return best_name, best_score


def global_shape_shared(tile_shape, XLEN, YLEN, ZLEN):
    nx, ny, nz = tile_shape
    nx_global = XLEN * (nx - 1) + 1
    ny_global = YLEN * (ny - 1) + 1
    nz_global = ZLEN * (nz - 1) + 1
    return nx_global, ny_global, nz_global


def global_offset_shared(i, j, k, tile_shape):
    nx, ny, nz = tile_shape
    x0 = i * (nx - 1)
    y0 = j * (ny - 1)
    z0 = k * (nz - 1)
    return x0, y0, z0


def lower_crop_indices(i, j, k):
    xs = 0 if i == 0 else 1
    ys = 0 if j == 0 else 1
    zs = 0 if k == 0 else 1
    return xs, ys, zs


def jz_path(current_mode, sp, cycle_name):
    ###! total mode: single combined dataset; species mode: per-species dataset
    if current_mode == "total":
        return f"moments/Jz/{cycle_name}"
    return f"moments/species_{sp}/Jz/{cycle_name}"


def read_current_sum_xy(f, cycle_name, species_list, raw_k, xs, ys, current_mode):
    # total -> one dataset (sp ignored); species -> sum over species_list
    sp_iter = [None] if current_mode == "total" else species_list
    Jz_sum = None

    for sp in sp_iter:
        path = jz_path(current_mode, sp, cycle_name)

        if path not in f:
            raise KeyError(f"Missing dataset: {path}")

        slab = np.array(f[path][xs:, ys:, raw_k], dtype=np.float64)

        if Jz_sum is None:
            Jz_sum = np.zeros_like(slab, dtype=np.float64)

        Jz_sum += slab

    return Jz_sum


def read_current_sum_zy(f, cycle_name, species_list, raw_i, ys, zs, current_mode):
    sp_iter = [None] if current_mode == "total" else species_list
    Jz_sum = None

    for sp in sp_iter:
        path = jz_path(current_mode, sp, cycle_name)

        if path not in f:
            raise KeyError(f"Missing dataset: {path}")

        slab_yz = np.array(f[path][raw_i, ys:, zs:], dtype=np.float64)
        slab_zy = slab_yz.T

        if Jz_sum is None:
            Jz_sum = np.zeros_like(slab_zy, dtype=np.float64)

        Jz_sum += slab_zy

    return Jz_sum


def assemble_cycle(cycle_name, local_files, rank_to_ijk, Bx_shape, Jz_shape,
                   Bx_global_shape, Jz_global_shape, z_index, x_index,
                   species_list, current_mode):
    Bx_nx, Bx_ny, Bx_nz = Bx_global_shape
    Jz_nx, Jz_ny, Jz_nz = Jz_global_shape

    local_Bx_XY = np.zeros((Bx_nx, Bx_ny), dtype=np.float64)
    local_Jz_XY = np.zeros((Jz_nx, Jz_ny), dtype=np.float64)
    local_Jz_ZY = np.zeros((Jz_nz, Jz_ny), dtype=np.float64)

    local_Bx_occ = np.zeros((Bx_nx, Bx_ny), dtype=np.int32)
    local_Jz_XY_occ = np.zeros((Jz_nx, Jz_ny), dtype=np.int32)
    local_Jz_ZY_occ = np.zeros((Jz_nz, Jz_ny), dtype=np.int32)

    for fp in local_files:
        pid = proc_id_from_filename(fp)
        i, j, k = rank_to_ijk(pid)

        Bx_raw_nx, Bx_raw_ny, Bx_raw_nz = Bx_shape
        Jz_raw_nx, Jz_raw_ny, Jz_raw_nz = Jz_shape

        Bx_ox, Bx_oy, Bx_oz = global_offset_shared(i, j, k, Bx_shape)
        Jz_ox, Jz_oy, Jz_oz = global_offset_shared(i, j, k, Jz_shape)

        Bx_xs, Bx_ys, Bx_zs = lower_crop_indices(i, j, k)
        Jz_xs, Jz_ys, Jz_zs = lower_crop_indices(i, j, k)

        Bx_x0 = Bx_ox + Bx_xs
        Bx_y0 = Bx_oy + Bx_ys
        Bx_z0 = Bx_oz + Bx_zs

        Jz_x0 = Jz_ox + Jz_xs
        Jz_y0 = Jz_oy + Jz_ys
        Jz_z0 = Jz_oz + Jz_zs

        Bx_use_nx = Bx_raw_nx - Bx_xs
        Bx_use_ny = Bx_raw_ny - Bx_ys
        Bx_use_nz = Bx_raw_nz - Bx_zs

        Jz_use_nx = Jz_raw_nx - Jz_xs
        Jz_use_ny = Jz_raw_ny - Jz_ys
        Jz_use_nz = Jz_raw_nz - Jz_zs

        Bx_has_xy = Bx_z0 <= z_index < Bx_z0 + Bx_use_nz
        Jz_has_xy = Jz_z0 <= z_index < Jz_z0 + Jz_use_nz
        Jz_has_zy = Jz_x0 <= x_index < Jz_x0 + Jz_use_nx

        if not (Bx_has_xy or Jz_has_xy or Jz_has_zy):
            continue

        with h5py.File(fp, "r") as f:
            if Bx_has_xy:
                raw_k = z_index - Bx_oz
                path = f"fields/Bx/{cycle_name}"

                if path not in f:
                    raise KeyError(f"Missing dataset: {path}")

                Bx_slab = np.array(f[path][Bx_xs:, Bx_ys:, raw_k], dtype=np.float64)
                local_Bx_XY[Bx_x0:Bx_x0 + Bx_use_nx, Bx_y0:Bx_y0 + Bx_use_ny] = Bx_slab
                local_Bx_occ[Bx_x0:Bx_x0 + Bx_use_nx, Bx_y0:Bx_y0 + Bx_use_ny] += 1

            if Jz_has_xy:
                raw_k = z_index - Jz_oz
                Jz_slab = read_current_sum_xy(f, cycle_name, species_list, raw_k,
                                              Jz_xs, Jz_ys, current_mode)
                local_Jz_XY[Jz_x0:Jz_x0 + Jz_use_nx, Jz_y0:Jz_y0 + Jz_use_ny] = Jz_slab
                local_Jz_XY_occ[Jz_x0:Jz_x0 + Jz_use_nx, Jz_y0:Jz_y0 + Jz_use_ny] += 1

            if Jz_has_zy:
                raw_i = x_index - Jz_ox
                Jz_slab = read_current_sum_zy(f, cycle_name, species_list, raw_i,
                                              Jz_ys, Jz_zs, current_mode)
                local_Jz_ZY[Jz_z0:Jz_z0 + Jz_use_nz, Jz_y0:Jz_y0 + Jz_use_ny] = Jz_slab
                local_Jz_ZY_occ[Jz_z0:Jz_z0 + Jz_use_nz, Jz_y0:Jz_y0 + Jz_use_ny] += 1

    Bx_XY = None
    Jz_XY = None
    Jz_ZY = None
    Bx_occ = None
    Jz_XY_occ = None
    Jz_ZY_occ = None

    if rank == 0:
        Bx_XY = np.zeros((Bx_nx, Bx_ny), dtype=np.float64)
        Jz_XY = np.zeros((Jz_nx, Jz_ny), dtype=np.float64)
        Jz_ZY = np.zeros((Jz_nz, Jz_ny), dtype=np.float64)
        Bx_occ = np.zeros((Bx_nx, Bx_ny), dtype=np.int32)
        Jz_XY_occ = np.zeros((Jz_nx, Jz_ny), dtype=np.int32)
        Jz_ZY_occ = np.zeros((Jz_nz, Jz_ny), dtype=np.int32)

    comm.Reduce(local_Bx_XY, Bx_XY, op=MPI.SUM, root=0)
    comm.Reduce(local_Jz_XY, Jz_XY, op=MPI.SUM, root=0)
    comm.Reduce(local_Jz_ZY, Jz_ZY, op=MPI.SUM, root=0)

    comm.Reduce(local_Bx_occ, Bx_occ, op=MPI.SUM, root=0)
    comm.Reduce(local_Jz_XY_occ, Jz_XY_occ, op=MPI.SUM, root=0)
    comm.Reduce(local_Jz_ZY_occ, Jz_ZY_occ, op=MPI.SUM, root=0)

    return Bx_XY, Jz_XY, Jz_ZY, Bx_occ, Jz_XY_occ, Jz_ZY_occ


def plot_cycle(cycle_name, Bx_XY, Jz_XY, Jz_ZY, Bx_occ, Jz_XY_occ, Jz_ZY_occ,
               z_index, x_index, outdir, species_list, extents, current_mode):
    ###! extents = (xmin, xmax, ymin, ymax, zmin, zmax) in physical units (user-supplied)
    xmin, xmax, ymin, ymax, zmin, zmax = extents

    if Bx_occ.min() == 0:
        print(f"WARNING: {cycle_name} Bx XY has gaps.", flush=True)

    if Jz_XY_occ.min() == 0:
        print(f"WARNING: {cycle_name} Jz XY has gaps.", flush=True)

    if Jz_ZY_occ.min() == 0:
        print(f"WARNING: {cycle_name} Jz ZY has gaps.", flush=True)

    fig, axs = plt.subplots(1, 3, figsize=(16, 8), dpi=200)

    # XY planes: horizontal = x (xmin..xmax), vertical = y (ymin..ymax)
    im0 = axs[0].imshow(Bx_XY.T, origin="lower", cmap="seismic", aspect="auto",
                        vmin=-0.25, vmax=0.25, extent=[xmin, xmax, ymin, ymax])
    axs[0].set_title(r"$B_x$ $(X,Y)$", fontsize=16)
    axs[0].set_xlabel(r"$x\,\omega_p/c$", fontsize=12)
    axs[0].set_ylabel(r"$y\,\omega_p/c$", fontsize=12)
    axs[0].tick_params(axis="both", which="major", labelsize=12, length=6)
    fig.colorbar(im0, ax=axs[0])

    im1 = axs[1].imshow(Jz_XY.T, origin="lower", cmap="seismic", aspect="auto",
                        vmin=-0.08, vmax=0.08, extent=[xmin, xmax, ymin, ymax])
    axs[1].set_title(r"$J_z$ $(X,Y)$", fontsize=16)
    axs[1].set_xlabel(r"$x\,\omega_p/c$", fontsize=12)
    axs[1].set_ylabel(r"$y\,\omega_p/c$", fontsize=12)
    axs[1].tick_params(axis="both", which="major", labelsize=12, length=6)
    fig.colorbar(im1, ax=axs[1])

    # ZY plane: horizontal = z (zmin..zmax), vertical = y (ymin..ymax)
    im2 = axs[2].imshow(Jz_ZY.T, origin="lower", cmap="seismic", aspect="auto",
                        vmin=-0.08, vmax=0.08, extent=[zmin, zmax, ymin, ymax])
    axs[2].set_title(r"$J_z$ $(Z,Y)$", fontsize=16)
    axs[2].set_xlabel(r"$z\,\omega_p/c$", fontsize=12)
    axs[2].set_ylabel(r"$y\,\omega_p/c$", fontsize=12)
    axs[2].tick_params(axis="both", which="major", labelsize=12, length=6)
    fig.colorbar(im2, ax=axs[2])

    cycle_number = int(cycle_name.replace("cycle_", ""))
    time_omega = cycle_number // 5          ###! T = cycle/10; verify dt matches this convention
    fig.suptitle(rf"$T = {time_omega}\,\omega_p^{{-1}}$", fontsize=18)
    fig.tight_layout()

    outfile = os.path.join(outdir, f"{cycle_name}.png")
    fig.savefig(outfile, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {outfile}", flush=True)


parser = argparse.ArgumentParser(description="Fast MPI plotter for Bx(XY), Jz(XY), and Jz(ZY). Slices fixed at the global midplane. Jz layout (total vs per-species) auto-detected.")

parser.add_argument("dir_data", type=str)
parser.add_argument("xlen", type=int)
parser.add_argument("ylen", type=int)
parser.add_argument("zlen", type=int)

###! Physical box extents (positional, in same order as the shell snippet)
parser.add_argument("xmin", type=float)
parser.add_argument("xmax", type=float)
parser.add_argument("ymin", type=float)
parser.add_argument("ymax", type=float)
parser.add_argument("zmin", type=float)
parser.add_argument("zmax", type=float)

parser.add_argument("--cycle-start", type=int, default=0)
parser.add_argument("--cycle-end", type=int, default=20000)
parser.add_argument("--cycle-step", type=int, default=500)

parser.add_argument("--species", type=int, nargs="+", default=[1, 2])

parser.add_argument("--outdir", type=str, default=None)
parser.add_argument("--mapping", type=str, default="auto", choices=["auto", "A", "B", "C", "D", "E", "F"])

args = parser.parse_args()

dir_data = args.dir_data
XLEN = args.xlen
YLEN = args.ylen
ZLEN = args.zlen
species_list = args.species
extents = (args.xmin, args.xmax, args.ymin, args.ymax, args.zmin, args.zmax)

if args.outdir is None:
    outdir = dir_data
else:
    outdir = args.outdir

if rank == 0:
    os.makedirs(outdir, exist_ok=True)

num_expected_files = XLEN * YLEN * ZLEN

if rank == 0:
    all_files = sorted(glob.glob(os.path.join(dir_data, "proc*.hdf")))

    if len(all_files) == 0:
        raise RuntimeError(f"No proc*.hdf files found in {dir_data}")

    if len(all_files) != num_expected_files:
        print(f"WARNING: expected {num_expected_files} files, found {len(all_files)}", flush=True)

    if args.mapping == "auto":
        map_name, map_score = choose_mapping(all_files, XLEN, YLEN, ZLEN)
    else:
        map_name = args.mapping
        map_score = -1

else:
    all_files = None
    map_name = None

all_files = comm.bcast(all_files, root=0)
map_name = comm.bcast(map_name, root=0)

maps = mapping_candidates(XLEN, YLEN, ZLEN)
rank_to_ijk = maps[map_name]

local_files = all_files[rank::size]

first_cycle = f"cycle_{args.cycle_start}"

if rank == 0:
    sample_file = all_files[0]

    with h5py.File(sample_file, "r") as f:
        Bx_path = f"fields/Bx/{first_cycle}"

        if Bx_path not in f:
            raise KeyError(f"Missing dataset in sample file: {Bx_path}")

        Bx_shape = f[Bx_path].shape

        ###! Detect current layout: total (moments/Jz) vs per-species (moments/species_N/Jz)
        total_path = f"moments/Jz/{first_cycle}"
        species_path = f"moments/species_{species_list[0]}/Jz/{first_cycle}"

        if total_path in f:
            current_mode = "total"
            Jz_shape = f[total_path].shape
        elif species_path in f:
            current_mode = "species"
            Jz_shape = f[species_path].shape
        else:
            raise KeyError(
                f"Found neither '{total_path}' nor '{species_path}' in sample file. "
                f"Cannot locate Jz."
            )

    Bx_global_shape = global_shape_shared(Bx_shape, XLEN, YLEN, ZLEN)
    Jz_global_shape = global_shape_shared(Jz_shape, XLEN, YLEN, ZLEN)

else:
    Bx_shape = None
    Jz_shape = None
    Bx_global_shape = None
    Jz_global_shape = None
    current_mode = None

Bx_shape = comm.bcast(Bx_shape, root=0)
Jz_shape = comm.bcast(Jz_shape, root=0)
Bx_global_shape = comm.bcast(Bx_global_shape, root=0)
Jz_global_shape = comm.bcast(Jz_global_shape, root=0)
current_mode = comm.bcast(current_mode, root=0)

if rank == 0:
    if current_mode == "species":
        print(f"Current layout: 'species' (summing species {species_list})", flush=True)
    else:
        print("Current layout: 'total' (single combined dataset; --species ignored)", flush=True)

Bx_nx, Bx_ny, Bx_nz = Bx_global_shape
Jz_nx, Jz_ny, Jz_nz = Jz_global_shape

###! ALWAYS slice at the global midplane (no longer user-selectable)
###! XY plane taken at z = nz_global // 2 ; ZY plane taken at x = nx_global // 2
z_index = Bx_nz // 2
x_index = Jz_nx // 2

###! Guards: midplane indices must be valid for the arrays they index into.
if not (0 <= z_index < Bx_nz):
    raise ValueError(f"midplane z_index={z_index} outside Bx z range [0,{Bx_nz - 1}]")
if not (0 <= z_index < Jz_nz):
    raise ValueError(f"midplane z_index={z_index} outside Jz z range [0,{Jz_nz - 1}] "
                     f"(Bx and Jz have different global z — midplanes differ)")
if not (0 <= x_index < Jz_nx):
    raise ValueError(f"midplane x_index={x_index} outside Jz x range [0,{Jz_nx - 1}]")

if rank == 0:
    print(f"Midplane slices: XY at z={z_index} (of {Bx_nz}), ZY at x={x_index} (of {Jz_nx})", flush=True)

cycles = list(range(args.cycle_start, args.cycle_end + 1, args.cycle_step))

for cycle in cycles:
    cycle_name = f"cycle_{cycle}"

    if rank == 0:
        print("", flush=True)
        print(f"Processing {cycle_name}", flush=True)

    Bx_XY, Jz_XY, Jz_ZY, Bx_occ, Jz_XY_occ, Jz_ZY_occ = assemble_cycle(
        cycle_name, local_files, rank_to_ijk, Bx_shape, Jz_shape,
        Bx_global_shape, Jz_global_shape, z_index, x_index,
        species_list, current_mode)

    if rank == 0:
        plot_cycle(cycle_name, Bx_XY, Jz_XY, Jz_ZY, Bx_occ, Jz_XY_occ, Jz_ZY_occ,
                   z_index, x_index, outdir, species_list, extents, current_mode)

if rank == 0:
    print("", flush=True)
    print(f"All plots saved in: {outdir}", flush=True)
    print(f"Complete. Time elapsed = {datetime.now() - start_time}", flush=True)