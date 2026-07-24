"""
Created on Wed Jun 17:00 2025

@author: Pranab JD, ChatGPT

Description: Plot total charge density (summed over all species; 2D)

    SCRIPT="../postprocessing_tools/python/Density.py"
    DATA_DIR="/scratch/project_465003132/turb_256/"
    OUT_DIR="/scratch/project_465003132/turb_256/plots"

    XLEN=16; YLEN=16; ZLEN=1
    xmin=0; xmax=64
    ymin=0; ymax=64
    zmin=0; zmax=0.1

    srun python3 -u "$SCRIPT" "$DATA_DIR" "$XLEN" "$YLEN" "$ZLEN" \
        $xmin $xmax $ymin $ymax $zmin $zmax \
        --cycle-start 0 \
        --cycle-end 4200 \
        --cycle-step 100 \
        --outdir "$OUT_DIR"

    ONE srun call renders the whole cycle range. Files are opened once each
    and every requested cycle is read while the file is open; each cycle is
    then reduced to the rank that plots it, so no rank idles.

    NOTES:
    1. 2D ONLY. The XY plane is taken at z index nz_global // 2. With 2 cells
       in z either plane is equivalent, so the choice does not matter.

    2. rho here is the TOTAL charge density (all species combined) as written
       by iPIC3D under moments/rho. It is SIGNED, so it gets a diverging
       colormap centred on zero: white really means rho = 0.
       ###! iPIC3D does not conserve charge by construction (Bacchini 2023,
       ###! Sec. 5, point 2), so a slow drift in the mean is a known property
       ###! of the method rather than necessarily a setup error.

    3. Tiles SHARE boundary planes: the global size is XLEN*(nx-1)+1, not
       XLEN*nx, and every tile after the first drops its leading plane. An
       assembly that assumed disjoint tiles would duplicate every internal
       boundary and produce a plane XLEN columns too wide.

    4. The proc -> (i,j,k) mapping is auto-detected from the six orderings
       iPIC3D might use, scored by how cleanly they tile the domain. With
       ZLEN = 1 several candidates tie (all give k = 0); the tie is harmless
       in 2D but means only the x/y ordering is really being determined. Pass
       --mapping to force one. The occupancy of the assembled plane is checked
       and any gap or overlap is reported, since a wrong mapping would
       otherwise silently scramble every figure.

    5. VLIM_RHO fixes the colour limits so frames are comparable across the
       run. Set it to None to autoscale each frame -- but then a decaying or
       growing signal looks identical in every frame, which defeats the point.
"""

import os
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
os.environ.setdefault("MPLBACKEND", "Agg")

import glob
import argparse

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpi4py import MPI


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

###! ------------------------------------------------------------------
###! Hardcoded parameters (no command-line switches for these)
###! ------------------------------------------------------------------
###! rho is signed; the limit is SYMMETRIC about zero: -VLIM_RHO .. +VLIM_RHO.
###! Its magnitude depends on the particle noise floor, which is hard to guess
###! a priori -- left on autoscale until you have read a value off one frame.
VLIM_RHO   = None        ###! symmetric limit (None -> autoscale per frame)

CMAP       = "seismic"   ###! diverging: rho is signed

DPI        = 100         ###! 100 dpi on an 8x7 in figure is ~800 px wide

DT              = 0.1    ###! from the .inp; used for the time label
TIME_PER_CYCLE  = DT     ###! T = cycle * dt, in units of 1/omega_p

comm.Barrier()


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
    ###! With ZLEN = 1 every candidate gives k = 0, so several mappings tie.
    ###! The tie is harmless in 2D (they tile the XY plane identically) but it
    ###! does mean the detector distinguishes only the x/y ordering. If the
    ###! assembled plane looks scrambled, pass --mapping explicitly.
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
    ###! Tiles share boundary planes, hence (n-1) per tile plus one closing plane
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
    ###! Every tile except the first along an axis drops its leading plane,
    ###! which is a duplicate of its neighbour's trailing plane.
    xs = 0 if i == 0 else 1
    ys = 0 if j == 0 else 1
    zs = 0 if k == 0 else 1
    return xs, ys, zs


def assemble_all_cycles(cycle_names, local_files, rank_to_ijk,
                        M_shape, M_global_shape, z_index):
    """
    Assemble the rho XY plane for every requested cycle in ONE pass over the
    files.

    Each file is opened once; all cycles are read from it before it closes.
    The obvious alternative -- cycle outermost, reopening every file each time
    -- costs n_cycles * n_files opens, and on a parallel filesystem the open
    dominates the cost of the tiny read that follows.

    Each cycle is then reduced to the rank that will PLOT it, so the data
    lands where it is needed and no separate scatter is required. Rank r ends
    up holding cycles r, r+size, r+2*size, ...
    """
    M_nx, M_ny, M_nz = M_global_shape

    ###! Local accumulator, one plane per cycle
    local_rho = {cyc: np.zeros((M_nx, M_ny), dtype=np.float64)
                 for cyc in cycle_names}

    ###! Occupancy is cycle-independent (the same tiles contribute every time),
    ###! so it is accumulated once rather than per cycle.
    local_occ = np.zeros((M_nx, M_ny), dtype=np.int32)

    for fp in local_files:
        pid = proc_id_from_filename(fp)
        i, j, k = rank_to_ijk(pid)

        M_raw_nx, M_raw_ny, M_raw_nz = M_shape
        M_ox, M_oy, M_oz = global_offset_shared(i, j, k, M_shape)
        M_xs, M_ys, M_zs = lower_crop_indices(i, j, k)

        M_x0 = M_ox + M_xs
        M_y0 = M_oy + M_ys
        M_z0 = M_oz + M_zs

        M_use_nx = M_raw_nx - M_xs
        M_use_ny = M_raw_ny - M_ys
        M_use_nz = M_raw_nz - M_zs

        if not (M_z0 <= z_index < M_z0 + M_use_nz):
            continue

        raw_k = z_index - M_oz

        ###! ONE open per file, all cycles read inside it
        with h5py.File(fp, "r") as f:
            for cyc in cycle_names:
                ###! Whole-tile read, then slice in memory. Cheaper than a
                ###! strided hyperslab for these small tiles.
                tile = np.array(f[f"moments/rho/{cyc}"], dtype=np.float64)
                slab = tile[M_xs:, M_ys:, raw_k]
                local_rho[cyc][M_x0:M_x0 + M_use_nx,
                               M_y0:M_y0 + M_use_ny] = slab

        local_occ[M_x0:M_x0 + M_use_nx, M_y0:M_y0 + M_use_ny] += 1

    ###! ---------------- collective reduction ----------------
    ###! Every rank must enter these in the SAME order -- cycle_names is a
    ###! list, so iteration order is deterministic and identical everywhere.
    if rank == 0:
        glob_occ = np.zeros((M_nx, M_ny), dtype=np.int32)
    else:
        glob_occ = None

    my_planes = {}

    for c, cyc in enumerate(cycle_names):
        dest = c % size

        if rank == dest:
            recvbuf = np.zeros((M_nx, M_ny), dtype=np.float64)
        else:
            recvbuf = None

        comm.Reduce(local_rho[cyc], recvbuf, op=MPI.SUM, root=dest)

        if rank == dest:
            my_planes[cyc] = recvbuf

        ###! Free the local copy as soon as it has been reduced -- otherwise
        ###! every rank holds every cycle for the whole run.
        local_rho[cyc] = None

    comm.Reduce(local_occ, glob_occ, op=MPI.SUM, root=0)

    return my_planes, glob_occ


def plot_cycle(cycle_name, rho, outdir, extents):
    ###! extents = (xmin, xmax, ymin, ymax, zmin, zmax) in physical units
    xmin, xmax, ymin, ymax, zmin, zmax = extents

    ###! Symmetric limits about zero so the colormap's white midpoint really
    ###! corresponds to rho = 0 rather than to the middle of the data range.
    if VLIM_RHO is not None:
        v = VLIM_RHO
    else:
        v = float(np.max(np.abs(rho)))
        if v == 0.0:
            v = 1.0e-30   ###! guard against an all-zero panel

    fig, ax = plt.subplots(1, 1, figsize=(8, 7), dpi=DPI)

    im = ax.imshow(rho.T, origin="lower", cmap=CMAP, aspect="equal",
                   norm=Normalize(vmin=-v, vmax=v),
                   extent=[xmin, xmax, ymin, ymax])

    ax.set_title(r"$\rho$ $(X,Y)$", fontsize=16)
    ax.set_xlabel(r"$x\,\omega_p/c$", fontsize=12)
    ax.set_ylabel(r"$y\,\omega_p/c$", fontsize=12)
    ax.tick_params(axis="both", which="major", labelsize=12, length=6)
    fig.colorbar(im, ax=ax, shrink=0.8)

    cycle_number = int(cycle_name.replace("cycle_", ""))
    time_omega = cycle_number * TIME_PER_CYCLE     ###! T = cycle * dt
    fig.suptitle(rf"$T = {time_omega:.0f}\,\omega_p^{{-1}}$", fontsize=18)
    fig.tight_layout()

    outfile = os.path.join(outdir, f"Density_{cycle_name}.png")
    fig.savefig(outfile, bbox_inches="tight")
    plt.close(fig)

    return outfile


parser = argparse.ArgumentParser(description="MPI plotter for the total charge density rho in the XY plane. 2D runs. One invocation renders the whole cycle range; files are opened once and plotting is distributed across ranks.")

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
parser.add_argument("--cycle-end", type=int, default=5000)
parser.add_argument("--cycle-step", type=int, default=100)

parser.add_argument("--outdir", type=str, default=None)
parser.add_argument("--mapping", type=str, default="auto", choices=["auto", "A", "B", "C", "D", "E", "F"])

args = parser.parse_args()

dir_data = args.dir_data
XLEN = args.xlen
YLEN = args.ylen
ZLEN = args.zlen
extents = (args.xmin, args.xmax, args.ymin, args.ymax, args.zmin, args.zmax)

if args.outdir is None:
    outdir = dir_data
else:
    outdir = args.outdir

if rank == 0:
    os.makedirs(outdir, exist_ok=True)

num_expected_files = XLEN * YLEN * ZLEN

###! ---------------- file discovery ----------------
###! Rank 0 does the discovery. Any exception it raises here would leave the
###! other ranks blocked in the bcast below, so failures are flagged via a
###! status broadcast rather than allowed to deadlock.
setup_error = None

if rank == 0:
    try:
        all_files = sorted(glob.glob(os.path.join(dir_data, "proc*.hdf")))

        if len(all_files) == 0:
            raise RuntimeError(f"No proc*.hdf files found in {dir_data}")

        if len(all_files) != num_expected_files:
            print(f"WARNING: expected {num_expected_files} files, "
                  f"found {len(all_files)}", flush=True)

        if args.mapping == "auto":
            map_name, map_score = choose_mapping(all_files, XLEN, YLEN, ZLEN)
        else:
            map_name = args.mapping

    except Exception as exc:
        setup_error = f"{type(exc).__name__}: {exc}"
        all_files = None
        map_name = None

else:
    all_files = None
    map_name = None

setup_error = comm.bcast(setup_error, root=0)

if setup_error is not None:
    if rank == 0:
        print(f"SETUP FAILED: {setup_error}", flush=True)
    comm.Barrier()
    raise SystemExit(1)

all_files = comm.bcast(all_files, root=0)
map_name = comm.bcast(map_name, root=0)

maps = mapping_candidates(XLEN, YLEN, ZLEN)
rank_to_ijk = maps[map_name]

local_files = all_files[rank::size]

requested_cycles = list(range(args.cycle_start, args.cycle_end + 1, args.cycle_step))
first_cycle = f"cycle_{args.cycle_start}"

###! ---------------- probe shape and cycle availability ----------------
probe_error = None

if rank == 0:
    try:
        with h5py.File(all_files[0], "r") as f:
            path = f"moments/rho/{first_cycle}"

            if path not in f:
                raise KeyError(
                    f"Missing dataset in sample file: {path}. Check that "
                    f"FieldOutputTag includes 'rho'."
                )

            M_shape = f[path].shape

            ###! Which requested cycles actually exist. Done ONCE here, before
            ###! any collective. A cycle discovered missing during assembly
            ###! would raise on one rank while the others sat in Reduce,
            ###! hanging the job.
            cycle_names = []
            missing = []

            for cyc in requested_cycles:
                name = f"cycle_{cyc}"

                if f"moments/rho/{name}" in f:
                    cycle_names.append(name)
                else:
                    missing.append(name)

            if not cycle_names:
                raise RuntimeError(
                    f"None of the requested cycles are present in the output. "
                    f"Requested {requested_cycles[0]}..{requested_cycles[-1]} "
                    f"step {args.cycle_step}."
                )

            if missing:
                print(f"WARNING: {len(missing)} requested cycle(s) absent from "
                      f"the output and skipped: {missing[0]} ... {missing[-1]}",
                      flush=True)

        M_global_shape = global_shape_shared(M_shape, XLEN, YLEN, ZLEN)

    except Exception as exc:
        probe_error = f"{type(exc).__name__}: {exc}"
        M_shape = None
        M_global_shape = None
        cycle_names = None

else:
    M_shape = None
    M_global_shape = None
    cycle_names = None

probe_error = comm.bcast(probe_error, root=0)

if probe_error is not None:
    if rank == 0:
        print(f"PROBE FAILED: {probe_error}", flush=True)
    comm.Barrier()
    raise SystemExit(1)

M_shape = comm.bcast(M_shape, root=0)
M_global_shape = comm.bcast(M_global_shape, root=0)
cycle_names = comm.bcast(cycle_names, root=0)

M_nx, M_ny, M_nz = M_global_shape

###! 2D: with 2 cells in z the two planes are equivalent, so which one is
###! taken does not matter.
z_index = M_nz // 2

if not (0 <= z_index < M_nz):
    raise ValueError(f"z_index={z_index} outside rho z range [0,{M_nz - 1}]")

if rank == 0 and M_nz > 3:
    print(f"WARNING: nz_global = {M_nz} — this does not look like a 2D run. "
          f"Only the z = {z_index} plane will be plotted.", flush=True)

###! ---------------- read and reduce ----------------
my_planes, glob_occ = assemble_all_cycles(
    cycle_names, local_files, rank_to_ijk,
    M_shape, M_global_shape, z_index)

###! Occupancy is cycle-independent, so it is checked once, not per frame.
###! A bad mapping here means every figure is scrambled, so it is worth
###! reporting even in an otherwise quiet run.
if rank == 0:
    if glob_occ.min() == 0:
        print(f"WARNING: assembled plane has gaps (occupancy min = 0) — "
              f"check the proc -> (i,j,k) mapping.", flush=True)

    if glob_occ.max() > 1:
        print(f"WARNING: assembled plane has overlaps (occupancy max = "
              f"{glob_occ.max()}) — check the proc -> (i,j,k) mapping.",
              flush=True)

###! ---------------- plot (distributed across ranks) ----------------
my_cycles = [cyc for c, cyc in enumerate(cycle_names) if c % size == rank]

for cycle_name in my_cycles:
    outfile = plot_cycle(cycle_name, my_planes[cycle_name], outdir, extents)

    ###! The only routine output. Lines from different ranks interleave.
    print(f"Saved: {outfile}", flush=True)

comm.Barrier()