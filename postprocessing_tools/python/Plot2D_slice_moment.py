"""
Created on Tue Feb 17 17:21 2026

@author: Pranab JD, ChatGPT

Description: Assemble & plot a 2D slice of MOMENTS (rho or Jx/Jy/Jz) from per-proc iPIC3D HDF tiles.

"""

import os, glob, argparse, re
import numpy as np
import h5py
from mpi4py import MPI
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1 import make_axes_locatable

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


# ----------------------------- Helpers -----------------------------
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


# ----------------------------- Args -----------------------------
parser = argparse.ArgumentParser(description="Plot 2D slice of summed moments (rho/Jx/Jy/Jz) from iPIC3D proc*.hdf tiles")

parser.add_argument("dir_data",   type=str, help="Directory containing proc*.hdf")
parser.add_argument("time_cycle", type=str, help="Cycle group name, e.g. cycle_19500")
parser.add_argument("xlen", type=int, help="Simulation XLEN")
parser.add_argument("ylen", type=int, help="Simulation YLEN")
parser.add_argument("zlen", type=int, help="Simulation ZLEN")

parser.add_argument("--quantity", type=str, required=True, choices=["rho","Jx","Jy","Jz"],
                    help="Which moments quantity to plot (summed over species)")

parser.add_argument("--axis", type=str, default="z", choices=["x","y","z"], help="Slice axis")
parser.add_argument("--index", type=int, default=0, help="Global nodal index along axis")

parser.add_argument("--species", type=str, default="all",
                    help="Species selection: 'all' or comma list like '0,1,3'")

parser.add_argument("--xmin", type=float, default=None)
parser.add_argument("--xmax", type=float, default=None)
parser.add_argument("--xticks", type=int, default=6)
parser.add_argument("--ymin", type=float, default=None)
parser.add_argument("--ymax", type=float, default=None)
parser.add_argument("--yticks", type=int, default=6)

parser.add_argument("--cmap", type=str, default="seismic")
parser.add_argument("--out", type=str, default="", help="Output PNG (default auto)")
parser.add_argument("--cb_decimals", type=int, default=2, help="Colorbar decimals")
args = parser.parse_args()

dir_data   = args.dir_data
time_cycle = args.time_cycle
XLEN, YLEN, ZLEN = args.xlen, args.ylen, args.zlen
axis = args.axis.lower()
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


# ----------------------------- Probe tile shape + species list -----------------------------
tile_shape = None
species_avail = None

if local_files:
    with h5py.File(local_files[0], "r") as f:
        # determine species list from file
        species_avail = discover_species(f)
        if not species_avail:
            raise RuntimeError("No /moments/species_* groups found in file.")

        # choose one species just to probe shape
        s0 = species_avail[0]
        if dset_path(s0, quantity, time_cycle) not in f:
            raise RuntimeError(f"Dataset not found: /{dset_path(s0, quantity, time_cycle)}")

        tile_shape = tuple(f[dset_path(s0, quantity, time_cycle)].shape)

tile_shape_all = comm.gather(tile_shape, root=0)
species_all    = comm.gather(species_avail, root=0)

if rank == 0:
    tile_shape = next(s for s in tile_shape_all if s is not None)
    # intersect species lists across ranks (should be identical; this is just safe)
    sp_lists = [set(s) for s in species_all if s is not None]
    species_avail = sorted(set.intersection(*sp_lists)) if sp_lists else []
    print("Detected tile shape:", tile_shape)
    print("Available species:", species_avail)
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

global_len = {"x": nx_global, "y": ny_global, "z": nz_global}[axis]
if not (0 <= slice_g < global_len):
    raise ValueError(f"slice index {slice_g} out of range for axis {axis} (0..{global_len-1})")

# plane shape in INDEX space
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
                if not (gz0 <= slice_g < gz0 + nz_use): continue
                Occ[gx0:gx0+nx_use, gy0:gy0+ny_use] += 1
            elif axis == "y":
                if not (gy0 <= slice_g < gy0 + ny_use): continue
                Occ[gx0:gx0+nx_use, gz0:gz0+nz_use] += 1
            else:
                if not (gx0 <= slice_g < gx0 + nx_use): continue
                Occ[gy0:gy0+ny_use, gz0:gz0+nz_use] += 1

        if bad:
            continue

        score, gaps, overlaps, maxv = score_occupancy(Occ)
        if best_stats is None or score < best_stats[0]:
            best_stats = (score, gaps, overlaps, maxv)
            best_name = name

    if best_name is None:
        raise RuntimeError("Could not infer proc->(i,j,k) mapping from filenames.")
    print(f"Selected mapping {best_name} (score={best_stats[0]}, gaps={best_stats[1]}, overlaps={best_stats[2]}, Occ.max={best_stats[3]})")

best_name = comm.bcast(best_name, root=0)
maps = {n: fn for n, fn in mapping_candidates(XLEN, YLEN, ZLEN)}
pid_to_ijk = maps[best_name]


# ----------------------------- Assemble summed moments plane -----------------------------
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

    # does this tile intersect the requested slice?
    if axis == "z":
        if not (gz0 <= slice_g < gz0 + nz_use): continue
        loc_k_orig = (slice_g - gz0) + zs
        with h5py.File(fp, "r") as f:
            slab_sum = None
            for s in species_sel:
                path = dset_path(s, quantity, time_cycle)
                arr = np.array(f[path][:, :, loc_k_orig], dtype=np.float64)
                slab_sum = arr if slab_sum is None else (slab_sum + arr)
        slab_sum = slab_sum[xs:, ys:]
        local_plane[gx0:gx0+nx_use, gy0:gy0+ny_use] += slab_sum
        local_occ[gx0:gx0+nx_use, gy0:gy0+ny_use]   += 1

    elif axis == "y":
        if not (gy0 <= slice_g < gy0 + ny_use): continue
        loc_j_orig = (slice_g - gy0) + ys
        with h5py.File(fp, "r") as f:
            slab_sum = None
            for s in species_sel:
                path = dset_path(s, quantity, time_cycle)
                arr = np.array(f[path][:, loc_j_orig, :], dtype=np.float64)
                slab_sum = arr if slab_sum is None else (slab_sum + arr)
        slab_sum = slab_sum[xs:, zs:]
        local_plane[gx0:gx0+nx_use, gz0:gz0+nz_use] += slab_sum
        local_occ[gx0:gx0+nx_use, gz0:gz0+nz_use]   += 1

    else:  # axis == "x"
        if not (gx0 <= slice_g < gx0 + nx_use): continue
        loc_i_orig = (slice_g - gx0) + xs
        with h5py.File(fp, "r") as f:
            slab_sum = None
            for s in species_sel:
                path = dset_path(s, quantity, time_cycle)
                arr = np.array(f[path][loc_i_orig, :, :], dtype=np.float64)
                slab_sum = arr if slab_sum is None else (slab_sum + arr)
        slab_sum = slab_sum[ys:, zs:]
        local_plane[gy0:gy0+ny_use, gz0:gz0+nz_use] += slab_sum
        local_occ[gy0:gy0+ny_use, gz0:gz0+nz_use]   += 1


# ----------------------------- Reduce to root -----------------------------
plane = None
occ = None
if rank == 0:
    plane = np.zeros(plane_shape, dtype=np.float64)
    occ   = np.zeros(plane_shape, dtype=np.int32)

comm.Reduce(local_plane, plane, op=MPI.SUM, root=0)
comm.Reduce(local_occ,   occ,   op=MPI.SUM, root=0)

# ----------------------------- Plot on root -----------------------------
if rank == 0:
    omin, omax = int(occ.min()), int(occ.max())
    print(f"Occupancy min/max: {omin}/{omax}")
    if omin == 0:
        print("WARNING: gaps remain (occ==0). Check slice index, mapping, or ghost-layer assumptions.")
    if omax > 1:
        print("WARNING: overlaps (occ>1). Might indicate >1 ghost layer or different centering.")

    out_plane = plane.copy()
    mask = (occ > 0)
    out_plane[mask] = plane[mask] / occ[mask]

    # ticks: positions in index space, labels in physical space
    nx, ny = out_plane.shape

    # defaults: if xmin/xmax not passed, use index range
    xmin = 0.0 if args.xmin is None else args.xmin
    xmax = float(nx-1) if args.xmax is None else args.xmax
    ymin = 0.0 if args.ymin is None else args.ymin
    ymax = float(ny-1) if args.ymax is None else args.ymax

    # Limits for color bar
    vmax = max(np.max(np.abs(out_plane)), 1e-6)
    vmin = -vmax

    x_pos = np.linspace(0, nx-1, args.xticks)
    y_pos = np.linspace(0, ny-1, args.yticks)
    x_lab = np.linspace(xmin, xmax, args.xticks)
    y_lab = np.linspace(ymin, ymax, args.yticks)

    cyc = cycle_to_number(time_cycle)
    cycle_number = int(int(cyc)/10)

    fig, ax = plt.subplots(figsize=(6.5, 8.0), dpi=250)

    im = ax.imshow(out_plane.T, origin="lower", aspect="auto", cmap=args.cmap, vmin=vmin, vmax=vmax)

    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"{v:.0f}" for v in x_lab])
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{v:.0f}" for v in y_lab])

    ax.tick_params(axis='x', which='major', labelsize=16, length=8)
    ax.tick_params(axis='y', which='major', labelsize=16, length=8)

    ax.set_xlabel("X", fontsize=16)
    ax.set_ylabel("Y", fontsize=16)

    # Title
    ax.set_title(f"{cycle_number} $\\omega^{{-1}}$", fontsize=18)

    # Colorbar matched to image height
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.10)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(f"{quantity}  ({axis}={slice_g})", fontsize=18)
    cbar.ax.tick_params(labelsize=16)
    cbar.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter(f"%.{args.cb_decimals}f"))

    plt.tight_layout()

    if args.out.strip():
        out_png = args.out
    else:
        out_png = os.path.join(f"{quantity}_slice_{axis}{slice_g}_{time_cycle}.png")

    plt.savefig(out_png)
    plt.close()
    print("Wrote:", out_png)
    print()