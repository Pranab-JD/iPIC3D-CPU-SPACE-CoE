"""
Created on Fri Apr 3 2026

@author: Pranab JD, ChatGPT

Description: Convert iPIC3D proc*.hdf files into one VTK file per current sheet and per time step.

This script:
    - reads proc*.hdf files in parallel with MPI,
    - reconstructs the global 3D subvolume for 2 user-defined current sheets,
    - writes one VTK legacy STRUCTURED_POINTS file per CS per time step,
    - stores:
            E   = (Ex, Ey, Ez)
            B   = (Bx, By, Bz)
            J   = (Jx, Jy, Jz)   summed over species if needed
            rho = scalar         summed over species if needed

Usage:
    srun -n 32 python3 HDF_to_VTK.py \
        /path/to/data /path/to/out \
        0 20000 500 \
        64 2 64 \
        --cs1 40 52 \
        --cs2 74 86 \
        --dtype float32 \
        --stride 3 1 3 \        ## Every 3 points along X and Z, every point along Y
        --write B               ## choices=["all", "B", "E", "rho", "J", "pressure"]

Output:
    outdir/
        CS1/
            CS1_cycle_0.vtk
            CS1_cycle_500.vtk
        ...
        CS2/
            CS2_cycle_0.vtk
            CS2_cycle_500.vtk
        ...
"""

import numpy as np
import os, re, glob
import argparse, h5py
from mpi4py import MPI

#! ============================================================
#! MPI
#! ============================================================

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


#! ============================================================
#! Helpers from the original logic / lightly adapted
#! ============================================================

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
    if "moments" not in f:
        return []
    out = []
    for k in f["moments"].keys():
        m = re.match(r"species_(\d+)$", k)
        if m:
            out.append(int(m.group(1)))
    return sorted(out)


def cycle_name(t: int) -> str:
    return f"cycle_{t}"


def get_field_path(f: h5py.File, quantity: str, cycle: str):
    p = f"fields/{quantity}/{cycle}"
    return p if p in f else None


def get_moment_path(f: h5py.File, quantity: str, cycle: str, species=None):
    p_flat = f"moments/{quantity}/{cycle}"
    if p_flat in f:
        return p_flat

    if species is not None:
        p_species = f"moments/species_{species}/{quantity}/{cycle}"
        if p_species in f:
            return p_species

    return None


def vtk_dtype_string(arr: np.ndarray) -> str:
    if arr.dtype == np.float32:
        return "float"
    if arr.dtype == np.float64:
        return "double"
    raise ValueError(f"Unsupported dtype {arr.dtype}")


def stride_array(arr, sx, sy, sz):
    return arr[::sx, ::sy, ::sz]


def is_pressure_name(name: str) -> bool:
    lname = name.lower()
    if lname == "pressure":
        return True
    if lname.startswith("p"):
        return True
    return False


#! ============================================================
#! Dataset discovery
#! ============================================================

FIELD_COMPONENTS = ["Bx", "By", "Bz", "Ex", "Ey", "Ez"]
MOMENT_COMPONENTS = ["Jx", "Jy", "Jz", "rho"]

def inspect_available_quantities(f: h5py.File, cyc: str):
    """
    Inspect a file for one cycle and return availability info.
    """
    available_fields = set()
    available_flat_moments = set()
    available_species_moments = {}

    for q in FIELD_COMPONENTS:
        if get_field_path(f, q, cyc) is not None:
            available_fields.add(q)

    if "moments" in f:
        for q in f["moments"].keys():
            if re.match(r"species_\d+$", q):
                continue
            p = get_moment_path(f, q, cyc, species=None)
            if p is not None:
                available_flat_moments.add(q)

    species = discover_species(f)
    for s in species:
        qset = set()
        grp = f.get(f"moments/species_{s}", None)
        if grp is None:
            continue
        for q in grp.keys():
            p = get_moment_path(f, q, cyc, species=s)
            if p is not None:
                qset.add(q)
        if qset:
            available_species_moments[s] = qset

    return available_fields, available_flat_moments, available_species_moments


def merge_availability(all_field_sets, all_flat_sets, all_species_dicts):
    """
    Use union over files: if something exists anywhere, we treat it as available globally.
    Missing pieces in some files will simply be skipped locally.
    """
    fields = sorted(set().union(*all_field_sets)) if all_field_sets else []
    flat_moments = sorted(set().union(*all_flat_sets)) if all_flat_sets else []

    species_merged = {}
    for d in all_species_dicts:
        for s, qset in d.items():
            if s not in species_merged:
                species_merged[s] = set()
            species_merged[s].update(qset)

    for s in species_merged:
        species_merged[s] = sorted(species_merged[s])

    return fields, flat_moments, species_merged


#! ============================================================
#! VTK writing
#! ============================================================

def write_legacy_vtk_structured_points(filename, arrays, origin=(0.0, 0.0, 0.0), spacing=(1.0, 1.0, 1.0)):
    """
    arrays: dict of named arrays with shape (nx, ny, nz)
            supported keys:
              scalar names like "rho"
              vector bundles must be passed already grouped:
                 arrays["B"] = (Bx, By, Bz)
                 arrays["E"] = (Ex, Ey, Ez)
                 arrays["J"] = (Jx, Jy, Jz)
    """
    sample = None
    for v in arrays.values():
        if isinstance(v, tuple):
            sample = v[0]
            break
        else:
            sample = v
            break

    if sample is None:
        raise RuntimeError("No arrays available to write to VTK.")

    nx, ny, nz = sample.shape
    npts = nx * ny * nz
    dtype_name = vtk_dtype_string(sample)

    def flat(a):
        return np.ravel(a, order="F")

    with open(filename, "w") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("iPIC3D CS extract\n")
        f.write("ASCII\n")
        f.write("DATASET STRUCTURED_POINTS\n")
        f.write(f"DIMENSIONS {nx} {ny} {nz}\n")
        f.write(f"ORIGIN {origin[0]} {origin[1]} {origin[2]}\n")
        f.write(f"SPACING {spacing[0]} {spacing[1]} {spacing[2]}\n")
        f.write(f"POINT_DATA {npts}\n")

        for vec_name in ["E", "B", "J"]:
            if vec_name in arrays:
                ax, ay, az = arrays[vec_name]
                f.write(f"VECTORS {vec_name} {dtype_name}\n")
                axf, ayf, azf = flat(ax), flat(ay), flat(az)
                for a, b, c in zip(axf, ayf, azf):
                    f.write(f"{a:.8e} {b:.8e} {c:.8e}\n")

        for key, val in arrays.items():
            if isinstance(val, tuple):
                continue
            f.write(f"SCALARS {key} {dtype_name} 1\n")
            f.write("LOOKUP_TABLE default\n")
            for a in flat(val):
                f.write(f"{a:.8e}\n")


#! ============================================================
#! Read one proc tile quantity
#! ============================================================

def read_total_moment_if_available(f: h5py.File, quantity: str, cyc: str, species_avail):
    """
    Return:
      ndarray if available
      None if not available
    """
    p_flat = get_moment_path(f, quantity, cyc, species=None)
    if p_flat is not None:
        return np.asarray(f[p_flat][...])

    if not species_avail:
        return None

    acc = None
    found = False
    for s in species_avail:
        p = get_moment_path(f, quantity, cyc, species=s)
        if p is None:
            continue
        arr = np.asarray(f[p][...])
        acc = arr if acc is None else (acc + arr)
        found = True

    return acc if found else None


#! ============================================================
#! Main parser
#! ============================================================

parser = argparse.ArgumentParser(description="Convert proc*.hdf to VTK for 2 user-defined CS slabs")
parser.add_argument("dir_data", type=str)
parser.add_argument("outdir", type=str)
parser.add_argument("t_start", type=int)
parser.add_argument("t_end", type=int)
parser.add_argument("t_step", type=int)
parser.add_argument("xlen", type=int)
parser.add_argument("ylen", type=int)
parser.add_argument("zlen", type=int)
parser.add_argument("--cs1", nargs=2, type=int, required=True, metavar=("YMIN", "YMAX"))
parser.add_argument("--cs2", nargs=2, type=int, required=True, metavar=("YMIN", "YMAX"))
parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])
parser.add_argument("--origin", nargs=3, type=float, default=[0.0, 0.0, 0.0])
parser.add_argument("--spacing", nargs=3, type=float, default=[1.0, 1.0, 1.0])
parser.add_argument("--stride", nargs=3, type=int, default=[1, 1, 1], metavar=("SX", "SY", "SZ"))
parser.add_argument("--write", nargs="+", default=["all"], choices=["all", "B", "E", "rho", "J", "pressure"])
args = parser.parse_args()

work_dtype = np.float32 if args.dtype == "float32" else np.float64
mpi_dtype = MPI.FLOAT if args.dtype == "float32" else MPI.DOUBLE

XLEN, YLEN, ZLEN = args.xlen, args.ylen, args.zlen
cs1_ymin, cs1_ymax = args.cs1
cs2_ymin, cs2_ymax = args.cs2
origin_global = tuple(args.origin)
spacing = tuple(args.spacing)
sx, sy, sz = args.stride

if sx <= 0 or sy <= 0 or sz <= 0:
    raise ValueError("Stride factors must be positive integers.")

write_all = "all" in args.write
want_B = write_all or ("B" in args.write)
want_E = write_all or ("E" in args.write)
want_J = write_all or ("J" in args.write)
want_rho = write_all or ("rho" in args.write)
want_pressure = write_all or ("pressure" in args.write)


#! ============================================================
#! Discover files
#! ============================================================

if rank == 0:
    all_files = sorted(glob.glob(os.path.join(args.dir_data, "proc*.hdf")))
    if not all_files:
        raise RuntimeError(f"No proc*.hdf found in {args.dir_data}")
else:
    all_files = None

all_files = comm.bcast(all_files, root=0)
local_files = all_files[rank::size]


#! ============================================================
#! Probe tile shape and availability
#! ============================================================

tile_shape = None
local_field_sets = []
local_flat_sets = []
local_species_dicts = []

probe_cycle = cycle_name(args.t_start)

for fp in local_files:
    try:
        with h5py.File(fp, "r") as f:
            af, amf, ams = inspect_available_quantities(f, probe_cycle)
            local_field_sets.append(af)
            local_flat_sets.append(amf)
            local_species_dicts.append(ams)

            if tile_shape is None:
                found_shape = False
                for q in FIELD_COMPONENTS:
                    p = get_field_path(f, q, probe_cycle)
                    if p is not None:
                        tile_shape = tuple(f[p].shape)
                        found_shape = True
                        break

                if not found_shape:
                    for q in amf:
                        p = get_moment_path(f, q, probe_cycle, species=None)
                        if p is not None:
                            tile_shape = tuple(f[p].shape)
                            found_shape = True
                            break

                if not found_shape:
                    species = discover_species(f)
                    for s in species:
                        grp = f.get(f"moments/species_{s}", None)
                        if grp is None:
                            continue
                        for q in grp.keys():
                            p = get_moment_path(f, q, probe_cycle, species=s)
                            if p is not None:
                                tile_shape = tuple(f[p].shape)
                                found_shape = True
                                break
                        if found_shape:
                            break
    except Exception:
        continue

tile_shape_all = comm.gather(tile_shape, root=0)
field_sets_all = comm.gather(local_field_sets, root=0)
flat_sets_all = comm.gather(local_flat_sets, root=0)
species_dicts_all = comm.gather(local_species_dicts, root=0)

if rank == 0:
    valid_shapes = [s for s in tile_shape_all if s is not None]
    if not valid_shapes:
        raise RuntimeError("Could not determine tile shape.")

    tile_shape = valid_shapes[0]

    gathered_field_sets = []
    gathered_flat_sets = []
    gathered_species_dicts = []

    for lst in field_sets_all:
        gathered_field_sets.extend(lst)
    for lst in flat_sets_all:
        gathered_flat_sets.extend(lst)
    for lst in species_dicts_all:
        gathered_species_dicts.extend(lst)

    available_fields, available_flat_moments, available_species_moments = merge_availability(
        gathered_field_sets,
        gathered_flat_sets,
        gathered_species_dicts)

    moment_union = set(available_flat_moments)
    for s, qlist in available_species_moments.items():
        moment_union.update(qlist)

    pressure_names = sorted([q for q in moment_union if is_pressure_name(q) and q not in ["Jx", "Jy", "Jz", "rho"]])

    has_B = want_B and all(q in available_fields for q in ["Bx", "By", "Bz"])
    has_E = want_E and all(q in available_fields for q in ["Ex", "Ey", "Ez"])
    has_J = want_J and all(q in moment_union for q in ["Jx", "Jy", "Jz"])
    has_rho = want_rho and ("rho" in moment_union)
    has_pressure = want_pressure and (len(pressure_names) > 0)

    availability = {"fields": available_fields,
                    "flat_moments": available_flat_moments,
                    "species_moments": available_species_moments,
                    "pressure_names": pressure_names,
                    "write_B": has_B,
                    "write_E": has_E,
                    "write_J": has_J,
                    "write_rho": has_rho,
                    "write_pressure": has_pressure}

    print("\nAvailable fields:", available_fields, flush=True)
    print("Available flat moments:", available_flat_moments, flush=True)
    print("Available per-species moments:", available_species_moments, flush=True)
    print("Available pressure-like moments:", pressure_names, flush=True)
    print(f"Will write B        : {has_B}", flush=True)
    print(f"Will write E        : {has_E}", flush=True)
    print(f"Will write J        : {has_J}", flush=True)
    print(f"Will write rho      : {has_rho}", flush=True)
    print(f"Will write pressure : {has_pressure}", flush=True)
else:
    tile_shape = None
    availability = None

tile_shape = comm.bcast(tile_shape, root=0)
availability = comm.bcast(availability, root=0)

nx_tile, ny_tile, nz_tile = tile_shape

nx_cell = nx_tile - 1
ny_cell = ny_tile - 1
nz_cell = nz_tile - 1

nx_global = XLEN * nx_cell + 1
ny_global = YLEN * ny_cell + 1
nz_global = ZLEN * nz_cell + 1


#! ============================================================
#! Infer mapping
#! ============================================================

best_name = None
if rank == 0:
    proc_ids = [proc_id_from_filename(fp) for fp in all_files]
    best_stats = None

    for name, fn in mapping_candidates(XLEN, YLEN, ZLEN):
        occ3d = np.zeros((nx_global, ny_global, nz_global), dtype=np.int16)
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
        raise RuntimeError("Could not infer proc mapping.")

best_name = comm.bcast(best_name, root=0)
maps = {n: fn for n, fn in mapping_candidates(XLEN, YLEN, ZLEN)}
pid_to_ijk = maps[best_name]


#! ============================================================
#! Assembly
#! ============================================================

def allocate_local_cs_buffers(ny_cs, availability):
    shape = (nx_global, ny_cs, nz_global)
    out = {}

    if availability["write_B"]:
        out["Bx"] = np.zeros(shape, dtype=work_dtype)
        out["By"] = np.zeros(shape, dtype=work_dtype)
        out["Bz"] = np.zeros(shape, dtype=work_dtype)

    if availability["write_E"]:
        out["Ex"] = np.zeros(shape, dtype=work_dtype)
        out["Ey"] = np.zeros(shape, dtype=work_dtype)
        out["Ez"] = np.zeros(shape, dtype=work_dtype)

    if availability["write_J"]:
        out["Jx"] = np.zeros(shape, dtype=work_dtype)
        out["Jy"] = np.zeros(shape, dtype=work_dtype)
        out["Jz"] = np.zeros(shape, dtype=work_dtype)

    if availability["write_rho"]:
        out["rho"] = np.zeros(shape, dtype=work_dtype)

    if availability["write_pressure"]:
        for q in availability["pressure_names"]:
            out[q] = np.zeros(shape, dtype=work_dtype)

    return out


def assemble_one_cs_for_cycle(cyc, y_min_cs, y_max_cs, availability):
    ny_cs = y_max_cs - y_min_cs + 1
    local = allocate_local_cs_buffers(ny_cs, availability)
    found_any_local = False

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

        tile_y_min = gy0
        tile_y_max = gy0 + ny_use - 1

        ov_y_min = max(tile_y_min, y_min_cs)
        ov_y_max = min(tile_y_max, y_max_cs)

        if ov_y_min > ov_y_max:
            continue

        ly0 = ov_y_min - tile_y_min
        ly1 = ov_y_max - tile_y_min + 1

        cy0 = ov_y_min - y_min_cs
        cy1 = ov_y_max - y_min_cs + 1

        try:
            with h5py.File(fp, "r") as f:
                tmp = {}

                if availability["write_B"]:
                    for q in ["Bx", "By", "Bz"]:
                        p = get_field_path(f, q, cyc)
                        if p is not None:
                            tmp[q] = np.asarray(f[p][...], dtype=work_dtype)[xs:, ys:, zs:]

                if availability["write_E"]:
                    for q in ["Ex", "Ey", "Ez"]:
                        p = get_field_path(f, q, cyc)
                        if p is not None:
                            tmp[q] = np.asarray(f[p][...], dtype=work_dtype)[xs:, ys:, zs:]

                if availability["write_J"]:
                    for q in ["Jx", "Jy", "Jz"]:
                        arr = read_total_moment_if_available(f, q, cyc, list(availability["species_moments"].keys()))
                        if arr is not None:
                            tmp[q] = np.asarray(arr, dtype=work_dtype)[xs:, ys:, zs:]

                if availability["write_rho"]:
                    arr = read_total_moment_if_available(f, "rho", cyc, list(availability["species_moments"].keys()))
                    if arr is not None:
                        tmp["rho"] = np.asarray(arr, dtype=work_dtype)[xs:, ys:, zs:]

                if availability["write_pressure"]:
                    for q in availability["pressure_names"]:
                        arr = read_total_moment_if_available(f, q, cyc, list(availability["species_moments"].keys()))
                        if arr is not None:
                            tmp[q] = np.asarray(arr, dtype=work_dtype)[xs:, ys:, zs:]

                if not tmp:
                    continue

                found_any_local = True

                for q, arr in tmp.items():
                    block = arr[:, ly0:ly1, :]
                    local[q][gx0:gx0+nx_use, cy0:cy1, gz0:gz0+nz_use] += block

        except Exception as e:
            raise RuntimeError(f"Rank {rank}: failed on {fp}, {cyc}: {e}")

    found_any_global = comm.allreduce(found_any_local, op=MPI.LOR)
    if not found_any_global:
        return None

    global_out = None
    if rank == 0:
        global_out = allocate_local_cs_buffers(ny_cs, availability)

    for q in local.keys():
        recvbuf = global_out[q] if rank == 0 else None
        comm.Reduce([local[q], mpi_dtype], [recvbuf, mpi_dtype], op=MPI.SUM, root=0)

    return global_out


#! ============================================================
#! Write summary file once
#! ============================================================

if rank == 0:
    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(os.path.join(args.outdir, "CS1"), exist_ok=True)
    os.makedirs(os.path.join(args.outdir, "CS2"), exist_ok=True)

    with open(os.path.join(args.outdir, "available_datasets.txt"), "w") as f:
        f.write("Available fields:\n")
        for q in availability["fields"]:
            f.write(f"  {q}\n")

        f.write("\nAvailable flat moments:\n")
        for q in availability["flat_moments"]:
            f.write(f"  {q}\n")

        f.write("\nAvailable per-species moments:\n")
        for s, qlist in sorted(availability["species_moments"].items()):
            f.write(f"  species_{s}:\n")
            for q in qlist:
                f.write(f"    {q}\n")

        f.write("\nAvailable pressure-like moments:\n")
        for q in availability["pressure_names"]:
            f.write(f"  {q}\n")

        f.write("\nWill be written to VTK:\n")
        if availability["write_B"]:
            f.write("  B = (Bx, By, Bz)\n")
        if availability["write_E"]:
            f.write("  E = (Ex, Ey, Ez)\n")
        if availability["write_J"]:
            f.write("  J = (Jx, Jy, Jz)\n")
        if availability["write_rho"]:
            f.write("  rho\n")
        if availability["write_pressure"]:
            for q in availability["pressure_names"]:
                f.write(f"  {q}\n")


#! ============================================================
#! Main loop
#! ============================================================

for t in range(args.t_start, args.t_end + 1, args.t_step):
    cyc = cycle_name(t)

    if rank == 0:
        print(f"\nProcessing {cyc}", flush=True)

    for cs_name, (ymin, ymax) in [("CS1", (cs1_ymin, cs1_ymax)), ("CS2", (cs2_ymin, cs2_ymax))]:
        slab = assemble_one_cs_for_cycle(cyc, ymin, ymax, availability)

        if rank == 0 and slab is not None:
            arrays_to_write = {}

            if availability["write_E"] and all(q in slab for q in ["Ex", "Ey", "Ez"]):
                arrays_to_write["E"] = (
                    stride_array(slab["Ex"], sx, sy, sz),
                    stride_array(slab["Ey"], sx, sy, sz),
                    stride_array(slab["Ez"], sx, sy, sz)
                )

            if availability["write_B"] and all(q in slab for q in ["Bx", "By", "Bz"]):
                arrays_to_write["B"] = (
                    stride_array(slab["Bx"], sx, sy, sz),
                    stride_array(slab["By"], sx, sy, sz),
                    stride_array(slab["Bz"], sx, sy, sz)
                )

            if availability["write_J"] and all(q in slab for q in ["Jx", "Jy", "Jz"]):
                arrays_to_write["J"] = (
                    stride_array(slab["Jx"], sx, sy, sz),
                    stride_array(slab["Jy"], sx, sy, sz),
                    stride_array(slab["Jz"], sx, sy, sz)
                )

            if availability["write_rho"] and "rho" in slab:
                arrays_to_write["rho"] = stride_array(slab["rho"], sx, sy, sz)

            if availability["write_pressure"]:
                for q in availability["pressure_names"]:
                    if q in slab:
                        arrays_to_write[q] = stride_array(slab[q], sx, sy, sz)

            if not arrays_to_write:
                print(f"Skipping {cs_name} {cyc}: no datasets available to write.", flush=True)
                continue

            y0_phys = origin_global[1] + ymin * spacing[1]
            vtkfile = os.path.join(args.outdir, cs_name, f"{cs_name}_{cyc}.vtk")

            write_legacy_vtk_structured_points(
                vtkfile,
                arrays_to_write,
                origin=(origin_global[0], y0_phys, origin_global[2]),
                spacing=(spacing[0] * sx, spacing[1] * sy, spacing[2] * sz)
            )
            print(f"Wrote {vtkfile}", flush=True)

comm.Barrier()
if rank == 0:
    print("\nDone.", flush=True)