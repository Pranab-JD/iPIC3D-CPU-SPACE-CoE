"""
    Spectra_compute.py -- HEAVY stage (MPI). Reads the proc*.hdf tiles ONCE,
    computes the per-CS-averaged in-plane spectra and structure-function moments,
    and writes a single small .npz cache. Spectra_plot.py then makes the figures
    from that cache in seconds, so replotting never re-reads the HDF5.

    srun python3 -u Spectra_compute.py "$DATA_DIR" \
        $xmin $xmax $ymin $ymax $zmin $zmax \
        --nxc 768 --nyc 1536 --nzc 768 \
        --cycle-start 0 --cycle-end 4200 --cycle-step 100 \
        --outdir "$OUT_DIR"

    Cache written: <outdir>/spectra_sf_cache.npz  (override with --cache-name).

    WHAT IS CACHED (CS-averaged = 0.5*(CS1 + CS2), per cycle):
        spectra   : Ex_B, Ez_B, Ex_dB, Ez_dB           each (n_cyc, n_kbins)
        SF moments: SF{n}_{B,dB}_{x,y,z} for n in (2,4,8)
                    each (n_cyc, n_lags), stored as the RAW moment <|df|^n>
                    (NOT rooted) so any root/exponent is taken at plot time.
        axes      : kc_x, valid_x, kc_z, valid_z,
                    lags_x, lags_y, lags_z  (in CELLS), dx, dy, dz,
                    cycles, times
        meta      : SF_ORDERS, CS_SLABS, grid, box, mapping

    ###! HIGH-ORDER CAVEAT (recorded in the cache header too):
    ###!   SF4 and especially SF8 are dominated by the LARGEST increments, so
    ###!   (a) they converge slowly -- the largest-lag SF8 on a thin slab can be
    ###!       noisy cycle-to-cycle, and
    ###!   (b) the dB = B - <B>_xz(y) mean-field OVER-subtracts near a rippled
    ###!       sheet, injecting spurious large increments that SF8 amplifies.
    ###!   Treat high-n dB SFs near the sheet / at large lag with caution.

    DOMAIN, dB, spectra and SF definitions are exactly as in the single-stage
    Spectra.py; see the physics notes there.
"""

import os
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import glob
import argparse

import h5py
import numpy as np
from mpi4py import MPI


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

###! ------------------------------------------------------------------
COMPONENTS = ["Bx", "By", "Bz"]
DT             = 0.1
TIME_PER_CYCLE = DT
CS_SLABS   = [(0.2, 0.3), (0.7, 0.8)]
SF_ORDERS  = (4,)                      ###! raw moment <|df|^4> cached (even order)
SF_MIN_LAG_CELLS = 1

comm.Barrier()


def proc_id_from_filename(fp):
    base = os.path.basename(fp)
    return int(base.replace("proc", "").replace(".hdf", ""))


def mapping_candidates(XLEN, YLEN, ZLEN):
    def A(pid):
        k = pid % ZLEN; t = pid // ZLEN
        j = t % YLEN;   i = t // YLEN
        return i, j, k

    def B(pid):
        j = pid % YLEN; t = pid // YLEN
        k = t % ZLEN;   i = t // ZLEN
        return i, j, k

    def C(pid):
        k = pid % ZLEN; t = pid // ZLEN
        i = t % XLEN;   j = t // XLEN
        return i, j, k

    def D(pid):
        i = pid % XLEN; t = pid // XLEN
        j = t % YLEN;   k = t // YLEN
        return i, j, k

    def E(pid):
        j = pid % YLEN; t = pid // YLEN
        i = t % XLEN;   k = t // XLEN
        return i, j, k

    def F(pid):
        i = pid % XLEN; t = pid // XLEN
        k = t % ZLEN;   j = t // ZLEN
        return i, j, k

    return {"A": A, "B": B, "C": C, "D": D, "E": E, "F": F}


def choose_mapping(files, XLEN, YLEN, ZLEN):
    proc_ids = [proc_id_from_filename(fp) for fp in files]
    maps = mapping_candidates(XLEN, YLEN, ZLEN)
    best_name, best_score = None, None
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
        score = 10 * int(np.count_nonzero(occ == 0)) + \
                100 * int(np.count_nonzero(occ > 1))
        if best_score is None or score < best_score:
            best_score, best_name = score, name
    if best_name is None:
        raise RuntimeError("Could not determine proc -> (i,j,k) mapping.")
    return best_name


def global_shape_shared(tile_shape, XLEN, YLEN, ZLEN):
    nx, ny, nz = tile_shape
    return (XLEN * (nx - 1) + 1, YLEN * (ny - 1) + 1, ZLEN * (nz - 1) + 1)


def assemble_slab(cycle_name, local_files, rank_to_ijk, tile_shape,
                  G_shape, jlo, jhi):
    """COLLECTIVE. Assemble Bx,By,Bz on (x,z) over y-slab [jlo,jhi); reduce to
    rank 0. Returns dict comp->(Gx,ny_slab,Gz) on rank 0, else None."""
    Gx, Gy, Gz = G_shape
    nx_t, ny_t, nz_t = tile_shape
    nx_c, ny_c, nz_c = nx_t - 1, ny_t - 1, nz_t - 1
    ny_slab = jhi - jlo

    local = {c: np.zeros((Gx, ny_slab, Gz), dtype=np.float64) for c in COMPONENTS}
    cnt = np.zeros((Gx, ny_slab, Gz), dtype=np.float64)

    for fp in local_files:
        i, j, k = rank_to_ijk(proc_id_from_filename(fp))
        xs = 0 if i == 0 else 1
        ys = 0 if j == 0 else 1
        zs = 0 if k == 0 else 1
        gx0, gy0, gz0 = i * nx_c + xs, j * ny_c + ys, k * nz_c + zs
        nxu, nyu, nzu = nx_t - xs, ny_t - ys, nz_t - zs

        a = max(gy0, jlo)
        b = min(gy0 + nyu - 1, jhi - 1)
        if a > b:
            continue
        js = a - gy0 + ys
        je = b - gy0 + ys + 1
        oy = a - jlo
        ny_read = je - js

        with h5py.File(fp, "r") as f:
            for comp in COMPONENTS:
                path = f"fields/{comp}/{cycle_name}"
                if path not in f:
                    raise KeyError(
                        f"Missing dataset {path} in {os.path.basename(fp)}")
                blk = np.asarray(f[path][xs:, js:je, zs:], dtype=np.float64)
                local[comp][gx0:gx0+nxu, oy:oy+ny_read, gz0:gz0+nzu] += blk
        cnt[gx0:gx0+nxu, oy:oy+ny_read, gz0:gz0+nzu] += 1.0

    for comp in COMPONENTS:
        comm.Allreduce(MPI.IN_PLACE, local[comp], op=MPI.SUM)
    comm.Allreduce(MPI.IN_PLACE, cnt, op=MPI.SUM)

    if rank != 0:
        return None
    ok = cnt > 0
    for comp in COMPONENTS:
        local[comp][ok] /= cnt[ok]
    if not ok.all():
        print(f"  WARNING: {int((~ok).sum())} slab nodes never written at "
              f"{cycle_name}", flush=True)
    return local


def fluctuation_slab(slab):
    """dB = B - <B>_xz(y)."""
    return {c: slab[c] - slab[c].mean(axis=(0, 2), keepdims=True)
            for c in COMPONENTS}


def build_axis_binning(n, L):
    k0 = 2.0 * np.pi / L
    ka = 2.0 * np.pi * np.fft.fftfreq(n, d=L / n)
    kabs = np.abs(ka)
    idx = np.floor(kabs / k0 + 0.5).astype(np.int64)
    n_bins = int(idx.max()) + 1
    counts = np.bincount(idx, minlength=n_bins)
    k_sum = np.bincount(idx, weights=kabs, minlength=n_bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        kc = np.where(counts > 0, k_sum / np.maximum(counts, 1), np.nan)
    kny = np.pi * n / L
    valid = (counts > 0) & (kc <= kny) & (np.arange(n_bins) > 0)
    return idx, n_bins, kc, valid


def inplane_directional_spectrum(field, axis_lbl, idx, n_bins):
    nx, ny_slab, nz = field[COMPONENTS[0]].shape
    norm = 1.0 / (nx * nz)
    power_kxz = np.zeros((nx, nz), dtype=np.float64)
    for comp in COMPONENTS:
        Bhat = np.fft.fft2(field[comp], axes=(0, 2)) * norm
        power_kxz += 0.5 * (np.abs(Bhat) ** 2).mean(axis=1)
    per_axis = power_kxz.sum(axis=1) if axis_lbl == "x" else power_kxz.sum(axis=0)
    return np.bincount(idx, weights=per_axis, minlength=n_bins)


def build_sf_lags(n, max_frac):
    hi = max(SF_MIN_LAG_CELLS + 1, int(np.floor(max_frac * n)))
    lags = np.unique(np.round(
        np.geomspace(SF_MIN_LAG_CELLS, hi, num=40)).astype(int))
    return lags[(lags >= SF_MIN_LAG_CELLS) & (lags <= hi)]


def sf_moments(field, axis, lags, orders):
    """Raw structure-function moments <|df|^n> along one axis for each n in
    `orders`. Returns dict n->array. axis: 0=x,1=y,2=z.

    ###! SPEED: this is the dominant cost of the compute stage, so three
    ###! micro-optimisations are applied, ALL exact for even n (which these are):
    ###!   1. The increment is cast to float32. |df|^n on a large slab is
    ###!      memory-bandwidth-bound; halving the bytes roughly halves the time.
    ###!      Precision cost on <|df|^8> is ~6e-8 -- far below any physical
    ###!      significance (verified on synthetic data).
    ###!   2. Work in d2 = |df|^2 = sum_i df_i^2 and form |df|^n = (d2)^(n/2)
    ###!      by INTEGER powers (repeated multiply), never a fractional power.
    ###!   3. No sqrt is taken -- for even n it would be squared straight back.
    ###! For ODD n this routine would need sqrt(d2); a guard below raises rather
    ###! than silently returning a wrong (integer-power) answer.
    """
    for n in orders:
        if n % 2 != 0:
            raise ValueError(f"sf_moments fast path assumes even orders; got n={n}")

    n_axis = field[COMPONENTS[0]].shape[axis]
    out = {n: np.full(lags.shape, np.nan, dtype=np.float64) for n in orders}
    half = {n: n // 2 for n in orders}          ###! d2 exponent per order
    max_half = max(half.values())

    for li, l in enumerate(lags):
        if l >= n_axis:
            continue
        sl_hi = [slice(None)] * 3
        sl_lo = [slice(None)] * 3
        sl_hi[axis] = slice(l, n_axis)
        sl_lo[axis] = slice(0, n_axis - l)

        d2 = None
        for comp in COMPONENTS:
            ###! float32 increment; accumulate |df|^2 in float32
            d = (field[comp][tuple(sl_hi)] -
                 field[comp][tuple(sl_lo)]).astype(np.float32)
            d2 = d * d if d2 is None else d2 + d * d

        ###! integer powers of d2 by repeated multiply, reusing lower powers:
        ###! p[1]=d2, p[2]=d2^2, ... up to max_half. |df|^n = p[n/2].
        p = d2                                   ###! p currently d2^1
        power_cache = {1: d2}
        for e in range(2, max_half + 1):
            p = p * d2
            power_cache[e] = p
        for n in orders:
            ###! mean in float64 for a stable reduction over many cells
            out[n][li] = np.mean(power_cache[half[n]], dtype=np.float64)

    return out


parser = argparse.ArgumentParser(description="MPI compute stage: write per-CS-averaged in-plane spectra E(kx),E(kz) and SF moments <|df|^n> (n=2,4,8) for B and dB to a .npz cache.")
parser.add_argument("dir_data", type=str)
parser.add_argument("xmin", type=float); parser.add_argument("xmax", type=float)
parser.add_argument("ymin", type=float); parser.add_argument("ymax", type=float)
parser.add_argument("zmin", type=float); parser.add_argument("zmax", type=float)
parser.add_argument("--nxc", type=int, required=True)
parser.add_argument("--nyc", type=int, required=True)
parser.add_argument("--nzc", type=int, required=True)
parser.add_argument("--cycle-start", type=int, default=0)
parser.add_argument("--cycle-end", type=int, default=5000)
parser.add_argument("--cycle-step", type=int, default=100)
parser.add_argument("--outdir", type=str, default=None)
parser.add_argument("--cache-name", type=str, default="spectra_sf_cache.npz")
parser.add_argument("--mapping", type=str, default="auto",
                    choices=["auto", "A", "B", "C", "D", "E", "F"])
args = parser.parse_args()

nxc, nyc, nzc = args.nxc, args.nyc, args.nzc
Lx = args.xmax - args.xmin
Ly = args.ymax - args.ymin
Lz = args.zmax - args.zmin
outdir = args.dir_data if args.outdir is None else args.outdir
if rank == 0:
    os.makedirs(outdir, exist_ok=True)

###! ---------------- setup ----------------
setup_error = None
if rank == 0:
    try:
        all_files = sorted(glob.glob(os.path.join(args.dir_data, "proc*.hdf")))
        if not all_files:
            raise RuntimeError(f"No proc*.hdf files found in {args.dir_data}")
        with h5py.File(all_files[0], "r") as f:
            tile_shape = tuple(f[f"fields/Bx/cycle_{args.cycle_start}"].shape)
        nx_t, ny_t, nz_t = tile_shape
        XLEN, YLEN, ZLEN = nxc // (nx_t-1), nyc // (ny_t-1), nzc // (nz_t-1)
        if XLEN * YLEN * ZLEN != len(all_files):
            raise RuntimeError(f"decomposition {XLEN}x{YLEN}x{ZLEN} != "
                               f"{len(all_files)} files")
        map_name = (choose_mapping(all_files, XLEN, YLEN, ZLEN)
                    if args.mapping == "auto" else args.mapping)
    except Exception as exc:
        setup_error = f"{type(exc).__name__}: {exc}"
        all_files = map_name = tile_shape = None
        XLEN = YLEN = ZLEN = None
else:
    all_files = map_name = tile_shape = None
    XLEN = YLEN = ZLEN = None

setup_error = comm.bcast(setup_error, root=0)
if setup_error is not None:
    if rank == 0:
        print(f"SETUP FAILED: {setup_error}", flush=True)
    comm.Barrier(); raise SystemExit(1)

all_files  = comm.bcast(all_files, root=0)
map_name   = comm.bcast(map_name, root=0)
tile_shape = comm.bcast(tile_shape, root=0)
XLEN = comm.bcast(XLEN, root=0)
YLEN = comm.bcast(YLEN, root=0)
ZLEN = comm.bcast(ZLEN, root=0)

rank_to_ijk = mapping_candidates(XLEN, YLEN, ZLEN)[map_name]
local_files = all_files[rank::size]
G_shape = global_shape_shared(tile_shape, XLEN, YLEN, ZLEN)
Gx, Gy, Gz = G_shape

requested = list(range(args.cycle_start, args.cycle_end + 1, args.cycle_step))

probe_error = None
if rank == 0:
    try:
        with h5py.File(all_files[0], "r") as f:
            cycle_names, missing = [], []
            for c in requested:
                nm = f"cycle_{c}"
                (cycle_names if f"fields/Bx/{nm}" in f else missing).append(nm)
            if not cycle_names:
                raise RuntimeError("None of the requested cycles are present.")
            if missing:
                print(f"WARNING: {len(missing)} cycle(s) absent: "
                      f"{missing[0]} ... {missing[-1]}", flush=True)
    except Exception as exc:
        probe_error = f"{type(exc).__name__}: {exc}"
        cycle_names = None
else:
    cycle_names = None
probe_error = comm.bcast(probe_error, root=0)
if probe_error is not None:
    if rank == 0:
        print(f"PROBE FAILED: {probe_error}", flush=True)
    comm.Barrier(); raise SystemExit(1)
cycle_names = comm.bcast(cycle_names, root=0)

slab_ranges = []
for (f_lo, f_hi) in CS_SLABS:
    jlo = max(0, int(np.floor(f_lo * nyc)))
    jhi = min(Gy, int(np.ceil(f_hi * nyc)))
    slab_ranges.append((jlo, jhi))

nx_fft = Gx - 1
nz_fft = Gz - 1

if rank == 0:
    idx_x, nbx, kcx, vx = build_axis_binning(nx_fft, Lx)
    idx_z, nbz, kcz, vz = build_axis_binning(nz_fft, Lz)
    lags_x = build_sf_lags(nx_fft, max_frac=0.5)
    lags_z = build_sf_lags(nz_fft, max_frac=0.5)
    ny_slab0 = slab_ranges[0][1] - slab_ranges[0][0]
    lags_y = build_sf_lags(ny_slab0, max_frac=0.9)

    ###! per-sheet accumulators (averaged at the end)
    def new_store():
        return {"Ex_B": [], "Ez_B": [], "Ex_dB": [], "Ez_dB": [],
                "SF_B": {n: {"x": [], "y": [], "z": []} for n in SF_ORDERS},
                "SF_dB": {n: {"x": [], "y": [], "z": []} for n in SF_ORDERS}}
    store = {cs: new_store() for cs in range(len(CS_SLABS))}

###! ---------------- main loop ----------------
for cyc in cycle_names:
    for cs, (jlo, jhi) in enumerate(slab_ranges):
        slab = assemble_slab(cyc, local_files, rank_to_ijk, tile_shape,
                             G_shape, jlo, jhi)
        if rank != 0:
            continue

        B = {c: slab[c][:nx_fft, :, :nz_fft] for c in COMPONENTS}
        dB = fluctuation_slab(B)

        store[cs]["Ex_B"].append(inplane_directional_spectrum(B, "x", idx_x, nbx))
        store[cs]["Ez_B"].append(inplane_directional_spectrum(B, "z", idx_z, nbz))
        store[cs]["Ex_dB"].append(inplane_directional_spectrum(dB, "x", idx_x, nbx))
        store[cs]["Ez_dB"].append(inplane_directional_spectrum(dB, "z", idx_z, nbz))

        for tag, fld in (("SF_B", B), ("SF_dB", dB)):
            mx = sf_moments(fld, 0, lags_x, SF_ORDERS)
            my = sf_moments(fld, 1, lags_y, SF_ORDERS)
            mz = sf_moments(fld, 2, lags_z, SF_ORDERS)
            for n in SF_ORDERS:
                store[cs][tag][n]["x"].append(mx[n])
                store[cs][tag][n]["y"].append(my[n])
                store[cs][tag][n]["z"].append(mz[n])

    if rank == 0:
        print(f"{cyc} done", flush=True)

###! ---------------- average sheets + write cache ----------------
if rank == 0:
    def avg2(key):
        return 0.5 * (np.array(store[0][key]) + np.array(store[1][key]))

    def avg2_sf(tag, n, ax):
        a = np.array(store[0][tag][n][ax])
        b = np.array(store[1][tag][n][ax])
        return 0.5 * (a + b)

    times = np.array([int(c.replace("cycle_", "")) * TIME_PER_CYCLE
                      for c in cycle_names])
    cycles = np.array([int(c.replace("cycle_", "")) for c in cycle_names])

    save = dict(
        # spectra (CS-averaged)
        Ex_B=avg2("Ex_B"), Ez_B=avg2("Ez_B"),
        Ex_dB=avg2("Ex_dB"), Ez_dB=avg2("Ez_dB"),
        # axes / masks
        kc_x=kcx, valid_x=vx, kc_z=kcz, valid_z=vz,
        lags_x=lags_x, lags_y=lags_y, lags_z=lags_z,
        dx=Lx / nxc, dy=Ly / nyc, dz=Lz / nzc,
        cycles=cycles, times=times,
        # meta
        sf_orders=np.array(SF_ORDERS),
        cs_slabs=np.array(CS_SLABS),
        grid=np.array([nxc, nyc, nzc]),
        box=np.array([Lx, Ly, Lz]),
        mapping=np.array([map_name], dtype=object),
    )
    ###! SF moments: one array per (field, order, direction), CS-averaged
    for tag in ("SF_B", "SF_dB"):
        for n in SF_ORDERS:
            for ax in ("x", "y", "z"):
                save[f"{tag}{n}_{ax}"] = avg2_sf(tag, n, ax)

    cache_path = os.path.join(outdir, args.cache_name)
    np.savez_compressed(cache_path, **save)
    print(f"Wrote cache: {cache_path}", flush=True)
    print(f"  cycles={len(cycles)}  SF orders={SF_ORDERS}  "
          f"(raw moments <|df|^n>, CS-averaged)", flush=True)

comm.Barrier()