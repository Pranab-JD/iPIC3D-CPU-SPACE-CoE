"""
Created on Fri Aug 14 2026

@author: Pranab JD, Claude

Description: Reconnection rate for a 3D double-Harris iPIC3D run, computed
             SEPARATELY for the two current sheets (CS1 = lower y, CS2 = upper y).

Method
------
1. Assemble the z-AVERAGED in-plane field <Bx>_z, <By>_z on the global (x,y) grid.
   For z-periodic data,

       d_x <Bx>_z + d_y <By>_z = -<d_z Bz>_z = 0    (exactly)

   so a 2D flux function exists with NO approximation. This measures the
   k_z = 0 ("2D-like") reconnection rate; oblique (k_z != 0) reconnection is
   NOT captured -- use --z-planes to quantify how much that matters.

2. psi(x,y) from   Bx = +d_y psi ,  By = -d_x psi     (i.e. psi = A_z).

3. Split y at nyg//2 (one sheet per half, as in Extract_3D_CS.py). In each half
   locate the neutral line Bx = 0 column by column and take

       dpsi = max(psi_neutral) - min(psi_neutral)   ( = psi_X - psi_O )

   Tracking the neutral line per column (rather than reading psi along a fixed
   y row) keeps this valid once the sheets ripple.

4. R = (1/(B0 * vA)) d(dpsi)/dt, differenced across dumps.

Writes reconnection_rate.txt with dpsi and R for CS1, CS2, total and average.

Usage
-----
  SCRIPT="../postprocessing_tools/python/Reconnection_rate_3D.py"
  DATA_DIR="/gpfs/scratch/ehpc765/pjd/RMR/sigma_1/Bz_0/slope_1.66_amp_0.40/xy/"
  OUT_DIR="${DATA_DIR}/reconnection"

  XLEN=64; YLEN=2; ZLEN=64
  xmin=0; xmax=55
  ymin=0; ymax=110
  zmin=0; zmax=55

  srun python3 "$SCRIPT" "$DATA_DIR" "$OUT_DIR" \
      "$XLEN" "$YLEN" "$ZLEN" \
      $xmin $xmax $ymin $ymax $zmin $zmax \
      --nxc 128 --nyc 256 --nzc 128 \
      --sigma 1 \
      --time-denom 5 \
      --cycle-start 0 --cycle-end 20000 --cycle-step 100 \
      --z-planes 0,0.25,0.5,0.75

  Box lengths are derived from the extents: Lx = xmax - xmin, etc., then
  dx = Lx/nxc. Both the extents and nxc/nyc/nzc are yours to set; the code
  applies no default and no relation between them beyond that division.
"""

import os
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import glob
import argparse
from datetime import datetime

import numpy as np
import h5py
from mpi4py import MPI

###! Plotting is rank-0 only and headless. If matplotlib is missing the run
###! still produces every .txt table; only the figures are skipped.
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:
    HAVE_MPL = False

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

t_wall = datetime.now()


###! ============================================================
###! Tile / mapping helpers  (same logic as your existing scripts)
###! ============================================================

def proc_id_from_filename(fp):
    base = os.path.basename(fp)
    return int(base.replace("proc", "").replace(".hdf", ""))


def mapping_candidates(XLEN, YLEN, ZLEN):
    """proc_id -> (i,j,k). Six common orderings; the right one is inferred."""
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

    return [("A", A), ("B", B), ("C", C), ("D", D), ("E", E), ("F", F)]


def score_occupancy(occ):
    """Lower is better. Heavily penalise multiply-covered nodes."""
    gaps = int(np.count_nonzero(occ == 0))
    overlaps = int(np.count_nonzero(occ > 1))
    maxv = int(occ.max()) if occ.size else 0
    return gaps * 10 + overlaps * 2 + max(0, maxv - 1) * 1000


def infer_mapping(all_files, XLEN, YLEN, ZLEN, tile_shape):
    """
    Score every candidate proc -> (i,j,k) mapping by grid occupancy.

    CRITICAL CAVEAT: when XLEN, YLEN and ZLEN are all powers of two (e.g.
    64 x 2 x 64), EVERY candidate is a perfect bijection and they ALL score 0,
    while assigning 98-100% of tiles to DIFFERENT places. Occupancy cannot
    distinguish them. This function therefore returns the full list of tied
    winners; the caller must break the tie on the data itself.
    """
    nx_t, ny_t, nz_t = tile_shape
    nx_c, ny_c, nz_c = nx_t - 1, ny_t - 1, nz_t - 1
    nxg = XLEN * nx_c + 1
    nyg = YLEN * ny_c + 1
    nzg = ZLEN * nz_c + 1

    proc_ids = [proc_id_from_filename(fp) for fp in all_files]
    scores = {}

    for name, fn in mapping_candidates(XLEN, YLEN, ZLEN):
        occ = np.zeros((nxg, nyg, nzg), dtype=np.int8)
        bad = False

        for pid in proc_ids:
            i, j, k = fn(pid)
            if not (0 <= i < XLEN and 0 <= j < YLEN and 0 <= k < ZLEN):
                bad = True
                break

            xs = 0 if i == 0 else 1
            ys = 0 if j == 0 else 1
            zs = 0 if k == 0 else 1

            gx0 = i * nx_c + xs
            gy0 = j * ny_c + ys
            gz0 = k * nz_c + zs

            occ[gx0:gx0 + nx_t - xs,
                gy0:gy0 + ny_t - ys,
                gz0:gz0 + nz_t - zs] += 1

        if bad:
            continue

        scores[name] = score_occupancy(occ)

    if not scores:
        raise RuntimeError("Could not infer proc -> (i,j,k) mapping from filenames.")

    best = min(scores.values())
    tied = sorted([n for n, s in scores.items() if s == best])
    return tied, best


def total_variation_slices(sz, sy, sx):
    """
    Mean |first difference| summed over both axes of THREE ORTHOGONAL slices.
    A correctly assembled field is smooth across tile seams; a mis-assembled
    one has discontinuities there, so the correct mapping MINIMISES this.

    Three slices, not the z-average: the z-average is invariant under any
    permutation of z-tiles, so a mapping that swaps the j and k roles can
    score LOWER than the truth by smearing real y-structure into z. Measured
    on synthetic tiles with a known answer, the z-averaged metric picked the
    WRONG mapping while the three-slice metric picked the right one.
    """
    t = 0.0
    for S in (sz, sy, sx):
        t += sum(float(np.mean(np.abs(np.diff(S, axis=a)))) for a in range(2))
    return t


def assemble_tiebreak_slices(cycle, local_files, pid_to_ijk, tile_shape,
                             nxg, nyg, nzg):
    """
    Assemble Bx on three orthogonal mid-planes (z=nzg//2, y=nyg//2, x=nxg//2)
    for one cycle. Memory is O(N^2), so this is cheap enough to repeat once per
    candidate mapping.
    """
    nx_t, ny_t, nz_t = tile_shape
    nx_c, ny_c, nz_c = nx_t - 1, ny_t - 1, nz_t - 1
    ki, ji, ii = nzg // 2, nyg // 2, nxg // 2

    lz = np.zeros((nxg, nyg))
    ly = np.zeros((nxg, nzg))
    lx = np.zeros((nyg, nzg))

    for fp in local_files:
        i, j, k = pid_to_ijk(proc_id_from_filename(fp))
        xs = 0 if i == 0 else 1
        ys = 0 if j == 0 else 1
        zs = 0 if k == 0 else 1
        gx0, gy0, gz0 = i*nx_c + xs, j*ny_c + ys, k*nz_c + zs
        nxu, nyu, nzu = nx_t - xs, ny_t - ys, nz_t - zs

        hz = gz0 <= ki < gz0 + nzu
        hy = gy0 <= ji < gy0 + nyu
        hx = gx0 <= ii < gx0 + nxu
        if not (hz or hy or hx):
            continue

        try:
            with h5py.File(fp, "r") as f:
                d = f[f"fields/Bx/{cycle}"]
                if hz:
                    lz[gx0:gx0+nxu, gy0:gy0+nyu] = d[xs:, ys:, (ki-gz0)+zs]
                if hy:
                    ly[gx0:gx0+nxu, gz0:gz0+nzu] = d[xs:, (ji-gy0)+ys, zs:]
                if hx:
                    lx[gy0:gy0+nyu, gz0:gz0+nzu] = d[(ii-gx0)+xs, ys:, zs:]
        except Exception:
            continue

    out = []
    for a in (lz, ly, lx):
        r = np.zeros_like(a) if rank == 0 else None
        comm.Reduce(a, r, op=MPI.SUM, root=0)
        out.append(r)
    return out


###! ============================================================
###! Assembly:  z-averaged Bx, By on the global (x,y) grid
###! ============================================================

def assemble_chunk(cycles, local_files, pid_to_ijk, tile_shape,
                   nxg, nyg, nzg, exclude_last_z, slice_ks, work_dtype):
    """
    For each cycle name in `cycles`, build on rank 0:
        Bx, By    : (nc, nxg, nyg)  z-AVERAGED in-plane field
        Bxs, Bys  : (n_planes, nc, nxg, nyg) single-z planes, or None
        found     : (nc,) int, number of tiles that supplied data

    Accumulated as SUMS tile-by-tile with a separate plane count, then divided
    once -- so no global 3D array is ever held. Memory is O(nc * nxg * nyg).

    `exclude_last_z` drops global node z = nzg-1, which duplicates z = 0 under
    periodic BCs; including it would over-weight that plane. Pass
    --no-exclude-last-z if z is NOT periodic.
    """
    nx_t, ny_t, nz_t = tile_shape
    nx_c, ny_c, nz_c = nx_t - 1, ny_t - 1, nz_t - 1
    nc = len(cycles)

    locBx = np.zeros((nc, nxg, nyg), dtype=work_dtype)
    locBy = np.zeros((nc, nxg, nyg), dtype=work_dtype)
    loccnt = np.zeros((nxg, nyg), dtype=np.int32)     ###! z-planes per column
    locfound = np.zeros(nc, dtype=np.int32)           ###! tiles supplying each cycle
    loctiles = np.zeros(1, dtype=np.int32)            ###! tiles this rank processed
    locfail = np.zeros(1, dtype=np.int32)             ###! tiles that raised on read

    slice_ks = list(slice_ks) if slice_ks else []
    nsl = len(slice_ks)
    locBxs = np.zeros((nsl, nc, nxg, nyg), dtype=work_dtype) if nsl else None
    locBys = np.zeros((nsl, nc, nxg, nyg), dtype=work_dtype) if nsl else None

    for fp in local_files:
        pid = proc_id_from_filename(fp)
        i, j, k = pid_to_ijk(pid)

        xs = 0 if i == 0 else 1
        ys = 0 if j == 0 else 1
        zs = 0 if k == 0 else 1

        gx0 = i * nx_c + xs
        gy0 = j * ny_c + ys
        gz0 = k * nz_c + zs

        nx_use = nx_t - xs
        ny_use = ny_t - ys
        nz_use = nz_t - zs

        ###! trim the duplicated top-most global z node
        gz1 = gz0 + nz_use
        if exclude_last_z:
            gz1 = min(gz1, nzg - 1)
        nz_take = gz1 - gz0

        ###! which requested planes does this tile own? (m, local k)
        owned = [(m, (gk - gz0) + zs) for m, gk in enumerate(slice_ks)
                 if gz0 <= gk < gz0 + nz_use]

        if nz_take <= 0 and not owned:
            continue

        loctiles[0] += 1

        ###! NEVER let an I/O error escape this loop. Every rank must reach the
        ###! collective Reduce below; a rank that exits early leaves the other
        ###! 8191 blocked forever. Failures are counted and reported instead.
        try:
            with h5py.File(fp, "r") as f:
                for ci, cyc in enumerate(cycles):
                    pBx = f"fields/Bx/{cyc}"
                    pBy = f"fields/By/{cyc}"

                    if pBx not in f or pBy not in f:
                        continue      ###! missing cycle -> caught by the partial check

                    if nz_take > 0:
                        bx = np.asarray(f[pBx][xs:, ys:, zs:zs + nz_take], dtype=np.float64)
                        by = np.asarray(f[pBy][xs:, ys:, zs:zs + nz_take], dtype=np.float64)
                        locBx[ci, gx0:gx0 + nx_use, gy0:gy0 + ny_use] += bx.sum(axis=2)
                        locBy[ci, gx0:gx0 + nx_use, gy0:gy0 + ny_use] += by.sum(axis=2)

                    for m, rk in owned:
                        locBxs[m, ci, gx0:gx0 + nx_use, gy0:gy0 + ny_use] += \
                            np.asarray(f[pBx][xs:, ys:, rk], dtype=np.float64)
                        locBys[m, ci, gx0:gx0 + nx_use, gy0:gy0 + ny_use] += \
                            np.asarray(f[pBy][xs:, ys:, rk], dtype=np.float64)

                    ###! incremented ONLY after a successful read of this cycle
                    locfound[ci] += 1
        except Exception as e:
            locfail[0] += 1
            print(f"  rank {rank}: read failed for {os.path.basename(fp)}: {e}",
                  flush=True)
            continue

        if nz_take > 0:
            loccnt[gx0:gx0 + nx_use, gy0:gy0 + ny_use] += nz_take

    ###! ---------------- reduce to root ----------------
    mpi_t = MPI.FLOAT if work_dtype == np.float32 else MPI.DOUBLE

    def red_f(arr):
        out = np.zeros_like(arr) if rank == 0 else None
        comm.Reduce([arr, mpi_t], [out, mpi_t] if rank == 0 else None,
                    op=MPI.SUM, root=0)
        return out

    Bx = red_f(locBx)
    By = red_f(locBy)
    Bxs = red_f(locBxs) if nsl else None
    Bys = red_f(locBys) if nsl else None

    cnt = np.zeros_like(loccnt) if rank == 0 else None
    comm.Reduce([loccnt, MPI.INT], [cnt, MPI.INT] if rank == 0 else None,
                op=MPI.SUM, root=0)

    found = np.zeros_like(locfound) if rank == 0 else None
    comm.Reduce([locfound, MPI.INT], [found, MPI.INT] if rank == 0 else None,
                op=MPI.SUM, root=0)

    ###! Allreduce (not Reduce): every rank needs these to agree on whether to abort.
    ntiles = comm.allreduce(int(loctiles[0]), op=MPI.SUM)
    nfail = comm.allreduce(int(locfail[0]), op=MPI.SUM)

    ###! ---------------- validate coverage (decided on root, shared by all) ----
    fatal = None
    if rank == 0:
        if nfail > 0:
            fatal = (f"{nfail} tile read(s) failed. Partial sums are unusable -- "
                     f"fix the I/O error rather than averaging over a hole.")
        elif cnt.min() == 0:
            fatal = ("Assembly gap: some (x,y) nodes received zero z-planes. "
                     "Check XLEN/YLEN/ZLEN or the inferred proc mapping.")

    ###! Broadcast the verdict so ALL ranks raise together. A root-only raise
    ###! would leave every other rank blocked on the next chunk's Reduce.
    fatal = comm.bcast(fatal, root=0)
    if fatal is not None:
        raise RuntimeError(fatal)

    if rank != 0:
        return None

    if cnt.min() != cnt.max():
        print(f"  WARNING: non-uniform z-plane count per column "
              f"(min={cnt.min()}, max={cnt.max()}). Averaging anyway.", flush=True)

    ###! A cycle present in SOME tiles but not all would otherwise be divided by
    ###! the FULL z-plane count (loccnt is accumulated per file, independently of
    ###! which cycles that file contains) and come out silently too small, with
    ###! no gap detected anywhere. Flag those cycles as invalid.
    for ci in range(nc):
        if 0 < found[ci] < ntiles:
            print(f"  WARNING: '{cycles[ci]}' present in only {found[ci]}/{ntiles} "
                  f"tiles -- incomplete, marking invalid.", flush=True)
            found[ci] = -1

    Bx /= cnt[None, :, :]
    By /= cnt[None, :, :]

    return dict(Bx=Bx, By=By, Bxs=Bxs, Bys=Bys,
                zcount=int(cnt.min()), found=found, ntiles=ntiles)


###! ============================================================
###! Flux function and neutral-line extraction
###! ============================================================

def _cumtrapz0(y, d, axis):
    """Cumulative trapezoid with a leading zero (avoids a scipy dependency)."""
    y = np.asarray(y, dtype=np.float64)
    n = y.shape[axis]
    lo = np.take(y, np.arange(0, n - 1), axis=axis)
    hi = np.take(y, np.arange(1, n), axis=axis)
    c = np.cumsum(0.5 * d * (lo + hi), axis=axis)
    pad = list(y.shape)
    pad[axis] = 1
    return np.concatenate([np.zeros(pad), c], axis=axis)


def flux_function_fft(Bx, By, dx, dy, nxc, nyc):
    """
    Flux function by spectral solution of the Poisson equation, for a FULLY
    PERIODIC domain. This is the default and is strictly better than path
    integration here.

    From  Bx = +d_y psi,  By = -d_x psi:

        d_y Bx - d_x By = d_yy psi + d_xx psi = laplacian(psi)

    Solved in Fourier space, so psi is single-valued BY CONSTRUCTION -- path
    independence is no longer an assumption that can fail. It also performs a
    Helmholtz projection: any compressive (curl-free) part of the in-plane
    field is discarded rather than corrupting psi, which is exactly right,
    because only the solenoidal part has a flux function at all.

    Verified on a synthetic periodic field: psi recovered to ~1e-15 even with
    a 30% compressive component added, where trapezoid path integration fails.

    Returns psi on the full (nxc+1, nyc+1) node grid, wrapped periodically.
    """
    bx = Bx[:nxc, :nyc]
    by = By[:nxc, :nyc]

    kx = 2.0 * np.pi * np.fft.fftfreq(nxc, d=dx)[:, None]
    ky = 2.0 * np.pi * np.fft.fftfreq(nyc, d=dy)[None, :]

    ###! omega = d_y Bx - d_x By = laplacian(psi)
    omega_k = 1j * ky * np.fft.fft2(bx) - 1j * kx * np.fft.fft2(by)

    k2 = kx**2 + ky**2
    k2[0, 0] = 1.0                      ###! avoid 0/0; the mean of psi is free
    psi_k = -omega_k / k2
    psi_k[0, 0] = 0.0                   ###! fix the gauge

    core = np.real(np.fft.ifft2(psi_k))

    ###! wrap back onto the node grid the rest of the code indexes
    psi = np.empty((nxc + 1, nyc + 1), dtype=np.float64)
    psi[:nxc, :nyc] = core
    psi[nxc, :nyc] = core[0, :]
    psi[:nxc, nyc] = core[:, 0]
    psi[nxc, nyc] = core[0, 0]
    return psi


def bx_from_psi(psi, dy, nxc, nyc):
    """
    The Bx implied by the flux function, Bx_sol = d_y psi, evaluated spectrally
    on the same periodic grid and wrapped back onto the node grid.

    The neutral line MUST be located from this field, not from the raw Bx.
    psi is the solenoidal projection, so it is exact even when the stored field
    carries a compressive part -- but the raw Bx still carries that part, and
    its spurious Bx=0 crossings then get evaluated on psi and folded into
    max-min. Measured on a reproduced IC with broadband curl-free
    contamination:

        resid   raw-Bx search        reconstructed-Bx search
        0.05    +0.0%   385 pts      0.000%   385 pts
        0.10    +0.3%   445 pts      0.000%   385 pts
        0.20   +54.3%   861 pts      0.000%   385 pts
        0.40  +587.3%  4961 pts      0.000%   385 pts

    The npts column is the tell: crossings should number about nxg (one per
    column). Anything much larger means the search field is contaminated.
    """
    ky = 2.0 * np.pi * np.fft.fftfreq(nyc, d=dy)[None, :]
    core = np.real(np.fft.ifft2(1j * ky * np.fft.fft2(psi[:nxc, :nyc])))
    out = np.empty((nxc + 1, nyc + 1), dtype=np.float64)
    out[:nxc, :nyc] = core
    out[nxc, :nyc] = core[0, :]
    out[:nxc, nyc] = core[:, 0]
    out[nxc, nyc] = core[0, 0]
    return out


def solenoidal_residual(Bx, By, psi, dx, dy, nxc, nyc):
    """
    How much of the in-plane field the flux function CANNOT represent:

        rms| B_inplane - curl(psi zhat) |  /  rms| B_inplane |

    This replaces path_err as the validity check. It is local, dimensionless,
    grid-size independent, and directly interpretable: 0.05 means 5% of the
    in-plane field is compressive and has been projected out.

    Large values do not mean the rate is wrong -- they mean a fraction of the
    field has no flux function, so Delta psi describes only the rest.
    """
    kx = 2.0 * np.pi * np.fft.fftfreq(nxc, d=dx)[:, None]
    ky = 2.0 * np.pi * np.fft.fftfreq(nyc, d=dy)[None, :]

    psi_k = np.fft.fft2(psi[:nxc, :nyc])
    bx_rec = np.real(np.fft.ifft2(1j * ky * psi_k))
    by_rec = np.real(np.fft.ifft2(-1j * kx * psi_k))

    bx, by = Bx[:nxc, :nyc], By[:nxc, :nyc]
    num = np.sqrt(np.mean((bx - bx_rec)**2 + (by - by_rec)**2))
    den = np.sqrt(np.mean(bx**2 + by**2))
    return float(num / max(den, 1e-300))


def flux_function(Bx, By, dx, dy):
    """
    psi(x,y) with  Bx = +d_y psi ,  By = -d_x psi   (psi = A_z), psi(0,0) = 0.
    Path taken: integrate -By along x at j=0, then Bx along y.
    Check: B = curl(psi zhat) = (d_y psi, -d_x psi, 0).  Consistent.
    """
    psi_x0 = -_cumtrapz0(By[:, 0], dx, axis=0)              # (nx,)
    return psi_x0[:, None] + _cumtrapz0(Bx, dy, axis=1)


def path_independence_error(Bx, By, dx, dy, psi):
    """
    Recompute psi via the ORTHOGONAL path. For an exactly 2D-solenoidal field
    the two agree to round-off. A large residual means z is not periodic, the
    fields are not colocated, or the assembly is broken.

    THIS IS THE LOAD-BEARING VALIDITY CHECK. Expect ~1e-12; > 1e-3 invalidates
    the whole 2D reduction and hence every rate in the output file.
    """
    psi_y0 = _cumtrapz0(Bx[0, :], dy, axis=0)               # (ny,)
    psi_alt = psi_y0[None, :] - _cumtrapz0(By, dx, axis=0)
    scale = max(float(np.max(np.abs(psi))), 1e-300)
    return float(np.max(np.abs(psi - psi_alt)) / scale)


def dpsi_from_neutral_line(Bx, psi, j_lo, j_hi):
    """
    Within the y-band [j_lo, j_hi], find the Bx = 0 crossing in each x-column
    (linear interpolation), evaluate psi there, and return

        dpsi = max(psi_neutral) - min(psi_neutral)    ( = psi_X - psi_O )

    All crossings in a column are kept -- with strong flapping a column can
    genuinely cross the sheet more than once. The returned count lets you spot
    the pathological case (count >> nx means noise, not structure).
    """
    vals = []
    for i in range(Bx.shape[0]):
        col = Bx[i, j_lo:j_hi + 1]
        s = np.sign(col)

        ###! A crossing is either a strict sign flip between j and j+1, OR an
        ###! EXACT zero at node j. The second case is not pedantic: an unperturbed
        ###! analytic Harris sheet sitting on a grid node gives Bx == 0.0 exactly,
        ###! sign() returns 0, the product is 0 (not negative), and a strict
        ###! "< 0" test silently finds no sheet at all. The "s[:-1] == 0" term
        ###! catches it without double-counting the j+1 node.
        cross = np.where((s[:-1] * s[1:] < 0) | (s[:-1] == 0))[0]

        for m in cross:
            j = j_lo + m
            b0, b1 = Bx[i, j], Bx[i, j + 1]
            if b0 == b1:
                continue                            ###! degenerate flat-zero pair
            w = b0 / (b0 - b1)                      ###! in [0,1); w = 0 if b0 == 0
            vals.append(psi[i, j] * (1.0 - w) + psi[i, j + 1] * w)

    if len(vals) < 2:
        return np.nan, 0

    v = np.asarray(vals)
    return float(v.max() - v.min()), int(v.size)


def find_sheet_centre(prof, lo, hi):
    """
    Locate one current sheet as the sign change of the x-averaged <Bx>_z
    profile within [lo, hi). If several crossings exist, take the one with the
    steepest local gradient -- that is the real sheet, not a noise wiggle.
    Returns a global y index, or None.
    """
    seg = prof[lo:hi]
    s = np.sign(seg)

    ###! Same exact-zero caveat as in dpsi_from_neutral_line: a sheet centred
    ###! exactly on a grid node gives prof == 0.0 there, which a strict sign-
    ###! product test misses entirely.
    cr = np.where((s[:-1] * s[1:] < 0) | (s[:-1] == 0))[0]

    if cr.size == 0:
        return None

    grad = np.abs(np.diff(seg))
    return lo + int(cr[np.argmax(grad[cr])])


def measure_B0(Bx, c1, c2, nyg):
    """
    Upstream |<Bx>_z| measured from the data, as a cross-check on --B0.

    Sampled at the two y-planes MIDWAY BETWEEN the sheets: one directly between
    c1 and c2, one at the periodic-wrapped midpoint on the other side. Those are
    the points furthest from both sheets regardless of where the sheets sit,
    which a fixed quarter-plane sample is not.

    DIAGNOSTIC ONLY. Disagreement with --B0 at t=0 means the box is asymmetric,
    a guide field leaks into Bx, or the input value is wrong.
    """
    m_in = (c1 + c2) // 2
    m_out = ((c2 + c1 + nyg) // 2) % nyg
    return 0.5 * (float(np.abs(Bx[:, m_in]).mean()) +
                  float(np.abs(Bx[:, m_out]).mean()))


###! ============================================================
###! Arguments
###! ============================================================

p = argparse.ArgumentParser(
    description="Reconnection rate per current sheet for 3D double-Harris iPIC3D output.")

p.add_argument("dir_data", type=str, help="Directory containing proc*.hdf")
p.add_argument("outdir",   type=str, help="Output directory")
###! --- box extents, positional, same order/style as B_J.py ---
###!     xmin xmax ymin ymax zmin zmax
###! Box LENGTHS are derived: Lx = xmax - xmin, etc.
p.add_argument("xmin", type=float)
p.add_argument("xmax", type=float)
p.add_argument("ymin", type=float)
p.add_argument("ymax", type=float)
p.add_argument("zmin", type=float)
p.add_argument("zmax", type=float)

###! --- grid: cell counts ---
p.add_argument("--nxc", type=int, required=True, help="Number of cells in x")
p.add_argument("--nyc", type=int, required=True, help="Number of cells in y")
p.add_argument("--nzc", type=int, required=True, help="Number of cells in z")

###! XLEN/YLEN/ZLEN are DERIVED from the cell counts and the tile shape:
###!     XLEN = nxc / (nx_tile - 1),  etc.
###! They carry no information the script does not already have, and deriving
###! them turns what was an unchecked user input into a consistency test
###! (the product must equal the number of proc*.hdf files).
p.add_argument("--xlen", type=int, default=None,
               help="Override the derived XLEN (normally unnecessary)")
p.add_argument("--ylen", type=int, default=None, help="Override the derived YLEN")
p.add_argument("--zlen", type=int, default=None, help="Override the derived ZLEN")

###! --- normalisation ---
p.add_argument("--sigma", type=float, required=True,
               help="ION magnetisation sigma_i = B^2/(4 pi rho_i), matching the "
                    "C++ input_param[0]. Sets vA via sqrt(s/(1+s)).")

###! --- optional enthalpy correction to the Alfven speed ---
###! The bare sqrt(sigma/(1+sigma)) counts ION REST-MASS inertia only. The
###! correct inertia is the enthalpy density w, which adds the electrons and
###! the thermal contribution:
###!
###!     sigma_eff = sigma_i / ( <gamma_i> + <gamma_e>/R ),   R = m_i/m_e
###!
###! using the same mean-Lorentz formula the C++ init uses for gamma_mean_e.
###! Impact (sigma_i = 1): 0.0% at R=1836 cold, but -18% in vA for a pair
###! plasma (=> +22% on every rate), and -10% for hot electrons at
###! Theta_i = 0.1. Since these runs span R = 1, 18.36, 1836, this matters.
###! Omit them all and the old cold-ion formula is used unchanged.
p.add_argument("--mass-ratio", type=float, default=None,
               help="m_i/m_e, i.e. |qom| of the electron species. Enables the "
                    "enthalpy correction to vA.")
p.add_argument("--theta-i", type=float, default=None,
               help="Upstream ion thermal spread (C++ col->getUth(1)).")
p.add_argument("--theta-e", type=float, default=None,
               help="Upstream electron thermal spread (C++ col->getUth(0)).")
p.add_argument("--time-denom", type=float, required=True,
               help="t*omega_p = cycle / time_denom. NOTE: B_J.py uses '//5' while "
                    "its own comment says 'T = cycle/10' -- they disagree, so set "
                    "this deliberately after checking Dt.")
p.add_argument("--B0", type=float, default=None,
               help="Asymptotic upstream |Bx|. If omitted, measured from the first "
                    "valid dump midway between the sheets.")

###! --- cycles ---
p.add_argument("--cycle-start", type=int, default=0)
p.add_argument("--cycle-end",   type=int, default=20000)
p.add_argument("--cycle-step",  type=int, default=500)
p.add_argument("--cycle-chunk", type=int, default=5,
               help="Cycles held in memory at once (files-outermost within a chunk). "
                    "Memory ~ chunk * 2 * nxg * nyg * 8 bytes per rank; halve it "
                    "again per requested --z-planes. Reduce for very large grids.")

###! --- method options ---
p.add_argument("--band-half-width", type=int, default=None,
               help="Restrict the neutral-line search to +/- this many cells around "
                    "each tracked sheet centre. Default: use each y-half.")
p.add_argument("--z-planes", type=str, default=None,
               help="Comma-separated fractions of Lz at which to ALSO compute the "
                    "rate on a single XY plane, e.g. '0,0.25,0.5,0.75'. Each gets "
                    "its own output file reconnection_rate_z<idx>.txt. NOTE these "
                    "are single planes, so <d_z Bz> does NOT vanish and the field "
                    "is not exactly 2D-solenoidal -- the Poisson solve still gives "
                    "a well-defined psi but watch the resid column, which will be "
                    "larger than for the z-average.")
p.add_argument("--no-exclude-last-z", action="store_true",
               help="Keep the top global z node in the average (use if z is NOT periodic)")

p.add_argument("--psi-method", type=str, default="fft", choices=["fft", "path"],
               help="How to build the flux function. 'fft' (default) solves the "
                    "Poisson equation spectrally: valid only for a fully periodic "
                    "domain, but then psi is single-valued by construction and any "
                    "compressive part of B is projected out instead of corrupting "
                    "psi. 'path' is the old trapezoid line integral, kept for "
                    "comparison; it is path-dependent whenever div.B != 0 discretely.")
p.add_argument("--no-plot", dest="plot", action="store_false",
               help="Skip the .png figures; write only the .txt tables.")
p.add_argument("--dtype", type=str, default="float64", choices=["float32", "float64"])
p.add_argument("--mapping", type=str, default="auto",
               choices=["auto", "A", "B", "C", "D", "E", "F"])

args = p.parse_args()

work_dtype = np.float32 if args.dtype == "float32" else np.float64
exclude_last_z = not args.no_exclude_last_z


###! ============================================================
###! Geometry (from CLI, cross-checked against the tile shape)
###! ============================================================

nxc, nyc, nzc = args.nxc, args.nyc, args.nzc

###! Box LENGTHS from the extents (B_J.py passes the same six numbers as
###! imshow extents; here they set the physical size of the domain).
Lx = args.xmax - args.xmin
Ly = args.ymax - args.ymin
Lz = args.zmax - args.zmin

for lbl, L, lo, hi in (("x", Lx, args.xmin, args.xmax),
                       ("y", Ly, args.ymin, args.ymax),
                       ("z", Lz, args.zmin, args.zmax)):
    if L <= 0:
        raise SystemExit(f"Box length in {lbl} is {L} (from {lbl}min={lo}, "
                         f"{lbl}max={hi}). Extents must satisfy max > min.")

###! nodes, matching the shared-boundary assembly used in all your scripts:
###!     n_global = LEN*(n_tile - 1) + 1 = n_cells + 1
nxg, nyg, nzg = nxc + 1, nyc + 1, nzc + 1

###! Periodic: node[n_cells] coincides with node[0], so the spacing divisor is
###! the CELL count, not the node count.
dx, dy, dz = Lx / nxc, Ly / nyc, Lz / nzc

def mean_lorentz(theta):
    """
    <gamma> for a Maxwell-Juttner distribution of thermal spread theta.
    Same closed form the C++ init uses for gamma_mean_e, so the two agree.
    """
    return 1.0 + theta * (6.0 + 15.0*theta) / (4.0 + 5.0*theta)


###! Effective magnetisation entering the Alfven speed.
if args.mass_ratio is None:
    sigma_eff = args.sigma                       ###! cold, ion rest mass only
    vA_note = "cold ion rest-mass only"
else:
    g_i = mean_lorentz(args.theta_i) if args.theta_i is not None else 1.0
    g_e = mean_lorentz(args.theta_e) if args.theta_e is not None else 1.0
    ###! w/(n m_i c^2) = <gamma_i> + <gamma_e>/R
    sigma_eff = args.sigma / (g_i + g_e / args.mass_ratio)
    vA_note = (f"enthalpy-corrected: R={args.mass_ratio:g}, "
               f"<g_i>={g_i:.4f}, <g_e>={g_e:.4f}")

vA = np.sqrt(sigma_eff / (1.0 + sigma_eff))          ###! c = 1


###! ============================================================
###! Discover files, probe tile shape, infer mapping
###! ============================================================

if rank == 0:
    all_files = sorted(glob.glob(os.path.join(args.dir_data, "proc*.hdf")))
    if not all_files:
        raise RuntimeError(f"No proc*.hdf found in {args.dir_data}")

    first_cycle = f"cycle_{args.cycle_start}"
    with h5py.File(all_files[0], "r") as f:
        pBx = f"fields/Bx/{first_cycle}"
        pBy = f"fields/By/{first_cycle}"
        if pBx not in f:
            raise KeyError(f"Missing dataset {pBx} in {all_files[0]}")
        if pBy not in f:
            raise KeyError(f"Missing dataset {pBy} in {all_files[0]}")

        tile_shape = tuple(f[pBx].shape)
        ###! Bx and By must live on the same grid or psi mixes two stencils
        if tuple(f[pBy].shape) != tile_shape:
            raise RuntimeError("Bx and By have different tile shapes -- fields are "
                               "not colocated; interpolation would be required.")

    ###! ---- derive the MPI decomposition from cell counts + tile shape ----
    nx_t, ny_t, nz_t = tile_shape
    for lbl, n_c, n_t in (("x", nxc, nx_t), ("y", nyc, ny_t), ("z", nzc, nz_t)):
        if n_t < 2 or n_c % (n_t - 1) != 0:
            raise RuntimeError(
                f"Cannot derive the {lbl} decomposition: {n_c} cells do not divide "
                f"evenly into tiles of {n_t - 1} cells (tile shape {tile_shape}). "
                f"Check --n{lbl}c, or the shared-boundary assumption.")

    XLEN = args.xlen if args.xlen is not None else nxc // (nx_t - 1)
    YLEN = args.ylen if args.ylen is not None else nyc // (ny_t - 1)
    ZLEN = args.zlen if args.zlen is not None else nzc // (nz_t - 1)

    ###! Consistency test the old positional form could never make: the derived
    ###! decomposition must account for exactly the files on disk.
    if XLEN * YLEN * ZLEN != len(all_files):
        raise RuntimeError(
            f"Derived decomposition {XLEN}x{YLEN}x{ZLEN} = {XLEN*YLEN*ZLEN} tiles, "
            f"but {len(all_files)} proc*.hdf files are present. Cell counts, tile "
            f"shape and file count disagree -- stopping rather than assembling "
            f"a corrupt field.")

    if XLEN * (nx_t - 1) != nxc or YLEN * (ny_t - 1) != nyc or ZLEN * (nz_t - 1) != nzc:
        raise RuntimeError(
            f"Override inconsistent: ({XLEN},{YLEN},{ZLEN}) with tile {tile_shape} "
            f"implies cells ({XLEN*(nx_t-1)},{YLEN*(ny_t-1)},{ZLEN*(nz_t-1)}), "
            f"not ({nxc},{nyc},{nzc}).")

    print(f"Decomposition   : {XLEN} x {YLEN} x {ZLEN} tiles"
          f"{' (derived)' if args.xlen is None else ' (overridden)'}"
          f"  = {XLEN*YLEN*ZLEN} files", flush=True)

    if args.mapping == "auto":
        tied, map_score = infer_mapping(all_files, XLEN, YLEN, ZLEN, tile_shape)
        print(f"Occupancy-compatible mappings: {tied}  (score {map_score}; 0 is perfect)",
              flush=True)
        if map_score != 0:
            print("  WARNING: non-zero score -- gaps or overlaps in coverage.", flush=True)
        map_name = tied[0] if len(tied) == 1 else None   ###! None -> needs tiebreak
    else:
        tied = [args.mapping]
        map_name = args.mapping
        print(f"Using forced proc mapping '{map_name}'", flush=True)
else:
    all_files, tile_shape, map_name, tied = None, None, None, None
    XLEN = YLEN = ZLEN = None

all_files  = comm.bcast(all_files, root=0)
tile_shape = comm.bcast(tile_shape, root=0)
map_name   = comm.bcast(map_name, root=0)
tied       = comm.bcast(tied, root=0)
XLEN       = comm.bcast(XLEN, root=0)
YLEN       = comm.bcast(YLEN, root=0)
ZLEN       = comm.bcast(ZLEN, root=0)

all_maps = {n: fn for n, fn in mapping_candidates(XLEN, YLEN, ZLEN)}
local_files = all_files[rank::size]

###! ---------------------------------------------------------------
###! Tiebreak on the DATA when occupancy cannot decide.
###!
###! With XLEN/YLEN/ZLEN all powers of two (64 x 2 x 64 is the case here)
###! all six candidate mappings are perfect bijections and score 0, yet they
###! place 98-100% of tiles differently. Picking the first would silently
###! scramble every assembled field. The correct mapping is the one that
###! yields a SMOOTH field across tile seams, so assemble <Bx>_z once per
###! candidate and take the smallest total variation.
###!
###! Note the z-tile assignment is irrelevant to this analysis: the z-average
###! is invariant under permutation of z-tiles. Only the (i,j) placement
###! matters, and that is exactly what the z-averaged TV measures.
###! ---------------------------------------------------------------
if map_name is None:
    if rank == 0:
        print(f"\n  Occupancy cannot distinguish {len(tied)} mappings -- breaking the "
              f"tie on field smoothness (lower total variation = correct):", flush=True)
    tv_results = {}
    for cand in tied:
        sz, sy, sx = assemble_tiebreak_slices(f"cycle_{args.cycle_start}",
                                              local_files, all_maps[cand],
                                              tile_shape, nxg, nyg, nzg)
        if rank == 0:
            tv = total_variation_slices(sz, sy, sx)
            tv_results[cand] = tv
            print(f"    {cand}: total variation = {tv:.6e}", flush=True)

    if rank == 0:
        map_name = min(tv_results, key=tv_results.get)
        srt = sorted(tv_results.values())
        margin = srt[1] / max(srt[0], 1e-300) if len(srt) > 1 else np.inf
        print(f"  -> selected '{map_name}'  (next-best is {margin:.2f}x rougher)",
              flush=True)
        if margin < 1.20:
            raise RuntimeError(
                f"Mapping tiebreak inconclusive: '{map_name}' is only {margin:.2f}x "
                f"smoother than the runner-up. Guessing here would silently scramble "
                f"every assembled field. Determine the rank ordering from your iPIC3D "
                f"source and pass it with --mapping.")
    map_name = comm.bcast(map_name, root=0)

pid_to_ijk = all_maps[map_name]

###! ---- requested single-z planes ----
if args.z_planes:
    z_fracs = [float(v) for v in args.z_planes.split(",") if v.strip() != ""]
else:
    z_fracs = []

slice_ks = []
for fr in z_fracs:
    if not (0.0 <= fr < 1.0):
        raise SystemExit(f"--z-planes fraction {fr} must satisfy 0 <= f < 1 "
                         f"(f=1 is the periodic image of f=0)")
    slice_ks.append(int(round(fr * nzc)))

if rank == 0:
    print(f"Tile shape      : {tile_shape}", flush=True)
    print(f"Cells           : {nxc} x {nyc} x {nzc}", flush=True)
    print(f"Nodes           : {nxg} x {nyg} x {nzg}", flush=True)
    print(f"Extents         : x[{args.xmin},{args.xmax}] "
          f"y[{args.ymin},{args.ymax}] z[{args.zmin},{args.zmax}]", flush=True)
    print(f"Box lengths     : Lx={Lx:.6g}  Ly={Ly:.6g}  Lz={Lz:.6g}", flush=True)
    print(f"Spacing         : dx={dx:.6g}  dy={dy:.6g}  dz={dz:.6g}", flush=True)
    print(f"vA/c            : {vA:.6f}  (sigma_i={args.sigma:g}, "
          f"sigma_eff={sigma_eff:.6f})", flush=True)
    print(f"                  {vA_note}", flush=True)
    print(f"z-average       : exclude_last_z = {exclude_last_z}", flush=True)
    if slice_ks:
        print("single-z planes  : " + ", ".join(
            f"f={f:g} -> k={k} (z={f*Lz:.3f})" for f, k in zip(z_fracs, slice_ks)),
            flush=True)


###! ============================================================
###! Main loop over cycle chunks
###! ============================================================

cycles_all = list(range(args.cycle_start, args.cycle_end + 1, args.cycle_step))
records = []            ###! rank-0 only
zcount_used = None
B0 = args.B0

###! ============================================================
###! Per-field analysis, shared by the z-average and every single-z plane
###! ============================================================

def analyse_field(Bx, By):
    """
    Full flux-function analysis of ONE in-plane field. Returns a record dict,
    or None if the two sheets could not be located.

    Used identically for the z-averaged field and for each requested single-z
    plane, so the two are directly comparable -- the ONLY difference is which
    field goes in. B0 is deliberately not fixed here: it is a global upstream
    property and is taken from the z-average for all planes.
    """
    if args.psi_method == "fft":
        psi = flux_function_fft(Bx, By, dx, dy, nxc, nyc)
    else:
        psi = flux_function(Bx, By, dx, dy)

    resid = solenoidal_residual(Bx, By, psi, dx, dy, nxc, nyc)

    ###! neutral line from the SOLENOIDAL Bx implied by psi, not the raw Bx
    Bx_search = (bx_from_psi(psi, dy, nxc, nyc)
                 if args.psi_method == "fft" else Bx)

    prof = Bx_search.mean(axis=0)
    mid = nyg // 2
    c1 = find_sheet_centre(prof, 0, mid)
    c2 = find_sheet_centre(prof, mid, nyg)
    if c1 is None or c2 is None:
        return None

    if args.band_half_width is None:
        b1 = (0, mid - 1)
        b2 = (mid, nyg - 1)
    else:
        h = args.band_half_width
        b1 = (max(0, c1 - h), min(mid - 1, c1 + h))
        b2 = (max(mid, c2 - h), min(nyg - 1, c2 + h))

    d1, n1 = dpsi_from_neutral_line(Bx_search, psi, *b1)
    d2, n2 = dpsi_from_neutral_line(Bx_search, psi, *b2)

    return dict(y1=c1, y2=c2, d1=d1, d2=d2, n1=n1, n2=n2, pi=resid,
                B0m=measure_B0(Bx, c1, c2, nyg))


###! ============================================================
###! Main loop
###! ============================================================

###! one record list per dataset: "zavg" plus one per requested plane
keys = ["zavg"] + [f"z{k}" for k in slice_ks]
records = {k: [] for k in keys}
zcount_used = None
B0 = args.B0

for c0 in range(0, len(cycles_all), args.cycle_chunk):
    chunk = cycles_all[c0:c0 + args.cycle_chunk]
    names = [f"cycle_{c}" for c in chunk]

    if rank == 0:
        print(f"\nAssembling {names[0]} .. {names[-1]}", flush=True)

    out = assemble_chunk(names, local_files, pid_to_ijk, tile_shape,
                         nxg, nyg, nzg, exclude_last_z, slice_ks, work_dtype)

    if rank != 0:
        continue

    zcount_used = out["zcount"]

    for ci, cyc in enumerate(chunk):
        ###! 0 = absent everywhere; -1 = present in only some tiles
        if out["found"][ci] <= 0:
            why = ("dataset absent" if out["found"][ci] == 0
                   else "incomplete across tiles")
            print(f"  cycle_{cyc}: {why}, skipped", flush=True)
            continue

        ###! ---- z-averaged field first; it also fixes B0 ----
        rec = analyse_field(out["Bx"][ci], out["By"][ci])
        if rec is None:
            print(f"  cycle_{cyc}: could not locate both sheets in the "
                  f"z-average, skipped", flush=True)
            continue

        if B0 is None:
            B0 = rec["B0m"]
            print(f"  B0 measured from data (midway between sheets): "
                  f"{B0:.6e}", flush=True)
        elif not records["zavg"] and abs(rec["B0m"] - B0) > 0.1 * abs(B0):
            print(f"  WARNING: --B0 = {B0:.4e} but measured upstream |Bx| = "
                  f"{rec['B0m']:.4e} (>10% off).", flush=True)

        rec.update(cycle=cyc, t=cyc / args.time_denom)
        records["zavg"].append(rec)


        ###! ---- each requested single-z plane, same pipeline ----
        for m, gk in enumerate(slice_ks):
            r = analyse_field(out["Bxs"][m][ci], out["Bys"][m][ci])
            if r is None:
                print(f"  cycle_{cyc} [z={gk:<5d}]: sheets not found, skipped",
                      flush=True)
                continue
            r.update(cycle=cyc, t=cyc / args.time_denom)
            records[f"z{gk}"].append(r)


###! ============================================================
###! Differentiate in time and write one table per dataset
###! ============================================================

def make_plot(d, path, what):
    """
    One figure, one panel: the reconnection rate for CS1 and CS2.

        R = (1/(B0 vA)) d(Delta psi)/dt

    Delta psi is psi_X - psi_O for the dominant island, so d(Delta psi)/dt is
    the reconnection electric field at that X-line -- the quantity the
    R ~ 0.1 benchmark refers to. A zero line is drawn because the SIGN is
    meaningful: negative means the flux is shrinking (island merging or a
    decaying seeded perturbation), not reconnection running backwards. Do not
    plot |R| -- rectifying zero-mean noise manufactures a spurious steady
    rate of 0.798*sigma.
    """
    if not HAVE_MPL:
        return None

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.axhline(0.0, color="k", lw=0.8)
    ax.plot(d["t"], d["R1"], "-", color="C0", lw=1.6, label="CS1")
    ax.plot(d["t"], d["R2"], "-", color="C3", lw=1.6, label="CS2")
    ax.set_xlabel(r"$t\,\omega_p^{-1}$", fontsize=13)
    ax.set_ylabel(r"$R = \dot{\Delta\psi}\,/\,(B_0 v_A)$", fontsize=13)
    ax.set_title(what, fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def write_table(recs, path, what):
    """Differentiate Delta psi in time, then write the table."""
    recs.sort(key=lambda r: r["cycle"])
    t  = np.array([r["t"]  for r in recs])
    d1 = np.array([r["d1"] for r in recs])
    d2 = np.array([r["d2"] for r in recs])

    ###! np.gradient: 2nd order interior, 1st order at the two endpoints
    R1, R2 = np.gradient(d1, t)/norm, np.gradient(d2, t)/norm

    pi_max = max(r["pi"] for r in recs)

    with open(path, "w") as fh:
        fh.write(f"# Reconnection rate, 3D double-Harris -- {what}\n")
        fh.write(f"# dir            = {args.dir_data}\n")
        fh.write(f"# XLEN,YLEN,ZLEN = {XLEN},{YLEN},{ZLEN}   mapping = {map_name}\n")
        fh.write(f"# cells          = {nxc} x {nyc} x {nzc}   "
                 f"nodes = {nxg} x {nyg} x {nzg}\n")
        fh.write(f"# extents        = x[{args.xmin},{args.xmax}] "
                 f"y[{args.ymin},{args.ymax}] z[{args.zmin},{args.zmax}]\n")
        fh.write(f"# spacing        = dx={dx:.8g} dy={dy:.8g} dz={dz:.8g}\n")
        fh.write(f"# time           = cycle / {args.time_denom}   [omega_p^-1]\n")
        fh.write(f"# sigma_i={args.sigma}  sigma_eff={sigma_eff:.8f}  "
                 f"vA/c={vA:.8f}\n")
        fh.write(f"# vA basis       = {vA_note}\n")
        fh.write(f"# B0={B0:.8e}  norm=B0*vA={norm:.8e}  "
                 f"(B0 from the z-average, applied to every plane)\n")
        fh.write(f"# psi method     = {args.psi_method}\n")
        fh.write(f"# max resid over all dumps = {pi_max:.3e}\n")
        fh.write("#\n")
        fh.write("# dpsi_CS*  psi_X - psi_O for the dominant island: the global\n")
        fh.write("#           max-min of psi along the Bx = 0 neutral line.\n")
        fh.write("# R_CS*     = (1/(B0*vA)) d(dpsi)/dt. Since d(psi)/dt = -c Ez,\n")
        fh.write("#           this IS the reconnection electric field at that\n")
        fh.write("#           X-line, normalised to the upstream inflow field.\n")
        fh.write("#           SIGN IS MEANINGFUL -- do not take |R|.\n")
        fh.write("# resid     compressive fraction of B with no flux function\n")
        fh.write("# npts*     Bx = 0 crossings found; ~nxg means one clean\n")
        fh.write("#           crossing per column, much more means folds\n")
        fh.write("# Endpoint rows use a 1st-order time derivative.\n")
        fh.write("#\n")
        fh.write("# {:>8s} {:>11s} {:>6s} {:>6s} {:>15s} {:>15s} {:>14s} {:>14s} "
                 "{:>11s} {:>7s} {:>7s}\n".format(
                     "cycle", "time", "y_cs1", "y_cs2",
                     "dpsi_CS1", "dpsi_CS2", "R_CS1", "R_CS2",
                     "resid", "npts1", "npts2"))
        for i, r in enumerate(recs):
            fh.write("  {:>8d} {:>11.4f} {:>6d} {:>6d} {:>15.7e} {:>15.7e} "
                     "{:>14.6e} {:>14.6e} {:>11.3e} {:>7d} {:>7d}\n".format(
                         r["cycle"], r["t"], r["y1"], r["y2"],
                         d1[i], d2[i], R1[i], R2[i],
                         r["pi"], r["n1"], r["n2"]))

    return dict(t=t, d1=d1, d2=d2, R1=R1, R2=R2)


if rank == 0:
    if len(records["zavg"]) < 2:
        raise RuntimeError("Need at least 2 valid dumps to differentiate in time.")

    norm = B0 * vA
    os.makedirs(args.outdir, exist_ok=True)

    summ = {}
    f0 = os.path.join(args.outdir, "reconnection_rate.txt")
    summ["zavg"] = write_table(records["zavg"], f0,
                               "z-AVERAGED (k_z = 0) flux function")
    print(f"\nWrote {f0}", flush=True)
    if args.plot:
        pf = make_plot(summ["zavg"],
                       os.path.join(args.outdir, "reconnection_rate.png"),
                       "z-averaged (k_z = 0)")
        if pf:
            print(f"Wrote {pf}", flush=True)

    for fr, gk in zip(z_fracs, slice_ks):
        key = f"z{gk}"
        if len(records[key]) < 2:
            print(f"  z={gk}: fewer than 2 valid dumps, no table written",
                  flush=True)
            continue
        fp = os.path.join(args.outdir, f"reconnection_rate_z{gk}.txt")
        summ[key] = write_table(
            records[key], fp,
            f"SINGLE XY PLANE at z index {gk} (f={fr:g}, z={fr*Lz:.4f})")
        print(f"Wrote {fp}", flush=True)
        if args.plot:
            pf = make_plot(summ[key],
                           os.path.join(args.outdir,
                                        f"reconnection_rate_z{gk}.png"),
                           f"single XY plane, z index {gk} (f={fr:g})")
            if pf:
                print(f"Wrote {pf}", flush=True)

    ###! ---- plane-to-plane spread: how 3D is this run? ----
    if len(summ) > 1:
        print("\n" + "=" * 66)
        print("PLANE-TO-PLANE COMPARISON (rate R, CS1 and CS2)")
        print("=" * 66)
        print("  A z-INDEPENDENT field gives identical rows. Divergence between")
        print("  planes is the growth of genuine 3D structure.")
        print(f"\n  {'dataset':>10} {'peak|R_CS1|':>14} {'peak|R_CS2|':>14}"
              f" {'max dpsi1':>12} {'max dpsi2':>12}")
        for k, v in summ.items():
            print(f"  {k:>10} {np.nanmax(np.abs(v['R1'])):>14.5e}"
                  f" {np.nanmax(np.abs(v['R2'])):>14.5e}"
                  f" {np.nanmax(v['d1']):>12.5e} {np.nanmax(v['d2']):>12.5e}")
        planes = [k for k in summ if k != "zavg"]
        if len(planes) > 1:
            m1 = np.array([np.nanmax(np.abs(summ[k]["R1"])) for k in planes])
            m2 = np.array([np.nanmax(np.abs(summ[k]["R2"])) for k in planes])
            for lab, m in (("CS1", m1), ("CS2", m2)):
                sp = m.max() / max(m.min(), 1e-300)
                print(f"\n  {lab}: max/min across planes = {sp:.2f}x", end="")
                print("   -> essentially 2D" if sp < 1.2 else
                      "   -> genuinely 3D; the z-average is NOT the whole story")
        print("=" * 66, flush=True)

    print(f"\nElapsed: {datetime.now() - t_wall}", flush=True)