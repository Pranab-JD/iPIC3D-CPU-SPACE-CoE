"""
Created on Sat Aug 16 2026

@author: Pranab JD

Oblique / 1D tearing spectra and kink modes for a 3D double-Harris run.

Reads the iPIC3D proc*.hdf tiles directly (B and J; E is not required) and
produces, per current sheet and per dump:

  1. TEARING, from By in a y-BAND around the sheet.
     rfft in x (m >= 0) and a FULL fft in z (n SIGNED), then the peak over y
     for each (m, n).
         n == 0  -> the ordinary 2D/1D tearing mode
         n != 0  -> OBLIQUE tearing
     A full fft in z is essential: +n and -n resonate at
         tanh(y/delta) = -(Bz/B0)(n/m)
     i.e. on OPPOSITE SIDES of the sheet, and are physically distinct. An
     rfft folds them together and destroys exactly that information.

     The peak is taken over y rather than evaluated at a predicted resonant
     surface, so no knowledge of delta or Bz/B0 is needed and the b = 0 case
     (where every mode resonates at y = 0) works unchanged. The y at which
     each peak occurs is written out as a VALIDATION column: it should follow
     delta*arctanh(-b*n/m). If it is scattered instead, the "modes" are noise.

  2. KINK, from the neutral surface y_n(x, z) itself.
     Bx is identically zero ON the surface, so there is nothing to transform
     there; the kink IS the displacement, so the surface position is the
     signal. fft along z, power averaged across x -- a kink is uniform in x,
     so an x-transform of it returns exactly zero.

     This is exact at any amplitude. The usual diagnostic, FFT_z of Bx on a
     FIXED y-plane, reads B0*tanh((y0 - y_n)/delta) and so under-reads by
     24% at a 1-delta displacement and 75% at 4 delta.

     CAVEAT (measured, not assumed): above dpsi = 2*B0*delta extra Bx = 0
     surfaces appear -- the island separatrices. Both of your runs exceed
     that in the nonlinear phase. This script therefore takes the crossing
     NEAREST the tracked sheet centre, which stays flat under tearing where
     a steepest-gradient pick does not.

Usage
-----
  srun python3 Tearing_kink_modes_3D.py DATA_DIR OUT_DIR \\
      xmin xmax ymin ymax zmin zmax \\
      --nxc 384 --nyc 768 --nzc 384 \\
      --time-denom 5 --mapping A \\
      --cycle-start 0 --cycle-end 20000 --cycle-step 500 \\
      --band-cells 40 --delta 1.0 --guide-field 0.0 --max-modes 10
"""

import os
import re
import glob
import argparse

import numpy as np
from mpi4py import MPI
import h5py

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


###! ============================================================
###! CLI
###! ============================================================

p = argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("dir_data")
p.add_argument("outdir")
p.add_argument("xmin", type=float); p.add_argument("xmax", type=float)
p.add_argument("ymin", type=float); p.add_argument("ymax", type=float)
p.add_argument("zmin", type=float); p.add_argument("zmax", type=float)

p.add_argument("--nxc", type=int, required=True)
p.add_argument("--nyc", type=int, required=True)
p.add_argument("--nzc", type=int, required=True)

p.add_argument("--time-denom", type=float, default=1.0,
               help="t*omega_p = cycle / time_denom.")
p.add_argument("--mapping", type=str, default="A",
               help="proc_id -> (i,j,k) ordering. A is MPI_Cart_create "
                    "row-major with dims (X,Y,Z): pid = (i*Y + j)*Z + k, so k "
                    "varies fastest. That is what your runs use, confirmed by "
                    "an exact-zero seam test. B..F are the other five "
                    "permutations of which index varies fastest.")

p.add_argument("--band-cells", type=int, default=40,
               help="Half-width in CELLS of the y-band taken around each sheet. "
                    "Must contain every resonant surface: y_res spans +/-2.6 "
                    "delta at the mode-existence edge for any Bz/B0, and must "
                    "also absorb the sheet's drift (~11 cells over your sigma=5 "
                    "run). 40 cells is about 5.7 delta at 7 cells per delta.")
p.add_argument("--skin-depth", type=float, default=1.0,
               help="Ion skin depth c/omega_pi in code units (default 1, which "
                    "is what rhoINIT[1] = 1 gives). RECORDED ONLY, for reference: "
                    "the spectra are normalised to delta instead, because "
                    "dBx/B0 = xi/delta makes (xi/delta)^2 the equivalent "
                    "(dB/B0)^2 and so keeps the tearing-to-kink ratio "
                    "dimensionally correct with no stray factor.")
p.add_argument("--delta", type=float, default=0.5,
               help="Current-sheet half-thickness in code units (CS_thickness in "
                    "the iPIC3D input; default 0.5). Sets the peak-search window "
                    "and the resonant-surface prediction. Used for reporting "
                    "the kink amplitude in units of delta and for the resonant- "
                    "surface prediction. Optional.")
p.add_argument("--guide-field", type=float, default=0.0,
               help="Bz/B0. Used ONLY to write the predicted resonant surface "
                    "y_res = delta*arctanh(-b*n/m) next to the measured y_peak, "
                    "and to report the mode-existence cutoff |n/m| < 1/b. It "
                    "does not affect any measured quantity.")
p.add_argument("--surface-search-delta", type=float, default=8.0,
               help="Restrict the Bx=0 search for the neutral surface to "
                    "+/-this many delta about the TRACKED sheet centre "
                    "(default 8). Without it the search spans the whole band "
                    "and can pick a crossing 10 delta out, which is neither the "
                    "sheet nor an island separatrix -- those sit at ~1.8 delta "
                    "for dpsi = 3.25. A genuine KINK displaces every x-column "
                    "at a given z together, so it trips in multiples of nxg; "
                    "SCATTERED far-out picks are spurious. Raise this only if "
                    "the surface reaches the window edge across whole z-slices.")
p.add_argument("--peak-search-delta", type=float, default=2.0,
               help="Restrict the peak-over-y search for P(m,n) to +/-this many "
                    "delta about the sheet. Plasmoids form on resonant surfaces "
                    "y_res = delta*arctanh(-b*n/m); half the mode wedge lies "
                    "within 0.55 delta of the neutral line and three quarters "
                    "within 0.97, so +/-2 delta covers every mode that is not "
                    "already marginal (beyond 2 delta the driving current is "
                    "under 2%% of peak). Searching the whole band instead lets "
                    "upstream noise win the max -- the max of 81 samples sits "
                    "~4.4x above the mean, against ~3.4x over 29. Requires "
                    "--delta; ignored with a warning otherwise.")
p.add_argument("--max-modes", type=int, default=10,
               help="Number of m and n modes written to the per-mode tables.")

p.add_argument("--cycle-start", type=int, default=0)
p.add_argument("--cycle-end", type=int, default=10**9)
p.add_argument("--cycle-step", type=int, default=1)
p.add_argument("--no-plot", dest="plot", action="store_false")
args = p.parse_args()

Lx = args.xmax - args.xmin
Ly = args.ymax - args.ymin
Lz = args.zmax - args.zmin
nxc, nyc, nzc = args.nxc, args.nyc, args.nzc
nxg, nyg, nzg = nxc + 1, nyc + 1, nzc + 1
dx, dy, dz = Lx / nxc, Ly / nyc, Lz / nzc


###! ============================================================
###! Tile bookkeeping (same conventions as Reconnection_rate_3D.py)
###! ============================================================

def proc_id_from_filename(path):
    m = re.search(r"proc(\d+)\.hdf$", os.path.basename(path))
    if m is None:
        raise ValueError(f"cannot parse proc id from {path}")
    return int(m.group(1))


def mapping_candidates(XLEN, YLEN, ZLEN):
    """The six permutations of which index varies fastest with proc_id."""
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

    return dict(A=A, B=B, C=C, D=D, E=E, F=F)


###! ============================================================
###! Setup: discover the files, the tile shape and the decomposition
###! ============================================================

if rank == 0:
    all_files = sorted(glob.glob(os.path.join(args.dir_data, "proc*.hdf")),
                       key=proc_id_from_filename)
    if not all_files:
        raise SystemExit(f"no proc*.hdf in {args.dir_data}")

    with h5py.File(all_files[0], "r") as f:
        cyc_names = sorted(f["fields/Bx"].keys(),
                           key=lambda s: int(s.split("_")[-1]))
        first = cyc_names[0]
        tile_shape = tuple(f[f"fields/Bx/{first}"].shape)
        ###! J is optional here: neither diagnostic needs it. It is probed only
        ###! so the header can record whether it was available.
        have_J = any(k in f for k in ("fields/Jz", "moments/Jz"))

    nx_t, ny_t, nz_t = tile_shape
    XLEN = nxc // (nx_t - 1)
    YLEN = nyc // (ny_t - 1)
    ZLEN = nzc // (nz_t - 1)
    if XLEN * YLEN * ZLEN != len(all_files):
        raise SystemExit(f"decomposition {XLEN}x{YLEN}x{ZLEN} = "
                         f"{XLEN*YLEN*ZLEN} != {len(all_files)} files")

    cycles = [int(s.split("_")[-1]) for s in cyc_names]
    cycles = [c for c in cycles
              if args.cycle_start <= c <= args.cycle_end
              and (c - args.cycle_start) % args.cycle_step == 0]

    os.makedirs(args.outdir, exist_ok=True)
    print(f"files           : {len(all_files)}   tile {tile_shape}", flush=True)
    print(f"decomposition   : {XLEN} x {YLEN} x {ZLEN}   mapping {args.mapping}",
          flush=True)
    print(f"grid            : {nxc} x {nyc} x {nzc} cells", flush=True)
    print(f"cycles          : {len(cycles)}", flush=True)
    print(f"J present       : {have_J}  (not needed by either diagnostic)",
          flush=True)
else:
    all_files = tile_shape = XLEN = YLEN = ZLEN = cycles = have_J = None

all_files  = comm.bcast(all_files, root=0)
tile_shape = comm.bcast(tile_shape, root=0)
XLEN       = comm.bcast(XLEN, root=0)
YLEN       = comm.bcast(YLEN, root=0)
ZLEN       = comm.bcast(ZLEN, root=0)
cycles     = comm.bcast(cycles, root=0)
have_J     = comm.bcast(have_J, root=0)

pid_to_ijk = mapping_candidates(XLEN, YLEN, ZLEN)[args.mapping]
local_files = all_files[rank::size]

nx_t, ny_t, nz_t = tile_shape
nx_c, ny_c, nz_c = nx_t - 1, ny_t - 1, nz_t - 1

###! Nominal sheet centres. The band is generous enough to absorb the drift;
###! a warning fires if the surface is ever found within 3 cells of an edge.
CS_CENTRES = [nyc // 4, 3 * nyc // 4]

###! Peak-search window, in band indices. Kept narrow on purpose: see
###! --peak-search-delta. Falls back to the whole band if delta is unknown.
###! (defined after NB below)

###! Nominal sheet centres. The band is generous enough to absorb the drift;
###! a warning fires if the surface is ever found within 3 cells of an edge.
###! The band must fit inside the domain on BOTH sides of BOTH sheets.
###! Otherwise the out-of-domain nodes are never written and enter the FFT as
###! silent zeros -- a step discontinuity that would smear power across every
###! mode. Reduce rather than abort, and say so once.
_room = min(min(CS_CENTRES), nyc - max(CS_CENTRES))
if args.band_cells > _room:
    if rank == 0:
        print(f"  WARNING: --band-cells {args.band_cells} exceeds the {_room} "
              f"cells available between a sheet and the domain edge; "
              f"reduced to {_room}.", flush=True)
    args.band_cells = _room

NB = 2 * args.band_cells + 1

###! Nominal window half-width, for the header only. The windows actually
###! used are recomputed per dump about the TRACKED sheet centre.
HALF_PK = max(1, min(int(round(args.peak_search_delta * args.delta / dy)),
                     args.band_cells))


###! ============================================================
###! Band assembly
###! ============================================================

def assemble_band(cycle, y0):
    """
    COLLECTIVE. Assemble Bx and By on the full (x, z) grid over the y-band
    [y0 - band_cells, y0 + band_cells], summed across the ranks that own the
    relevant tiles.

    float32 is used for the band: it is ~170 MB per field on a 769 x 71 x 769
    grid, and the FFT is done in double on rank 0 afterwards. Shared tile
    boundary planes are counted and divided out, exactly as in the flux script.
    """
    ylo, yhi = y0 - args.band_cells, y0 + args.band_cells

    Bx = np.zeros((nxg, NB, nzg), dtype=np.float32)
    By = np.zeros((nxg, NB, nzg), dtype=np.float32)
    cnt = np.zeros((nxg, NB, nzg), dtype=np.float32)

    for fp in local_files:
        i, j, k = pid_to_ijk(proc_id_from_filename(fp))

        ###! interior tiles drop their first plane, which duplicates the
        ###! previous tile's last plane
        xs = 0 if i == 0 else 1
        ys = 0 if j == 0 else 1
        zs = 0 if k == 0 else 1

        gx0, gy0, gz0 = i * nx_c + xs, j * ny_c + ys, k * nz_c + zs
        nxu, nyu, nzu = nx_t - xs, ny_t - ys, nz_t - zs

        ###! does this tile intersect the band at all?
        a = max(gy0, ylo)
        b = min(gy0 + nyu - 1, yhi)
        if a > b:
            continue
        js = a - gy0 + ys          ###! first y index to read inside the tile
        je = b - gy0 + ys + 1
        oy = a - ylo               ###! offset into the band array

        try:
            with h5py.File(fp, "r") as f:
                bx = np.asarray(f[f"fields/Bx/{cycle}"][xs:, js:je, zs:],
                                dtype=np.float32)
                by = np.asarray(f[f"fields/By/{cycle}"][xs:, js:je, zs:],
                                dtype=np.float32)
        except Exception as e:
            print(f"  rank {rank}: {os.path.basename(fp)} {cycle}: {e}",
                  flush=True)
            continue

        ny_read = je - js
        Bx[gx0:gx0+nxu, oy:oy+ny_read, gz0:gz0+nzu] += bx
        By[gx0:gx0+nxu, oy:oy+ny_read, gz0:gz0+nzu] += by
        cnt[gx0:gx0+nxu, oy:oy+ny_read, gz0:gz0+nzu] += 1.0

    comm.Allreduce(MPI.IN_PLACE, Bx, op=MPI.SUM)
    comm.Allreduce(MPI.IN_PLACE, By, op=MPI.SUM)
    comm.Allreduce(MPI.IN_PLACE, cnt, op=MPI.SUM)

    if rank != 0:
        return None, None

    ok = cnt > 0
    Bx[ok] /= cnt[ok]
    By[ok] /= cnt[ok]
    if not ok.all():
        print(f"  WARNING: {int((~ok).sum())} band nodes were never written "
              f"at {cycle}", flush=True)
    return Bx.astype(np.float64), By.astype(np.float64)


###! ============================================================
###! Neutral surface y_n(x, z)
###! ============================================================

def neutral_surface(Bx, c_track, half):
    """
    y_n(x, z) in CELL units relative to the band centre, from the Bx = 0
    crossing NEAREST the band centre in each (x, z) column.

    Nearest-to-centre, not steepest-gradient: above dpsi = 2*B0*delta the
    island separatrices are also Bx = 0 surfaces, and a steepest-gradient pick
    jumps onto them. Measured on a synthetic island, nearest-to-centre stays
    flat to 4e-16 at eps/(B0 delta) = 5 while steepest-gradient wanders by 1.0.

    Returns (y_n, n_multi, n_missing, n_edge).
    """
    nx, nb, nz = Bx.shape
    ###! search window follows the TRACKED centre, not the nominal one, so a
    ###! global drift of the sheet does not push the surface toward an edge
    mid = c_track
    klo, khi = max(0, mid - half), min(nb - 1, mid + half + 1)
    yn = np.full((nx, nz), np.nan)
    n_multi = n_missing = n_edge = 0

    for ix in range(nx):
        col = Bx[ix, klo:khi]              ###! (window, nz)
        s = np.sign(col)
        ###! exact-zero guard: a sheet sitting exactly on a node gives Bx = 0
        ###! there, and a strict sign-product test misses it entirely
        cross = (s[:-1] * s[1:] < 0) | (s[:-1] == 0)
        for iz in range(nz):
            idx = np.nonzero(cross[:, iz])[0]
            if idx.size == 0:
                n_missing += 1
                continue
            if idx.size > 1:
                n_multi += 1
            b0 = col[idx, iz]
            b1 = col[idx + 1, iz]
            den = b0 - b1
            good = den != 0
            if not good.any():
                n_missing += 1
                continue
            pos = idx[good] + klo + b0[good] / den[good]     ###! sub-cell
            yn[ix, iz] = pos[np.argmin(np.abs(pos - mid))] - mid
            ###! at the WINDOW edge, not the band edge: reaching it means the
            ###! surface genuinely left the plausible range
            if abs(yn[ix, iz]) > half - 3:
                n_edge += 1

    return yn, n_multi, n_missing, n_edge


###! ============================================================
###! Spectra
###! ============================================================

def kink_spectrum(yn):
    """
    Kink power per z-mode from the kx = 0 neutral-surface displacement.

    y_n is averaged over x FIRST, projecting onto kx = 0, and the resulting
    xi(z) is then transformed along z. This isolates the COHERENT kink rather
    than averaging the power of every x-dependent distortion.

    Returns (xi_n/delta)^2, DIMENSIONLESS.

    delta is the right yardstick here, not the skin depth, because the linear
    relation is dBx/B0 = xi/delta. Normalising by delta therefore makes
    (xi/delta)^2 the EQUIVALENT (dBx/B0)^2, so Psrf can be ratioed against
    P1D and Pob with no conversion factor -- the ratio is dimensionally
    correct by construction. Normalising by d_i instead would leave a stray
    (d_i/delta)^2 in every such ratio.

    Full rfft is computed; the caller slices to --max-modes.
    """
    core = yn[:nxc, :nzc]
    good = np.isfinite(core)
    if not good.all():
        ###! fill the few dead columns with the surface mean so the transform
        ###! stays defined; they are counted and reported separately
        core = np.where(good, core, np.nanmean(core))

    xi_z = core.mean(axis=0)                 ###! project onto kx = 0
    xi_z = xi_z - xi_z.mean()

    hat = np.fft.rfft(xi_z) / nzc
    ###! x2 folds the negative frequency back in, so |xi_n| is the PEAK
    ###! amplitude -- the same convention as the tearing spectrum
    amp = 2.0 * np.abs(hat)
    amp[0] *= 0.5
    return amp ** 2 * (dy / args.delta) ** 2


def kink_spectrum_fixed(Bx, B0, c_track, half_pk):
    """
    The same kink measured from Bx instead of from the surface: average Bx
    over x, remove the z-mean, transform along z, then average power over the
    +/-2 delta window.

    It reads LOW against kink_spectrum for two reasons, both real:
      * tanh saturation once the displacement approaches delta;
      * <Bx>_x is not B0*tanh(<xi>_x/delta) -- averaging a nonlinear function
        of a corrugated surface smears the profile. Measured on synthetic
        data: ratio 3.8 with no x-structure, 10x at 2 delta of it, 18x at 3.
    The gap is therefore a measure of the surface corrugation, not an error.
    """
    ###! window tracks the sheet, same as the tearing spectrum
    nb = Bx.shape[1]
    jlo = max(0, c_track - half_pk); jhi = min(nb, c_track + half_pk + 1)
    core = Bx[:nxc, jlo:jhi, :nzc]
    bx_z = core.mean(axis=0)                 ###! project onto kx = 0
    bx_z = bx_z - bx_z.mean(axis=1, keepdims=True)

    hat = np.fft.rfft(bx_z, axis=1) / nzc
    amp = 2.0 * np.abs(hat)
    amp[:, 0] *= 0.5
    return np.mean(amp ** 2, axis=0) / (B0 * B0)


def tearing_spectrum(By, B0, c_track, half_pk):
    """
    P(m, n) from By over the band: rfft in x, full fft in z, then the MEAN
    over the y-window rather than the peak.

    The mean has no extreme-value bias -- a max over N noise samples sits
    ~ln(N) above the mean, which inflated empty bins. The cost is that y_pk,
    the validation column, is no longer defined: it is returned as NaN.

    Returns P[m, n] = (dBy_mn / B0)^2, so sqrt(P) is the mode's PEAK AMPLITUDE
    relative to B0 -- matching the kink convention, which is what makes a
    tearing-to-kink ratio meaningful. Full spectrum is computed; the caller
    slices to --max-modes.
    """
    ###! The window is centred on the TRACKED sheet, not the nominal centre.
    ###! With a fixed window a sheet that drifts past +/-2 delta (10 cells at
    ###! delta = 0.5, which the sigma=5 run reached) puts the whole search in
    ###! the tanh wing: P(m,n) is then suppressed for reasons unrelated to
    ###! tearing, and y_pk merely reports the drift.
    nb = By.shape[1]
    jlo = max(0, c_track - half_pk)
    jhi = min(nb, c_track + half_pk + 1)

    core = By[:nxc, jlo:jhi, :nzc]
    hat = np.fft.rfft(core, axis=0)                  ###! m >= 0
    hat = np.fft.fft(hat, axis=2)                    ###! n SIGNED
    ###! /(nxc*nzc) then x2, so |hat| is the peak amplitude of that mode
    hat = 2.0 * hat / (nxc * nzc)

    pw = np.abs(hat) ** 2                            ###! (m, y, n)
    P = pw.mean(axis=1)                              ###! mean over the window
    ###! y_pk is the argmax POSITION, reported alongside the mean power. It
    ###! costs one line, changes no reported power, and is the check that
    ###! caught clipped rows before: it should sit at
    ###! delta*arctanh(-b*n/m), which is 0 for b = 0.
    ###! relative to the TRACKED sheet, so y_pk is a resonance offset and not
    ###! a drift measurement
    ypk = np.argmax(pw, axis=1) + jlo - c_track
    return P / (B0 ** 2), ypk


###! ============================================================
###! Main loop
###! ============================================================

recs = {0: [], 1: []}

for cyc in cycles:
    name = f"cycle_{cyc}"
    t = cyc / args.time_denom

    for cs, y0 in enumerate(CS_CENTRES):
        Bx, By = assemble_band(name, y0)
        if rank != 0:
            continue

        ###! B0 from the band edge: at +/-5 delta tanh is 0.9999, so the
        ###! largest |<Bx>_xz| in the band IS the asymptotic reconnecting field
        prof = Bx.mean(axis=(0, 2))
        B0 = float(np.abs(prof).max())
        if B0 <= 0:
            print(f"  WARNING: B0 = 0 at {name} CS{cs+1}; skipping", flush=True)
            continue

        ###! track the sheet inside the band from the x,z-averaged profile
        sgn = np.sign(prof)
        xr = np.nonzero((sgn[:-1] * sgn[1:] < 0) | (sgn[:-1] == 0))[0]
        c_track = (int(xr[np.argmax(np.abs(np.diff(prof))[xr])]) if xr.size
                   else args.band_cells)
        half = max(3, int(round(args.surface_search_delta * args.delta / dy)))

        yn, n_multi, n_missing, n_edge = neutral_surface(Bx, c_track, half)

        ###! peak-search half-width, recomputed here so both spectra use the
        ###! SAME window, centred on the tracked sheet
        half_pk = min(max(1, int(round(args.peak_search_delta * args.delta / dy))),
                      args.band_cells)

        Pk = kink_spectrum(yn)
        Pkf = kink_spectrum_fixed(Bx, B0, c_track, half_pk)
        Pt, ypk = tearing_spectrum(By, B0, c_track, half_pk)

        M = min(args.max_modes, Pt.shape[0] - 1)
        N = min(args.max_modes, nzc // 2)

        ###! n == 0 is the ordinary 2D tearing; everything else is oblique.
        ###! EVERY sum below is capped at m <= M and n <= N, the same range the
        ###! per-mode columns report. Summing to the Nyquist instead made the
        ###! two totals count wildly different numbers of bins -- 192 for
        ###! tot1D against 32304 for totOB on a 384^2 grid -- so 1D/OB was
        ###! biased by ~168x and read "oblique-dominated" purely from the mode
        ###! count. Capping both makes it at most M x N against M, i.e. N.
        P_1D = Pt[1:M+1, 0].copy()
        tot_1D = float(P_1D.sum())

        ###! ---- mode-existence wedge ----
        ###! An oblique mode is resonant only where tanh(y/delta) = -b*(n/m)
        ###! has a solution, i.e. |n/m| < 1/b. Outside that wedge there is NO
        ###! resonant surface and the bin holds only noise -- and since P is a
        ###! MAX over y, that noise is biased UP by ~ln(n_band) ~ 4x. Summing
        ###! it would inflate the oblique fraction, badly at large b: half the
        ###! (m,n) pairs are non-existent at b = 1, a quarter at b = 0.5.
        ###! b = 0 admits everything, so the mask is all-true there.
        ###! m, n are integer MODE NUMBERS: k_x = 2 pi m/Lx, k_z = 2 pi n/Lz.
        ###! The resonance condition is on k_z/k_x, which equals (n/m)*(Lx/Lz)
        ###! -- it reduces to n/m only for a box with Lx = Lz, which yours are.
        ###! Written out in full so a non-cubic x-z box stays correct.
        mm = np.arange(Pt.shape[0])[:, None]
        nn = np.fft.fftfreq(nzc, d=1.0 / nzc)[None, :]
        if args.guide_field > 0.0:
            with np.errstate(divide="ignore", invalid="ignore"):
                kratio = np.abs(nn * Lx / (np.where(mm == 0, np.nan, mm) * Lz))
                wedge = kratio < 1.0 / args.guide_field
            wedge = np.where(np.isnan(wedge), False, wedge)
        else:
            wedge = np.ones_like(Pt, dtype=bool)

        Pw = np.where(wedge, Pt, 0.0)

        ###! ---- POSITIVE n ONLY ----
        ###! Only the n > 0 half is reported, matching the MATLAB script, which
        ###! plots ff(:,2:nplot) and normalises over kx = kxmin:kxmin:pi/dx.
        ###!
        ###! NOTE this is NOT the same as an rfft. For a real 1D signal +k and
        ###! -k are conjugates and rfft loses nothing. Here, at fixed m > 0,
        ###! (m,+n) and (m,-n) are DIFFERENT physical modes -- opposite tilts,
        ###! resonating on opposite sides of the sheet -- so dropping n < 0
        ###! discards real information rather than redundancy. Fine as long as
        ###! the layer has no preferred tilt (exactly true at b = 0); at b > 0
        ###! the two halves can differ and half the oblique power is then
        ###! simply not counted.
        ###! m capped at M here too: Pob_n previously summed all 192 x-modes
        ###! while P1D_m was a single one, which is not a like-for-like column.
        P_ob_n = np.array([Pw[1:M+1, n].sum() for n in range(1, N + 1)])
        tot_ob = float(Pw[1:M+1, 1:N+1].sum())
        ###! the unmasked total is kept so nothing is hidden by the theory cut
        tot_ob_raw = float(Pt[1:M+1, 1:N+1].sum())

        ###! Dominant oblique mode, restricted to m <= M and |n| <= N: beyond
        ###! the tabulated range the modes are not well resolved, so an argmax
        ###! there reports where noise happened to peak, not a real mode.
        sub = Pw[1:M+1, :].copy()
        nvals = np.fft.fftfreq(nzc, d=1.0 / nzc).astype(int)
        sub[:, (nvals < 1) | (nvals > N)] = 0.0     ###! positive n only
        if sub.max() > 0:
            im, inn = np.unravel_index(np.argmax(sub), sub.shape)
            ###! ypk is in cells about the sheet; dy converts to code length
            ypk_dom = float(ypk[im + 1, inn]) * dy / args.delta
            m_dom, n_dom = im + 1, int(nvals[inn])
        else:
            ypk_dom, m_dom, n_dom = np.nan, 0, 0

        ###! Dominant kink mode, from the same 1..N range the columns report.
        ###! Reported for BOTH estimators: if they disagree, the surface and
        ###! the field are peaking at different wavelengths, which is worth
        ###! knowing before either is quoted.
        ks, kf = Pk[1:N+1], Pkf[1:N+1]
        nk_srf = int(np.argmax(ks)) + 1 if ks.size and ks.max() > 0 else 0
        nk_fix = int(np.argmax(kf)) + 1 if kf.size and kf.max() > 0 else 0
        ###! k*delta of the dominant surface mode: the physical label. DKI is
        ###! kinetic and lives near k*delta ~ O(1); the MHD kink is k*delta << 1.
        kd_srf = 2.0 * np.pi * nk_srf * args.delta / Lz if nk_srf else np.nan

        recs[cs].append(dict(cycle=cyc, t=t, B0=B0,
                             nk_srf=nk_srf, nk_fix=nk_fix, kd_srf=kd_srf,
                             Pk=Pk[1:N+1], Pkf=Pkf[1:N+1],
                             P_1D=P_1D, P_ob_n=P_ob_n,
                             tot_1D=tot_1D, tot_ob=tot_ob,
                             tot_ob_raw=tot_ob_raw,
                             ypk_dom=ypk_dom, m_dom=m_dom, n_dom=n_dom,
                             n_multi=n_multi, n_missing=n_missing,
                             n_edge=n_edge))

    if rank == 0:
        print(f"  {name} done", flush=True)


###! ============================================================
###! Output
###! ============================================================

def average_sheets():
    """
    Mean of CS1 and CS2, cycle by cycle. Matches the 0.5*(CS1 + CS2) convention
    already used in your mode script.

    Every quantity averaged here is a POWER, which averages cleanly: positive,
    and the mean is linear.
    """
    by_cyc = {}
    for cs in (0, 1):
        for x in recs[cs]:
            by_cyc.setdefault(x["cycle"], {})[cs] = x

    out = []
    for cyc in sorted(by_cyc):
        pair = by_cyc[cyc]
        if len(pair) < 2:
            print(f"  WARNING: cycle {cyc} present for only one sheet; "
                  f"omitted from the average", flush=True)
            continue
        a, b = pair[0], pair[1]
        out.append(dict(
            cycle=cyc, t=a["t"],
            Pk=0.5*(a["Pk"] + b["Pk"]),
            Pkf=0.5*(a["Pkf"] + b["Pkf"]),
            ###! recomputed from the AVERAGED spectra, not averaged as indices
            nk_srf=int(np.argmax(0.5*(a["Pk"] + b["Pk"]))) + 1,
            nk_fix=int(np.argmax(0.5*(a["Pkf"] + b["Pkf"]))) + 1,
            kd_srf=2.0*np.pi*(int(np.argmax(0.5*(a["Pk"] + b["Pk"]))) + 1)
                   * args.delta / Lz,
            P_1D=0.5*(a["P_1D"] + b["P_1D"]),
            P_ob_n=0.5*(a["P_ob_n"] + b["P_ob_n"]),
            tot_1D=0.5*(a["tot_1D"] + b["tot_1D"]),
            tot_ob=0.5*(a["tot_ob"] + b["tot_ob"]),
            tot_ob_raw=0.5*(a["tot_ob_raw"] + b["tot_ob_raw"]),
            ypk_dom=a["ypk_dom"],
            m_dom=a["m_dom"], n_dom=a["n_dom"],
            n_multi=a["n_multi"] + b["n_multi"],
            n_missing=a["n_missing"] + b["n_missing"]))
    return out


def header(fh, what, cs):
    tag = "MEAN of CS1 and CS2" if cs is None else f"CS{cs+1}"
    fh.write(f"# {what} -- {tag}\n")
    fh.write(f"# dir          = {args.dir_data}\n")
    fh.write(f"# cells        = {nxc} x {nyc} x {nzc}   mapping = {args.mapping}\n")
    fh.write(f"# spacing      = dx={dx:.8g} dy={dy:.8g} dz={dz:.8g}\n")
    fh.write(f"# time         = cycle / {args.time_denom}   [omega_p^-1]\n")
    cen = "both sheets" if cs is None else f"y index {CS_CENTRES[cs]}"
    fh.write(f"# band         = +/-{args.band_cells} cells "
             f"({args.band_cells*dy:.2f} c/wp) about {cen}\n")
    fh.write(f"# delta        = {args.delta:g} = {args.delta/args.skin_depth:g} d_i "
             f"(band = +/-{args.band_cells*dy/args.delta:.1f} delta)\n")
    fh.write(f"# spectra are normalised to DELTA, so the kink and tearing powers\n")
    fh.write(f"# are directly ratio-able. d_i = {args.skin_depth:g} is recorded only for\n")
    fh.write(f"# reference.\n")
    if args.guide_field:
        fh.write(f"# Bz/B0        = {args.guide_field:g}  -> oblique modes "
                 f"exist only for |k_z/k_x| < {1.0/args.guide_field:.2f},\n")
        fh.write(f"#                i.e. |n/m| < {Lz/(Lx*args.guide_field):.2f} "
                 f"for this box (Lx={Lx:g}, Lz={Lz:g})\n")
    fh.write("#\n")


def write_kink(r, cs, path):
    N = len(r[0]["Pk"])
    with open(path, "w") as fh:
        header(fh, "KINK modes", cs)
        fh.write("# COLUMNS\n")
        fh.write("#  cycle, time   dump number and t*omega_p = cycle/time_denom.\n")
        fh.write("#  Psrf_n   (xi_n/delta)^2 for z-mode n, from the neutral SURFACE\n")
        fh.write("#           y_n(x,z); sqrt(Psrf_n) is the sheet displacement in\n")
        fh.write("#           units of the half-thickness. Exact at any amplitude.\n")
        fh.write("#           Since dBx/B0 = xi/delta in the linear limit, this IS\n")
        fh.write("#           the equivalent (dB/B0)^2 and can be ratioed against\n")
        fh.write("#           P1D or Pob DIRECTLY, with no conversion factor.\n")
        fh.write("#  Pfix_n   (dBx_n/B0)^2 for the same mode, from FFT_z of Bx on\n")
        fh.write("#           a FIXED y-plane. Matches Psrf_n while the kink is\n")
        fh.write("#           small, then reads low as tanh saturates: 24% at a\n")
        fh.write("#           1-delta displacement, 75% at 4 delta.\n")
        fh.write("#  n_multi  Columns with more than one Bx = 0 crossing. Island\n")
        fh.write("#           separatrices become crossings above dpsi = 2*B0*delta,\n")
        fh.write("#           and the one NEAREST the tracked centre is used.\n")
        fh.write("#  n_missing Columns with no crossing at all, filled with the\n")
        fh.write("#           surface mean. Costs ~0.15% on the amplitude at 0.16%\n")
        fh.write("#           dead, ~1% at 1%.\n")
        fh.write(f"#  n_srf    Dominant z-mode of Psrf, 1..{N}. Recomputed from the\n")
        fh.write("#           sheet-averaged spectrum, not averaged as an index.\n")
        fh.write(f"#  n_fix    Dominant z-mode of Pfix, 1..{N}. If it differs from\n")
        fh.write("#           n_srf the surface and the field are peaking at\n")
        fh.write("#           different wavelengths -- check before quoting either.\n")
        fh.write("#  k_delta  k_z*delta of the n_srf mode = 2*pi*n_srf*delta/Lz.\n")
        fh.write("#           The physical label: DKI is kinetic and sits near\n")
        fh.write("#           k*delta ~ O(1), the MHD kink at k*delta << 1. Use\n")
        fh.write("#           this rather than n, which does not transfer between\n")
        fh.write("#           runs of different box size.\n")
        fh.write("#\n")
        c1 = "".join(f"{'Psrf_'+str(n):>14s}" for n in range(1, N + 1))
        c2 = "".join(f"{'Pfix_'+str(n):>14s}" for n in range(1, N + 1))
        fh.write(f"# {'cycle':>8s} {'time':>11s}{c1}{c2}{'n_multi':>10s}"
                 f"{'n_missing':>11s}{'n_srf':>7s}{'n_fix':>7s}{'k_delta':>10s}\n")
        for x in r:
            v1 = "".join(f"{v:>14.6e}" for v in x["Pk"])
            v2 = "".join(f"{v:>14.6e}" for v in x["Pkf"])
            fh.write(f"  {x['cycle']:>8d} {x['t']:>11.4f}{v1}{v2}"
                     f"{x['n_multi']:>10d}{x['n_missing']:>11d}"
                     f"{x['nk_srf']:>7d}{x['nk_fix']:>7d}{x['kd_srf']:>10.4f}\n")


def write_tearing(r, cs, path):
    M = len(r[0]["P_1D"]); N = len(r[0]["P_ob_n"])
    with open(path, "w") as fh:
        header(fh, "TEARING: 1D (n=0) and OBLIQUE (n!=0) from By", cs)
        fh.write("# COLUMNS\n")
        fh.write("#  cycle, time   dump number and t*omega_p = cycle/time_denom.\n")
        fh.write("#  P1D_m    Power in the ordinary 2D tearing mode of x-mode m\n")
        fh.write("#           (n = 0), as max over y of |FFT_x FFT_z By|^2 / B0^2.\n")
        fh.write("#           Dimensionless; sqrt(P1D_m) is the PEAK dBy/B0 of\n")
        fh.write("#           that mode, same amplitude convention as the kink.\n")
        fh.write(f"#  Pob_n    Oblique power at z-mode n > 0, summed over m = 1..{M}\n")
        fh.write("#           (the same range the P1D_m columns cover).\n")
        fh.write("#           Same units. n != 0 means the mode is tilted out of\n")
        fh.write("#           the x-y plane. ONLY POSITIVE n is counted, matching\n")
        fh.write("#           the MATLAB script. At fixed m the (m,+n) and (m,-n)\n")
        fh.write("#           modes are physically distinct (opposite tilts), so\n")
        fh.write("#           this drops real information, not redundancy: at\n")
        fh.write("#           b = 0 the two halves are equal by symmetry, but at\n")
        fh.write("#           b > 0 the discarded half need not match.\n")
        fh.write(f"#  tot1D    P1D summed over m = 1..{M}.\n")
        fh.write(f"#  totOB    Oblique power summed over m = 1..{M}, n = 1..{N},\n")
        fh.write("#           only inside the resonance wedge |k_z/k_x| < 1/b;\n")
        fh.write("#           outside it no resonant surface exists and the bin\n")
        fh.write("#           holds only noise. b = 0 admits every mode.\n")
        fh.write("#  totOBraw Same sum with the wedge cut removed, so the effect\n")
        fh.write("#           of the cut stays visible. Equals totOB when b = 0.\n")
        fh.write("#  1D/OB    tot1D / totOB. Above 1 the layer is 2D-dominated,\n")
        fh.write("#           below 1 it is oblique-dominated. Both sums are\n")
        fh.write(f"#           capped at the same {M} x-modes, so this is not\n")
        fh.write("#           inflated by the oblique side simply having more\n")
        fh.write(f"#           bins -- it still counts {N}x more (n = 1..{N} vs\n")
        fh.write("#           n = 0 alone), so divide by that for a per-mode view.\n")
        fh.write("#  m_dom    x-mode number of the strongest oblique mode, i.e.\n")
        fh.write(f"#           roughly the island count along x. Capped at {M}\n")
        fh.write("#           because higher modes are not well resolved.\n")
        fh.write(f"#  n_dom    z-mode number of that same mode, 1..{N} (positive\n")
        fh.write("#           half only, as above).\n")
        fh.write("#  y_pk     Distance from the SHEET at which that mode's power\n")
        fh.write("#           peaks, in units of DELTA (positive = away from the\n")
        fh.write("#           domain centre). VALIDATION COLUMN: in these units it\n")
        fh.write("#           should equal arctanh(-b*n/m) directly, with no\n")
        fh.write("#           scaling -- 0 for b = 0. A value\n")
        fh.write(f"#           pinned at the search edge ({HALF_PK*dy/args.delta:.2f} delta) means the\n")
        fh.write("#           peak was clipped and that row is noise, not a mode.\n")
        fh.write("#           The window TRACKS the sheet, so this is a resonance\n")
        fh.write("#           offset and not a drift.\n")
        fh.write("#\n")
        c1 = "".join(f"{'P1D_'+str(m):>14s}" for m in range(1, M + 1))
        c2 = "".join(f"{'Pob_'+str(n):>14s}" for n in range(1, N + 1))
        fh.write(f"# {'cycle':>8s} {'time':>11s}{c1}{c2}"
                 f"{'tot1D':>14s}{'totOB':>14s}{'totOBraw':>14s}{'1D/OB':>12s}"
                 f"{'m_dom':>7s}{'n_dom':>7s}{'y_pk':>11s}\n")
        for x in r:
            v1 = "".join(f"{v:>14.6e}" for v in x["P_1D"])
            v2 = "".join(f"{v:>14.6e}" for v in x["P_ob_n"])
            ratio = x["tot_1D"] / x["tot_ob"] if x["tot_ob"] > 0 else np.inf
            fh.write(f"  {x['cycle']:>8d} {x['t']:>11.4f}{v1}{v2}"
                     f"{x['tot_1D']:>14.6e}{x['tot_ob']:>14.6e}"
                     f"{x['tot_ob_raw']:>14.6e}{ratio:>12.4e}"
                     f"{x['m_dom']:>7d}{x['n_dom']:>7d}"
                     f"{x['ypk_dom']:>11.4f}\n")


YMIN = 1e-8          ###! shared lower limit on all three panels


def make_figure(r, cs, path):
    """
    2 x 2: tearing on top (1D, oblique), the two kink measures underneath.
    All four are dimensionless and share one y-axis. No legend -- with 10
    curves it would cover more of the axes than it explains, and the colour
    ramp already orders the modes.
    """
    if not HAVE_MPL:
        return None
    t = np.array([x["t"] for x in r])
    A1 = np.array([x["P_1D"] for x in r])       ###! tearing, n = 0
    A2 = np.array([x["P_ob_n"] for x in r])     ###! tearing, n != 0
    A3 = np.array([x["Pk"] for x in r])         ###! kink from the surface
    A4 = np.array([x["Pkf"] for x in r])        ###! kink from the fixed plane

    ###! All four are now DIMENSIONLESS and on the same footing, so one shared
    ###! axis makes the competition readable at a glance -- that is the whole
    ###! point of normalising the kink by delta. YMIN clips the round-off
    ###! floor; empty bins sit at 1e-30 and would stretch the decades.
    top = max(YMIN * 10, 3.0 * max(np.nanmax(A) for A in (A1, A2, A3, A4)))

    fig = plt.figure(figsize=(11, 8), dpi=200)
    cmap = plt.cm.jet

    panels = ((A1, "1D Tearing"),
              (A2, "Oblique Tearing"),
              (A3, r"Kink from $y_n(x,z)$"),
              (A4, r"Kink from $B_x$, fixed plane"))

    for j, (A, ttl) in enumerate(panels):
        plt.subplot(2, 2, j + 1)

        nm = A.shape[1]
        for i in range(nm):
            plt.semilogy(t, np.maximum(A[:, i], YMIN),
                         color=cmap(i / max(nm - 1, 1)), lw=1)

        plt.title(ttl, fontsize=12)
        plt.ylim(YMIN, top)
        plt.tick_params(labelsize=12, length=5)
        plt.minorticks_off()

        ###! shared axes without the axes objects: labels on the left column
        ###! and the bottom row only
        if j % 2 == 0:
            plt.ylabel("FFT Power", fontsize=12)
        else:
            plt.tick_params(labelleft=False)
        if j >= 2:
            plt.xlabel(r"$t/\omega_{\mathrm{p,i}}^{-1}$", fontsize=12)
        else:
            plt.tick_params(labelbottom=False)

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


if rank == 0:
    ###! Surface-quality summary. Reported ONCE, and only when it matters:
    ###!   edge picks   a 10-delta pick against a ~0.5 c/wp true displacement
    ###!                injects ~33% of the signal variance -> always worth saying
    ###!   dead columns filled with the surface mean, which costs -0.15% at
    ###!                0.16% dead and -1% at 1% -> silent below 1%
    for cs in (0, 1):
        if not recs[cs]:
            continue
        e = max(x["n_edge"] for x in recs[cs])
        m = max(x["n_missing"] for x in recs[cs])
        fe, fm = e / nxg, m / (nxg * nzg)
        if e:
            note = ("whole z-slices -> looks like a REAL kink, raise "
                    "--surface-search-delta" if fe > 0.5 else
                    "scattered -> spurious picks, correctly excluded")
            print(f"  CS{cs+1}: worst dump had {e} columns at the search-window "
                  f"edge ({fe:.2f} z-slices) -- {note}", flush=True)
        if fm > 0.01:
            tag = "noticeable, ~1% low" if fm < 0.05 else "SPECTRUM UNRELIABLE"
            print(f"  CS{cs+1}: worst dump had {m} columns with no Bx=0 crossing "
                  f"({100*fm:.1f}%, {tag})", flush=True)

    avg = average_sheets()
    if avg:
        fk = os.path.join(args.outdir, "kink_modes_avg.txt")
        ft = os.path.join(args.outdir, "tearing_modes_avg.txt")
        write_kink(avg, None, fk);     print(f"Wrote {fk}", flush=True)
        write_tearing(avg, None, ft);  print(f"Wrote {ft}", flush=True)
        if args.plot:
            fp = make_figure(avg, None,
                             os.path.join(args.outdir, "modes_avg.png"))
            if fp:
                print(f"Wrote {fp}", flush=True)

        rr = np.array([x["tot_1D"] / x["tot_ob"] if x["tot_ob"] > 0 else np.inf
                       for x in avg])
        print(f"  mean of CS1 & CS2: median tot1D/totOB = {np.nanmedian(rr):.3g}"
              f"  ({'2D-dominated' if np.nanmedian(rr) > 1 else 'oblique-dominated'})",
              flush=True)