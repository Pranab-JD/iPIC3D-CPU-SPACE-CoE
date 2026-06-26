import numpy as np
import glob, re, os
import argparse
from mpi4py import MPI


# ============================================================
# MPI
# ============================================================
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


# ============================================================
# Read VTK
# ============================================================
def read_vtk_vector(filename):
    with open(filename, "r") as f:
        lines = f.readlines()

    dims = None
    spacing = None
    origin = None
    data_start = None
    npoints = None
    header = []

    for i, line in enumerate(lines):
        header.append(line)

        if line.startswith("DIMENSIONS"):
            dims = tuple(map(int, line.split()[1:4]))
        elif line.startswith("SPACING"):
            spacing = tuple(map(float, line.split()[1:4]))
        elif line.startswith("ORIGIN"):
            origin = tuple(map(float, line.split()[1:4]))
        elif line.startswith("POINT_DATA"):
            npoints = int(line.split()[1])
        elif line.startswith("VECTORS"):
            data_start = i + 1
            break

    nx, ny, nz = dims
    data_lines = lines[data_start:data_start + npoints]
    raw = np.loadtxt(data_lines, dtype=np.float64)
    J = raw.reshape((nx, ny, nz, 3), order="F")

    return J, dims, origin, spacing, header


# ============================================================
# Write VTK
# ============================================================
def write_vtk_vector(filename, J, header):
    with open(filename, "w") as f:
        for line in header:
            f.write(line)

        flat = J.reshape((-1, 3), order="F")
        np.savetxt(f, flat, fmt="%.8e")


# ============================================================
# Smoothing (vectorized)
# ============================================================
def smooth_3d(data, num_smoothings=1):
    for _ in range(num_smoothings):

        c  = data[1:-1,1:-1,1:-1]

        f  = (data[:-2,1:-1,1:-1] + data[2:,1:-1,1:-1] +
              data[1:-1,:-2,1:-1] + data[1:-1,2:,1:-1] +
              data[1:-1,1:-1,:-2] + data[1:-1,1:-1,2:])

        e  = (data[:-2,:-2,1:-1] + data[2:,:-2,1:-1] +
              data[:-2,2:,1:-1] + data[2:,2:,1:-1] +
              data[:-2,1:-1,:-2] + data[2:,1:-1,:-2] +
              data[1:-1,:-2,:-2] + data[1:-1,2:,:-2] +
              data[:-2,1:-1,2:] + data[2:,1:-1,2:] +
              data[1:-1,:-2,2:] + data[1:-1,2:,2:])

        co = (data[:-2,:-2,:-2] + data[2:,:-2,:-2] +
              data[:-2,2:,:-2] + data[2:,2:,:-2] +
              data[:-2,:-2,2:] + data[2:,:-2,2:] +
              data[:-2,2:,2:] + data[2:,2:,2:])

        data[1:-1,1:-1,1:-1] = 0.015625 * (8.0*c + 4.0*f + 2.0*e + co)

    return data


def smooth_vector_field(J, num_smoothings=1):
    J_s = np.empty_like(J)
    for comp in range(3):
        J_s[..., comp] = smooth_3d(J[..., comp].copy(), num_smoothings)
    return J_s


# ============================================================
# Extract cycle
# ============================================================
def get_cycle(fname):
    m = re.search(r'cycle_(\d+)', fname)
    return int(m.group(1)) if m else -1


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("indir", type=str)
    parser.add_argument("--outdir", type=str, default=None)
    parser.add_argument("--nsmooth", type=int, default=16)

    args = parser.parse_args()

    indir = args.indir.rstrip("/")
    outdir = args.outdir if args.outdir else indir + "/smoothed"
    nsmooth = args.nsmooth

    if rank == 0:
        os.makedirs(outdir, exist_ok=True)

    comm.Barrier()

    files = sorted(glob.glob(f"{indir}/*.vtk"), key=get_cycle)

    # distribute files
    files_local = files[rank::size]

    if rank == 0:
        print(f"Found {len(files)} files in {indir}", flush=True)

    for fpath in files_local:

        cycle = get_cycle(fpath)
        print(f"Processing cycle {cycle}", flush=True)

        J, dims, origin, spacing, header = read_vtk_vector(fpath)

        J_s = smooth_vector_field(J, num_smoothings=nsmooth)

        basename = os.path.basename(fpath)
        outname = os.path.join(outdir, basename)

        write_vtk_vector(outname, J_s, header)