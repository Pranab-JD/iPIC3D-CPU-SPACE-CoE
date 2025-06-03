"""
Created on Wed Jun 17:00 2025

@author: Pranab JD

Description: Plot magnetic field (2D)
"""

import numpy as np
from mpi4py import MPI
import os, glob, h5py, argparse
import matplotlib.pyplot as plt

from datetime import datetime

startTime = datetime.now()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

###* =================================================================== *###

###* Argument parser
parser = argparse.ArgumentParser(description= "Plot 2D charge density from iPIC3D output data")

parser.add_argument("dir_data",   type=str, help="Directory where proc.hdf files are stored, e.g., './data_reconnection/'")
parser.add_argument("time_cycle", type=str, help="Time cycle to plot, e.g., 'cycle_100'")

parser.add_argument("xlen", type=int, help="Number of MPI processes along X (must match simulation)")
parser.add_argument("ylen", type=int, help="Number of MPI processes along Y (must match simulation)")
parser.add_argument("zlen", type=int, help="Number of MPI processes along Z (must match simulation)")

args = parser.parse_args()

###* Directory where proc.hdf files are saved (plots are saved to the same directory)
dir_data = args.dir_data

###* Time cycle when data is to plotted 
time_cycle = args.time_cycle

###! MPI topology (must match simulation)
XLEN, YLEN, ZLEN = args.xlen, args.ylen, args.zlen
num_expected_files = XLEN * YLEN * ZLEN

###* =================================================================== *###

# def get_mpi_dtype(numpy_dtype):
#     """
#     Map NumPy dtype to corresponding MPI datatype.
#     """
#     if numpy_dtype == np.float64:
#         return MPI.DOUBLE
#     elif numpy_dtype == np.float32:
#         return MPI.FLOAT
#     elif numpy_dtype == np.int32:
#         return MPI.INT
#     elif numpy_dtype == np.int64:
#         return MPI.LONG
#     else:
#         raise TypeError(f"Unsupported dtype: {numpy_dtype}")

###? Read and process data
if rank == 0:
    print("Plotting magnetic field at ", time_cycle, " with ", size,  " MPI ranks\n")

###* Discover all HDF5 files
all_hdf_files = sorted(glob.glob(os.path.join(dir_data, "proc*.hdf")))
if rank == 0:
   print("Expected ", num_expected_files, "files, found ", len(all_hdf_files), "\n")

###* Broadcast number of local grid cells
with h5py.File(all_hdf_files[0], "r") as f:
    sample = np.array(f["fields/Bx/" + time_cycle])
    nx_local, ny_local, nz = sample.shape
    # dtype_in_file = sample.dtype

###* Define global size
nx_global = XLEN * nx_local
ny_global = YLEN * ny_local

###* Divide files among ranks (chunked distribution)
local_files = all_hdf_files[rank::size]
if rank == 0:
    print("Processing ", len(all_hdf_files), " files with ", size, " MPI tasks")

###* Local storage (per MPI task)
local_data_X = np.zeros((nx_global, ny_global))
local_data_Y = np.zeros((nx_global, ny_global))
local_data_Z = np.zeros((nx_global, ny_global))

###* Process assigned files
if local_files:
    for file_path in local_files:
        
        rank_id = int(os.path.basename(file_path).replace("proc", "").replace(".hdf", ""))

        i = rank_id // YLEN
        j = rank_id % YLEN
        x0 = i * nx_local
        y0 = j * ny_local

        with h5py.File(file_path, "r") as f:

            Bx_data = np.array(f["fields/Bx/" + time_cycle])
            By_data = np.array(f["fields/By/" + time_cycle])
            Bz_data = np.array(f["fields/Bz/" + time_cycle])

            local_data_X[x0:x0 + nx_local, y0:y0 + ny_local] = Bx_data[:, :, 0]
            local_data_Y[x0:x0 + nx_local, y0:y0 + ny_local] = By_data[:, :, 0]
            local_data_Z[x0:x0 + nx_local, y0:y0 + ny_local] = Bz_data[:, :, 0]

###* =================================================================== *###

###? Gather results at root MPI process
Bx = None; By = None; Bz = None
if rank == 0:
    Bx = np.zeros((nx_global, ny_global))
    By = np.zeros((nx_global, ny_global))
    Bz = np.zeros((nx_global, ny_global))

# mpi_dtype = get_mpi_dtype(dtype_in_file)

comm.Reduce(local_data_X, Bx, op=MPI.SUM, root=0)
comm.Reduce(local_data_Y, By, op=MPI.SUM, root=0)
comm.Reduce(local_data_Z, Bz, op=MPI.SUM, root=0)

###* =================================================================== *###

###* Plot 2D on root MPI process
if rank == 0:

    fig = plt.figure(figsize = (14, 4), dpi = 250)

    plt.subplot(1, 3, 1)
    plt.imshow(Bx, origin='lower', cmap='seismic', aspect = "auto")
    plt.xlabel("X", fontsize = 16); plt.ylabel("Y", fontsize = 16)
    plt.tick_params(axis = 'x', which = 'major', labelsize = 12, length = 6)
    plt.tick_params(axis = 'y', which = 'major', labelsize = 12, length = 6)
    plt.title("Bx", fontsize = 18); plt.colorbar()

    plt.subplot(1, 3, 2)
    plt.imshow(By, origin='lower', cmap='seismic', aspect = "auto")
    plt.xlabel("X", fontsize = 16); plt.ylabel("Y", fontsize = 16)
    plt.tick_params(axis = 'x', which = 'major', labelsize = 12, length = 6)
    plt.tick_params(axis = 'y', which = 'major', labelsize = 12, length = 6)
    plt.title("By", fontsize = 18); plt.colorbar()

    plt.subplot(1, 3, 3)
    plt.imshow(Bz, origin='lower', cmap='seismic', aspect = "auto")
    plt.xlabel("X", fontsize = 16); plt.ylabel("Y", fontsize = 16)
    plt.tick_params(axis = 'x', which = 'major', labelsize = 12, length = 6)
    plt.tick_params(axis = 'y', which = 'major', labelsize = 12, length = 6)
    plt.title("Bz", fontsize = 18); plt.colorbar()
    
    fig.tight_layout()
    plt.savefig(dir_data + "B_field_" + time_cycle + ".png")
    plt.close()

###* =================================================================== *###

    print()
    print("Plots are saved in ", dir_data)
    print()
    print("Complete .....", "Time Elapsed = ", datetime.now() - startTime)
    print()
