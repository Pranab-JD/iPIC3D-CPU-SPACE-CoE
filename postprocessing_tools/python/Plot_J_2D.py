"""
Created on Wed May 31 21:00 2025

@author: Pranab JD
"""

import os
import glob, h5py
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime

startTime = datetime.now()

###* =================================================================== *###

dir_data = "./data_DH/"
time_cycle = "cycle_30"

###? MPI topology (must match simulation)
XLEN = 10; YLEN = 10
num_expected_files = XLEN * YLEN

print("Plotting magnetic field at", time_cycle, "\n")

###* Discover all processor files
hdf_files = sorted(glob.glob(os.path.join(dir_data, "proc*.hdf")))
assert len(hdf_files) == num_expected_files, "Mismatch between expected and found HDF5 files."

###* Read one file to get local shape
with h5py.File(hdf_files[0], "r") as f:
    sample = np.array(f["moments/Jx/" + time_cycle])
    nx_local, ny_local, nz = sample.shape

###* Define global domain size
nx_global = XLEN * nx_local
ny_global = YLEN * ny_local

Jx = np.zeros((nx_global, ny_global))
Jy = np.zeros((nx_global, ny_global))
Jz = np.zeros((nx_global, ny_global))

for rank in range(num_expected_files):

    file_path = os.path.join(dir_data, f"proc{rank}.hdf")
    
    with h5py.File(file_path, "r") as f:

        Jx_data = np.array(f["moments/Jx/" + time_cycle])
        Jy_data = np.array(f["moments/Jy/" + time_cycle])
        Jz_data = np.array(f["moments/Jz/" + time_cycle])

        ## Determine processor position in 2D MPI grid
        i = rank // YLEN  # x index
        j = rank % YLEN   # y index

        x0 = i * nx_local
        y0 = j * ny_local

        Jx[x0:x0 + nx_local, y0:y0 + ny_local] = Jx_data[:, :, 0]
        Jy[x0:x0 + nx_local, y0:y0 + ny_local] = Jy_data[:, :, 0]
        Jz[x0:x0 + nx_local, y0:y0 + ny_local] = Jz_data[:, :, 0]


###? Plot (2D)
fig = plt.figure(figsize = (14, 4), dpi = 250)

plt.subplot(1, 3, 1)
plt.imshow(Jx, origin='lower', cmap='seismic', aspect = "auto")
plt.xlabel("X", fontsize = 16); plt.ylabel("Y", fontsize = 16)
plt.tick_params(axis = 'x', which = 'major', labelsize = 12, length = 6)
plt.tick_params(axis = 'y', which = 'major', labelsize = 12, length = 6)
plt.title("Jx", fontsize = 18); plt.colorbar()

plt.subplot(1, 3, 2)
plt.imshow(Jy, origin='lower', cmap='seismic', aspect = "auto")
plt.xlabel("X", fontsize = 16); plt.ylabel("Y", fontsize = 16)
plt.tick_params(axis = 'x', which = 'major', labelsize = 12, length = 6)
plt.tick_params(axis = 'y', which = 'major', labelsize = 12, length = 6)
plt.title("Jy", fontsize = 18); plt.colorbar()

plt.subplot(1, 3, 3)
plt.imshow(Jz, origin='lower', cmap='seismic', aspect = "auto")
plt.xlabel("X", fontsize = 16); plt.ylabel("Y", fontsize = 16)
plt.tick_params(axis = 'x', which = 'major', labelsize = 12, length = 6)
plt.tick_params(axis = 'y', which = 'major', labelsize = 12, length = 6)
plt.title("Jz", fontsize = 18); plt.colorbar()

fig.tight_layout(rect = [0.0, 0.0, 1.0, 1.0])
plt.savefig(dir_data + "Current_" + time_cycle + ".png")
plt.savefig(dir_data + "Current_" + time_cycle + ".eps")
plt.close()

###* =================================================================== *###

print()
print("Complete .....", "Time Elapsed = ", datetime.now() - startTime, "\n\n")