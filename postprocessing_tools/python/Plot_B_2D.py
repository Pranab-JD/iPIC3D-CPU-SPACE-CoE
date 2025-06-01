"""
Created on Wed May 31 21:00 2025

@author: Pranab JD

Description: Plot the magnetic field (2D)
"""

import os
import glob, h5py
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime

startTime = datetime.now()

###* =================================================================== *###

dir_data = "./data_DH/"
time_cycle = "cycle_100"

###? MPI topology (must match simulation)
XLEN = 10; YLEN = 10
num_expected_files = XLEN * YLEN

print("Plotting magnetic field at", time_cycle, "\n")

###* Discover all processor files
hdf_files = sorted(glob.glob(os.path.join(dir_data, "proc*.hdf")))
assert len(hdf_files) == num_expected_files, "Mismatch between expected and found HDF5 files."

###* Read one file to get local shape
with h5py.File(hdf_files[0], "r") as f:
    sample = np.array(f["fields/Bx/" + time_cycle])
    nx_local, ny_local, nz = sample.shape

###* Define global domain size
nx_global = XLEN * nx_local
ny_global = YLEN * ny_local

Bx = np.zeros((nx_global, ny_global))
By = np.zeros((nx_global, ny_global))
Bz = np.zeros((nx_global, ny_global))

for rank in range(num_expected_files):

    file_path = os.path.join(dir_data, f"proc{rank}.hdf")
    
    with h5py.File(file_path, "r") as f:

        Bx_data = np.array(f["fields/Bx/" + time_cycle])
        By_data = np.array(f["fields/By/" + time_cycle])
        Bz_data = np.array(f["fields/Bz/" + time_cycle])

        ## Determine processor position in 2D MPI grid
        i = rank // YLEN  # x index
        j = rank % YLEN   # y index

        x0 = i * nx_local
        y0 = j * ny_local

        Bx[x0:x0 + nx_local, y0:y0 + ny_local] = Bx_data[:, :, 0]
        By[x0:x0 + nx_local, y0:y0 + ny_local] = By_data[:, :, 0]
        Bz[x0:x0 + nx_local, y0:y0 + ny_local] = Bz_data[:, :, 0]


###? Plot (2D)
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

fig.tight_layout(rect = [0.0, 0.0, 1.0, 1.0])
plt.savefig(dir_data + "B_field_" + time_cycle + ".png")
plt.savefig(dir_data + "B_field_" + time_cycle + ".eps")
plt.close()

###* =================================================================== *###

print()
print("Complete .....", "Time Elapsed = ", datetime.now() - startTime, "\n\n")