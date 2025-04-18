"""
Created on Wed Apr 16 20:29:15 2025

@author: Pranab JD
"""

import os
import glob, h5py
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime

startTime = datetime.now()

###* =================================================================== *###

dir_data = "./data_double_H/"
time_cycle = "cycle_22"

print("Plotting data at", time_cycle, "\n")

num_hdf_files = len([name for name in os.listdir(dir_data) if os.path.isfile(os.path.join(dir_data, name))])

Ex = np.zeros(num_hdf_files); Ey = np.zeros(num_hdf_files); Ez = np.zeros(num_hdf_files)
Bx = np.zeros(num_hdf_files); By = np.zeros(num_hdf_files); Bz = np.zeros(num_hdf_files)

###? Iterate through all hdf files (one for each processor)
for ii in range(num_hdf_files):

    for file in glob.glob(dir_data + "/proc" + str(ii) + ".hdf"):

        data = h5py.File(file, "r")
        field = data.get("fields")

        ###? Get the fields (Ex, Ey, Ez, Bx, By, Bz)
        Bx_file = field.get("Bx"); By_file = field.get("By"); Bz_file = field.get("Bz")
        Ex_file = field.get("Ex"); Ey_file = field.get("Ey"); Ez_file = field.get("Ez")

        ###? Convert to numpy arrays
        Bx_data = np.array(Bx_file.get(time_cycle)); By_data = np.array(By_file.get(time_cycle)); Bz_data = np.array(Bz_file.get(time_cycle))
        Ex_data = np.array(Ex_file.get(time_cycle)); Ey_data = np.array(Ey_file.get(time_cycle)); Ez_data = np.array(Ez_file.get(time_cycle))


        fig = plt.figure(figsize = (12, 5), dpi = 150)
        
        plt.subplot(2, 3, 1)
        plt.imshow(Bx_data[:, :, 0], origin = 'lower', cmap = plt.cm.seismic)
        plt.title("Bx")
        plt.colorbar()

        plt.subplot(2, 3, 2)
        plt.imshow(By_data[:, :, 0], origin = 'lower', cmap = plt.cm.seismic)
        plt.title("By")
        plt.colorbar()

        plt.subplot(2, 3, 3)
        plt.imshow(Bz_data[:, :, 0], origin = 'lower', cmap = plt.cm.seismic)
        plt.title("Bz")
        plt.colorbar()

        plt.subplot(2, 3, 4)
        plt.imshow(Ex_data[:, :, 0], origin = 'lower', cmap = plt.cm.hot)
        plt.title("Ex")
        plt.colorbar()

        plt.subplot(2, 3, 5)
        plt.imshow(Ey_data[:, :, 0], origin = 'lower', cmap = plt.cm.hot)
        plt.title("Ey")
        plt.colorbar()

        plt.subplot(2, 3, 6)
        plt.imshow(Ez_data[:, :, 0], origin = 'lower', cmap = plt.cm.hot)
        plt.title("Ez")
        plt.colorbar()

        fig.tight_layout()
        plt.savefig("Fields_" + time_cycle + ".png")

###* =================================================================== *###

print()
print("Complete .....", "Time Elapsed = ", datetime.now() - startTime)