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
species = "3"

print("Plotting data at", time_cycle, "\n")

num_hdf_files = len([name for name in os.listdir(dir_data) if os.path.isfile(os.path.join(dir_data, name))])

Jx = np.zeros(num_hdf_files); Jy = np.zeros(num_hdf_files); Jz = np.zeros(num_hdf_files)
Bx = np.zeros(num_hdf_files); By = np.zeros(num_hdf_files); Bz = np.zeros(num_hdf_files)

###? Iterate through all hdf files (one for each processor)
for ii in range(num_hdf_files):

    for file in glob.glob(dir_data + "/proc" + str(ii) + ".hdf"):

        data = h5py.File(file, "r")
        moments = data.get("moments")

        ###? Get the moments
        Jx_file = moments.get("Jx"); Jy_file = moments.get("Jy"); Jz_file = moments.get("Jz")     #* Total current density
        Jxs_file = moments.get("species_" + species + "/Jx")                                      #* Current density per species (X)
        Jys_file = moments.get("species_" + species + "/Jy")                                      #* Current density per species (Y)
        Jzs_file = moments.get("species_" + species + "/Jz")                                      #* Current density per species (Z)
        rhos_file = moments.get("species_" + species + "/rho")                                    #* Charge density per species

        ###? Convert to numpy arrays
        Jx_data = np.array(Jx_file.get(time_cycle)); Jxs_data = np.array(Jxs_file.get(time_cycle))
        Jy_data = np.array(Jy_file.get(time_cycle)); Jys_data = np.array(Jys_file.get(time_cycle))
        Jz_data = np.array(Jz_file.get(time_cycle)); Jzs_data = np.array(Jzs_file.get(time_cycle))
        rhos_data = np.array(rhos_file.get(time_cycle))

        ###? Plot current density for given species
        fig = plt.figure(figsize = (6, 5), dpi = 150)
        
        plt.subplot(1, 3, 1)
        plt.imshow(Jxs_data[:, :, 0], origin = 'lower', cmap = plt.cm.seismic)
        plt.title("Jx")
        plt.colorbar()

        plt.subplot(1, 3, 2)
        plt.imshow(Jys_data[:, :, 0], origin = 'lower', cmap = plt.cm.seismic)
        plt.title("Jy")
        plt.colorbar()

        plt.subplot(1, 3, 3)
        plt.imshow(Jzs_data[:, :, 0], origin = 'lower', cmap = plt.cm.seismic)
        plt.title("Jz")
        plt.colorbar()

        fig.tight_layout()
        plt.savefig("J_" + "species_" + species + "_" + time_cycle + ".png")

        ###? Plot current density for given species
        fig = plt.figure(figsize = (8, 4), dpi = 150)
        
        plt.subplot(1, 3, 1)
        plt.imshow(Jx_data[:, :, 0], origin = 'lower', cmap = plt.cm.seismic)
        plt.title("Jx")
        plt.colorbar()

        plt.subplot(1, 3, 2)
        plt.imshow(Jy_data[:, :, 0], origin = 'lower', cmap = plt.cm.seismic)
        plt.title("Jy")
        plt.colorbar()

        plt.subplot(1, 3, 3)
        plt.imshow(Jz_data[:, :, 0], origin = 'lower', cmap = plt.cm.seismic)
        plt.title("Jz")
        plt.colorbar()

        fig.tight_layout()
        plt.savefig("J_" + time_cycle + ".png")

        ###? Plot charge density for given species
        fig = plt.figure(figsize = (8, 6), dpi = 100)
        
        plt.imshow(rhos_data[:, :, 0], origin = 'lower', cmap = plt.cm.jet)
        plt.title("rho")
        plt.colorbar()

        fig.tight_layout()
        plt.savefig("rho_" + "species_" + species + "_" + time_cycle + ".png")

###* =================================================================== *###

print()
print("Complete .....", "Time Elapsed = ", datetime.now() - startTime)