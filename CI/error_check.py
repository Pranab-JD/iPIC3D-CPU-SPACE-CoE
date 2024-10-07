"""
Created on Tue May 7 12:50 2024

@author: Pranab JD
"""

import os
import glob, h5py
import numpy as np

from datetime import datetime

startTime = datetime.now()

###* =================================================================== *###

dir_ref = "./build/CI_Maxwell2D/"
dir_data = "./CI_Maxwell2D_ref/"
time_cycle = "cycle_4"

print("Comparing error at ", time_cycle, " time step.\n")

num_hdf_files = len([name for name in os.listdir(dir_ref) if os.path.isfile(os.path.join(dir_data, name))])

Jx_0_error = np.zeros(num_hdf_files); Jy_0_error = np.zeros(num_hdf_files); 
Jz_0_error = np.zeros(num_hdf_files); rho_0_error = np.zeros(num_hdf_files)
Jx_1_error = np.zeros(num_hdf_files); Jy_1_error = np.zeros(num_hdf_files); 
Jz_1_error = np.zeros(num_hdf_files); rho_1_error = np.zeros(num_hdf_files)

Ex_error = np.zeros(num_hdf_files); Ey_error = np.zeros(num_hdf_files); Ez_error = np.zeros(num_hdf_files)
Bx_error = np.zeros(num_hdf_files); By_error = np.zeros(num_hdf_files); Bz_error = np.zeros(num_hdf_files)

###? Iterate through all hdf files (one for each processor)
for ii in range(num_hdf_files):

    ###? Reference dataset
    for file in glob.glob(dir_ref + "/proc" + str(ii) + ".hdf"):

        data = h5py.File(file, "r")
        moments = data.get("moments")
        field = data.get("fields")

        ###? Moments (Jx, Jy, Jz, rho)
        sp_0_Jx = moments.get("species_0/Jx"); sp_0_Jy = moments.get("species_0/Jy"); 
        sp_0_Jz = moments.get("species_0/Jz"); sp_0_rho = moments.get("species_0/rho")

        sp_1_Jx = moments.get("species_1/Jx"); sp_1_Jy = moments.get("species_1/Jy"); 
        sp_1_Jz = moments.get("species_1/Jz"); sp_1_rho = moments.get("species_1/rho")

        sp_0_Jx_ref = np.array(sp_0_Jx.get(time_cycle)); sp_0_Jy_ref = np.array(sp_0_Jy.get(time_cycle));
        sp_0_Jz_ref = np.array(sp_0_Jz.get(time_cycle)); sp_0_rho_ref = np.array(sp_0_rho.get(time_cycle));

        sp_1_Jx_ref = np.array(sp_1_Jx.get(time_cycle)); sp_1_Jy_ref = np.array(sp_1_Jy.get(time_cycle));
        sp_1_Jz_ref = np.array(sp_1_Jz.get(time_cycle)); sp_1_rho_ref = np.array(sp_1_rho.get(time_cycle));
        
        ###? Fields (Ex, Ey, Ez, Bx, By, Bz)
        Bx = field.get("Bx"); By = field.get("By"); Bz = field.get("Bz")
        Ex = field.get("Ex"); Ey = field.get("Ey"); Ez = field.get("Ez")

        Bx_ref = np.array(Bx.get(time_cycle)); By_ref = np.array(By.get(time_cycle)); Bz_ref = np.array(Bz.get(time_cycle))
        Ex_ref = np.array(Ex.get(time_cycle)); Ey_ref = np.array(Ey.get(time_cycle)); Ez_ref = np.array(Ez.get(time_cycle))

    ###? Modified dataset
    for file in glob.glob(dir_data + "/proc" + str(ii) + ".hdf"):

        data = h5py.File(file, "r")
        moments = data.get("moments")
        field = data.get("fields")
        
        ###? Moments (Jx, Jy, Jz, rho)
        sp_0_Jx = moments.get("species_0/Jx"); sp_0_Jy = moments.get("species_0/Jy"); 
        sp_0_Jz = moments.get("species_0/Jz"); sp_0_rho = moments.get("species_0/rho")

        sp_1_Jx = moments.get("species_1/Jx"); sp_1_Jy = moments.get("species_1/Jy"); 
        sp_1_Jz = moments.get("species_1/Jz"); sp_1_rho = moments.get("species_1/rho")

        sp_0_Jx_data = np.array(sp_0_Jx.get(time_cycle)); sp_0_Jy_data = np.array(sp_0_Jy.get(time_cycle));
        sp_0_Jz_data = np.array(sp_0_Jz.get(time_cycle)); sp_0_rho_data = np.array(sp_0_rho.get(time_cycle));

        sp_1_Jx_data = np.array(sp_1_Jx.get(time_cycle)); sp_1_Jy_data = np.array(sp_1_Jy.get(time_cycle));
        sp_1_Jz_data = np.array(sp_1_Jz.get(time_cycle)); sp_1_rho_data = np.array(sp_1_rho.get(time_cycle));
        
        ###? Fields (Ex, Ey, Ez, Bx, By, Bz)
        Bx1 = field.get("Bx"); By = field.get("By"); Bz = field.get("Bz")
        Ex = field.get("Ex"); Ey = field.get("Ey"); Ez = field.get("Ez")

        Bx_data = np.array(Bx1.get(time_cycle)); By_data = np.array(By.get(time_cycle)); Bz_data = np.array(Bz.get(time_cycle))
        Ex_data = np.array(Ex.get(time_cycle)); Ey_data = np.array(Ey.get(time_cycle)); Ez_data = np.array(Ez.get(time_cycle))

    ###? Compute error
    Jx_0_diff = np.mean(abs(sp_0_Jx_ref - sp_0_Jx_data))/np.linalg.norm(sp_0_Jx_ref)
    Jy_0_diff = np.mean(abs(sp_0_Jy_ref - sp_0_Jy_data))/np.linalg.norm(sp_0_Jy_ref)
    Jz_0_diff = np.mean(abs(sp_0_Jz_ref - sp_0_Jz_data))/np.linalg.norm(sp_0_Jy_ref)
    rho_0_diff = np.mean(abs(sp_0_rho_ref - sp_0_rho_data))/np.linalg.norm(sp_0_rho_ref)

    Jx_1_diff = np.mean(abs(sp_1_Jx_ref - sp_1_Jx_data))/np.linalg.norm(sp_1_Jx_ref)
    Jy_1_diff = np.mean(abs(sp_1_Jy_ref - sp_1_Jy_data))/np.linalg.norm(sp_1_Jy_ref)
    Jz_1_diff = np.mean(abs(sp_1_Jz_ref - sp_1_Jz_data))/np.linalg.norm(sp_1_Jy_ref)
    rho_1_diff = np.mean(abs(sp_1_rho_ref - sp_1_rho_data))/np.linalg.norm(sp_1_rho_ref)

    Bx_diff = np.mean(abs(Bx_ref - Bx_data))/np.linalg.norm(Bx_ref)
    By_diff = np.mean(abs(By_ref - By_data))/np.linalg.norm(By_ref)
    Bz_diff = np.mean(abs(Bz_ref - Bz_data))/np.linalg.norm(Bz_ref)

    Ex_diff = np.mean(abs(Ex_ref - Ex_data))/np.linalg.norm(Ex_ref)
    Ey_diff = np.mean(abs(Ey_ref - Ey_data))/np.linalg.norm(Ey_ref)
    Ez_diff = np.mean(abs(Ez_ref - Ez_data))/np.linalg.norm(Ez_ref)

    Jx_0_error[ii] = Jx_0_diff; Jy_0_error[ii] = Jy_0_diff; Jz_0_error[ii] = Jz_0_diff; rho_0_error[ii] = rho_0_diff;
    Jx_1_error[ii] = Jx_1_diff; Jy_1_error[ii] = Jy_1_diff; Jz_1_error[ii] = Jz_1_diff; rho_1_error[ii] = rho_1_diff;

    Bx_error[ii] = Bx_diff; By_error[ii] = By_diff; Bz_error[ii] = Bz_diff
    Ex_error[ii] = Ex_diff; Ey_error[ii] = Ey_diff; Ez_error[ii] = Ez_diff


###* =================================================================== *###

print("Error in J(x, y, z) for species 0: ", np.mean(Jx_0_error), ", ", np.mean(Jy_0_error), ", ", np.mean(Jz_0_error))
print("Error in J(x, y, z) for species 1: ", np.mean(Jx_1_error), ", ", np.mean(Jy_1_error), ", ", np.mean(Jz_1_error))
print()
print("Error in density for species 0 & 1: ", np.mean(rho_0_error), ", ", np.mean(rho_1_error))
print()
print("Error in B(x, y, z): ", np.mean(Bx_error), ", ", np.mean(By_error), ", ", np.mean(Bz_error))
print("Error in E(x, y, z): ", np.mean(Ex_error), ", ", np.mean(Ey_error), ", ", np.mean(Ez_error))


print()
print("Complete .....", "Time Elapsed = ", datetime.now() - startTime)