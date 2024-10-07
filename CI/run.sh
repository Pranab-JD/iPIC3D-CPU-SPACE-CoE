#!/bin/bash


ml CMake/3.27.6-GCCcore-13.2.0 OpenMPI/4.1.6-GCC-13.2.0 HDF5/1.14.3-gompi-2023b
ml

echo "========================================================================"
echo " "
date
echo " "

cd ./build/

rm -r CI_Maxwell2D
mkdir -p CI_Maxwell2D

mpirun -np 25 ./iPIC3D ../CI_Maxwell2D.inp

echo " "
echo "========================================================================"
date
echo " "