#!/bin/bash

ml CMake/3.27.6-GCCcore-13.2.0 OpenMPI/4.1.6-GCC-13.2.0 HDF5/1.14.3-gompi-2023b
ml

echo "========================================================================"
echo " "
date
echo " "

mkdir -p ./build
cd ./build
cmake ../.. -DUSE_HDF5=ON
make

echo " "
echo "========================================================================"
date
echo " "
