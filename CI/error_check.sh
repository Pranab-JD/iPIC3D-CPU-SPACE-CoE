#!/bin/bash

ml Python/3.11.3-GCCcore-12.3.0 

module list
echo "========================================================================"
echo " "
date
echo " "

cd ./build/
python -m venv venv
. venv/bin/activate
pip install h5py
python ./error_check.py

echo " "
echo "========================================================================"
date
echo " "