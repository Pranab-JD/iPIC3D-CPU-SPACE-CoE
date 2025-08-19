# iPIC3D

## Requirements
  - gcc/g++ compiler
  - cmake (minimum version 2.8)
  - MPI (OpenMPI or MPICH)
  - HDF5 (optional)
  - VTK (optional)
  - Paraview/Catalyst (optional)

## Installation
1. Download the code
``` shell
git clone https://github.com/Pranab-JD/iPIC3D-CPU-SPACE-CoE.git
```

2. Create build directory
``` shell
cd iPIC3D-CPU-SPACE-CoE && mkdir build && cd build
```

3. Checkout to the ```ecsim_relsim``` branch
``` shell
git checkout ecsim_relsim
```

4. Compile the code (part 1)
``` shell
cmake ..
```

5. Compile the code (part 2)
``` shell
make -j     # -j = build with max # of threads - fast, recommended
```

iPIC3D should now be installed! The executable is called 'iPIC3D'.

To run the code, please try
``` shell
# no_of_proc = XLEN x YLEN x ZLEN (specify in the input file)
mpirun -np no_of_proc ./iPIC3D  inputfilename.inp           # try 'srun', if mpirun does not work
```

# Acknowledgements and Citations
This version of iPIC3D (with the implicit moment method) has been developed by Prof Stefano Markidis and his team. The energy conserving semi-implicit method (ECSIM) and relativistic semi-implicit method (RelSIM) have been implemented by Dr Pranab J Deka and Prof Fabio Bacchini.

If you use this iPIC3D code, please cite: <br />
Stefano Markidis, Giovanni Lapenta, and Rizwan-uddin (2010), *Multi-scale simulations of plasma with iPIC3D*, Mathematics and Computers in Simulation, 80, 7, 1509-1519 [[DOI]](https://doi.org/10.1016/j.matcom.2009.08.038)

If you use the ECSIM algorithm (within iPIC3D), please cite (in addition to the aforementioned article): <br />
Giovanni Lapenta (2017), *Exactly energy conserving semi-implicit particle in cell formulation*, Journal of Computational Physics, 334, 349 
[[DOI]](http://dx.doi.org/10.1016/j.jcp.2017.01.002)

If you use the RelSIM algorithm (within iPIC3D), please cite (in addition to the aforementioned articles): <br />
Fabio Bacchini (2023), *RelSIM: A Relativistic Semi-implicit Method for Particle-in-cell Simulations*, The Astrophysical Journal Supplement Series, 268:60 [[DOI]](https://doi.org/10.3847/1538-4365/acefba)