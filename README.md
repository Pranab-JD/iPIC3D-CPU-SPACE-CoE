# iPIC3D

## Requirements
  - gcc/g++ compiler
  - cmake (minimum version 2.8)
  - MPI (OpenMPI or MPICH)
  - HDF5 (optional)
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

3. Compile the code
``` shell
cmake ..
make -j     # -j = build with max # of threads - fast, recommended
```

4. Run
``` shell
# no_of_proc = XLEN x YLEN x ZLEN (as specified in the input file)
mpirun -np no_of_proc ./iPIC3D  inputfilename.inp
```

**Important:** make sure `number of MPI process = XLEN x YLEN x ZLEN` as specified in the input file.

On a supercomputer or a cluster (especially a multinode system), you should use `srun` to launch iPIC3D. 

# Acknowledgements and Citations
This version of iPIC3D (with the implicit moment method (IMM)) has been developed by Prof Stefano Markidis and his team. The energy conserving semi-implicit method (ECSIM) and Relativistic semi-implicit method (RelSIM) has been implemented by Dr Pranab J Deka and Prof Fabio Bacchini.

If you use this iPIC3D code, please cite
Stefano Markidis, Giovanni Lapenta, and Rizwan-uddin (2010), *Multi-scale simulations of plasma with iPIC3D*, Mathematics and Computers in Simulation, 80, 7, 1509-1519 [[DOI]](https://doi.org/10.1016/j.matcom.2009.08.038)

If you use the ECSIM algorithm (within iPIC3D), please cite
Giovanni Lapenta (2017), *Exactly energy conserving semi-implicit particle in cell formulation*, Journal of Computational Physics, 334 (2017) 349–366 [[DOI]](http://dx.doi.org/10.1016/j.jcp.2017.01.002)

If you use the RelSIM algorithm (within iPIC3D), please cite
Fabio Bacchini (2023), *RelSIM: A Relativistic Semi-implicit Method for Particle-in-cell Simulations*, The Astrophysical Journal Supplement Series, 268:60 [[DOI]](https://doi.org/10.3847/1538-4365/acefba)