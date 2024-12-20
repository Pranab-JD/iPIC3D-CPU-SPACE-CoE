# iPIC3D

## Requirements
  - gcc/g++ compiler
  - cmake (minimum version 2.8)
  - MPI (OpenMPI or MPICH)
  - HDF5 (optional)
  - Paraview/Catalyst (optional)

If you are on a supercomputer or cluster, it's highly possible that you can use tools like `module` to change the compiler, MPI or libraries used.

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

If you are on a super-computer, especially a multi-node system, it's likely that you should use `srun` to launch the program. 

# Citation
Markidis, Stefano and Giovanni Lapenta (2010), *Multi-scale simulations of plasma with iPIC3D*, Mathematics and Computers in Simulation, 80, 7, 1509-1519 [[DOI]](https://doi.org/10.1016/j.matcom.2009.08.038)