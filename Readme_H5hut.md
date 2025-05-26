# iPIC3D (H5hut)

## Requirements
  - HDF5 (load or download the parallel hdf5 library on the cluster)


## Installation
1. Set variable for H5hut directory
``` shell
export H5HUT_DIR=$HOME/H5hut
```
Set a variable for the H5hut directory (iPIC3D uses this variable). This will create the lib and include files in the ```$HOME/H5hut``` directory. If you do not wish to install H5hut in the home directory, please replace this with the desired directory.

2. Download H5hut
``` shell
git clone https://gitlab.psi.ch/H5hut/src.git H5hut-2.0.0rc3
```

3. Create build directory
``` shell
cd H5hut-2.0.0rc3
```

4. Compile (part 1)
``` shell
./autogen.sh
```

5. Compile (part 2)
``` shell
CC=mpicc CXX=mpicxx ./configure --enable-parallel --enable-large-indices --enable-shared --enable-static --with-hdf5=$EBROOTHDF5 --prefix=$H5HUT_DIR
```

6. Compile (part 3)
``` shell
CC=mpicc CXX=mpicxx make -j
```

7. Compile (part 4)
``` shell
CC=mpicc CXX=mpicxx make install
```