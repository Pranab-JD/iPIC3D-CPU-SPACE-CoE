# iPIC3D (H5hut)

## Requirements
  - HDF5 (load or download the parallel hdf5 library on the cluster)


## Installation
1. Set compiler environment
``` shell
### Cray environment (e.g., LUMI)
export CC=cc
export CXX=cxx

### Non-Cray environment
export CC=mpicc
export CXX=mpicxx
```

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
./configure --enable-parallel --enable-large-indices --enable-shared --enable-static --with-hdf5=$EBROOTHDF5 --prefix=$HOME/H5hut
```
This will create the lib and include files in the ```$HOME/H5hut``` directory. If you are not installing H5hut in the home directory, please replace ```--prefix=$HOME/H5hut``` with the correct directory. 

6. Compile (part 3)
``` shell
make -j
```

7. Compile (part 4)
``` shell
make install
```