/* iPIC3D was originally developed by Stefano Markidis and Giovanni Lapenta. 
 * This release was contributed by Alec Johnson and Ivy Bo Peng.
 * Publications that use results from iPIC3D need to properly cite  
 * 'S. Markidis, G. Lapenta, and Rizwan-uddin. "Multi-scale simulations of 
 * plasma with iPIC3D." Mathematics and Computers in Simulation 80.7 (2010): 1509-1519.'
 *
 *        Copyright 2015 KTH Royal Institute of Technology
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at 
 *
 *         http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


#ifndef __PHDF5_H__ 
#define __PHDF5_H__

#ifndef NO_HDF5
    #include "ipichdf5.h"
#endif

#include "mpi.h"
#include "arraysfwd.h"
#include <string>
using namespace std;

class PHDF5fileClass
{
    #ifndef NO_HDF5
    public:

        PHDF5fileClass(string filestr, int nd, const int *coord, MPI_Comm mpicomm);
        PHDF5fileClass(string filestr);

        //* Functions to create files and write data
        void CreatePHDF5file(double *L, int *dglob, int *dlocl, const char* param);
        int CreatePHDF5fileParticles(const std::string& root_group);
        void ClosePHDF5file();

         //* Functions to open files and read data
        void OpenPHDF5file();
        void ReadPHDF5param();

        //* Functions to write field and moment data
        int WritePHDF5dataset_nodes_f32(const std::string& grp, const std::string& name, const float* buffer,
                                        const hsize_t gdim[3], const hsize_t start[3], const hsize_t count[3]);
        int WritePHDF5dataset_nodes_f64(const std::string& grp, const std::string& name, const double* buffer,
                                        const hsize_t gdim[3], const hsize_t start[3], const hsize_t count[3]);

        //* Functions to write particle data
        int WritePHDF5dataset_particles_f32(const std::string& grp, const std::string& name, const float* buffer,
                                            const hsize_t gdim[2], const hsize_t start[2], const hsize_t count[2]);
        int WritePHDF5dataset_particles_f64(const std::string& grp, const std::string& name, const double* buffer,
                                            const hsize_t gdim[2], const hsize_t start[2], const hsize_t count[2]);            

        int  getPHDF5ndim();
        int  getPHDF5ncx();
        int  getPHDF5ncy();
        int  getPHDF5ncz();


    private:

        /* The group names are fixed and must be initialized in the constructor */
        string grpnames[3];
        static const int ngrp = 3;

        /* Private variables */
        MPI_Comm comm;

        int      ndim;
        hid_t    file_id;
        double   LxLyLz  [3];  // Using dynamic allocation of these vectors caused segfault and mpi problems
        int      mpicoord[3];  // as the vectors were corrupted at the second call of the Write function.
        hsize_t  dim     [3];  // Is there a way to avoid this problem?
        hsize_t  chdim   [3];  //
        int mpidims[3] = {1,1,1};

        string   filename;
        bool     bparticles;
        MPI_Comm comm_;
        bool using_mpio_vfd = false;
    #endif
};

#endif
