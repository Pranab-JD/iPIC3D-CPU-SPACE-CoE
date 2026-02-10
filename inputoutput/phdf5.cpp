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


#include <mpi.h>
#include "phdf5.h"
#include "ipicdefs.h"
#include "errors.h"
#include "debug.h"
#include "Alloc.h"
#include "MPIdata.h"
#include "Collective.h"
#include <vector>

#ifdef PHDF5

PHDF5fileClass::PHDF5fileClass(string filestr, int nd, const int *coord, MPI_Comm mpicomm)
{
    // SetDefaultGroups();
    filename = filestr;
    ndim = nd;
    comm = mpicomm;
    
    for (int i=0; i<ndim; i++) 
        mpicoord[i] = coord[i];

    // get cartesian dims from the communicator (works if comm is cartesian)
    int periods[3];
    MPI_Cart_get(comm, ndim, mpidims, periods, mpicoord);
}

PHDF5fileClass::PHDF5fileClass(string filestr)
{
    // SetDefaultGroups();
    filename = filestr;
}

// void PHDF5fileClass::SetDefaultGroups(void)
// {
//     grpnames[0] = "Fields";
//     grpnames[1] = "Moments";
//     grpnames[2] = "Parameters";
// }

void PHDF5fileClass::CreatePHDF5file(double *L, int *dglob, int *dlocl, const char* param)
{
    hid_t   acc_t;
    hid_t   Ldataspace;
    hid_t   Ldataset;
    hid_t   ndataspace;
    hid_t   ndataset;
    herr_t  status;
    hsize_t d[1];

    for (int i=0; i<ndim; i++)
    {
        LxLyLz[i] = L[i];
        dim[i]    = (hsize_t)dglob[i];
        chdim[i]  = (hsize_t)dlocl[i];
    }

    acc_t = H5Pcreate(H5P_FILE_ACCESS);
    const char* force_sec2 = getenv("IPIC_FORCE_HDF5_SEC2");

    if (force_sec2 && atoi(force_sec2) == 1)
        H5Pset_fapl_sec2(acc_t);
    else
    {
        #ifdef USING_PARALLEL_HDF5

            herr_t perr = H5Pset_fapl_mpio(acc_t, comm, MPI_INFO_NULL);
            if (perr < 0) 
            {
                if (MPIdata::get_rank() == 0) std::cerr << "H5Pset_fapl_mpio failed\n";
                MPI_Abort(comm, 911);
            }

            herr_t lerr = H5Pset_file_locking(acc_t, 0 /*use_locking*/, 1 /*ignore_when_disabled*/);
            if (lerr < 0) 
            {
                if (MPIdata::get_rank() == 0) std::cerr << "H5Pset_file_locking failed\n";
                MPI_Abort(comm, 912);
            }

        #endif
    }

    file_id = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, acc_t);
    H5Pclose(acc_t);

    hid_t gid = H5Gcreate2(file_id, param, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (gid >= 0) H5Gclose(gid);
}

int PHDF5fileClass::CreatePHDF5fileParticles(const std::string& root_group)
{
    // Create file access property list for MPI-IO
    hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
    
    #ifdef USING_PARALLEL_HDF5
        H5Pset_fapl_mpio(fapl, comm, MPI_INFO_NULL);
    #endif

    // Create file collectively
    file_id = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
    H5Pclose(fapl);

    if (file_id < 0) return 1;

    // Normalize group path (ensure leading '/')
    std::string g = root_group;
    if (g.empty()) g = "/particles";
    if (g[0] != '/') g = "/" + g;

    // mkdir -p for the group path
    {
        H5E_auto2_t old_func = nullptr;
        void* old_client_data = nullptr;
        H5Eget_auto2(H5E_DEFAULT, &old_func, &old_client_data);
        H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr);

        size_t pos = 0;
        while (true) 
        {
            pos = g.find('/', pos + 1);
            std::string cur = (pos == std::string::npos) ? g : g.substr(0, pos);

            hid_t gid = H5Gopen2(file_id, cur.c_str(), H5P_DEFAULT);
            if (gid >= 0) 
                H5Gclose(gid);
            else 
            {
                gid = H5Gcreate2(file_id, cur.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
                if (gid >= 0) H5Gclose(gid);
            }

            if (pos == std::string::npos) break;
        }

        H5Eset_auto2(H5E_DEFAULT, old_func, old_client_data);
    }

    return 0;
}

void PHDF5fileClass::ClosePHDF5file()
{
    H5Fclose(file_id);
}

int PHDF5fileClass::WritePHDF5dataset_particles_f32(const std::string& grp, const std::string& name, const float* buffer,
                                                    const hsize_t gdim[2], const hsize_t start[2], const hsize_t count[2])
{
    const bool empty = (count[0] == 0 || count[1] == 0);

    hid_t filespace = H5Screate_simple(2, gdim, NULL);
    hid_t memspace  = empty ? H5Screate(H5S_NULL) : H5Screate_simple(2, count, NULL);

    // --- normalize grp (strip leading '/', strip trailing '/') ---
    std::string grp_norm = grp;
    if (!grp_norm.empty() && grp_norm[0] == '/') grp_norm.erase(0, 1);
    while (!grp_norm.empty() && grp_norm.back() == '/') grp_norm.pop_back();

    std::string dname = "/" + grp_norm + "/" + name;

    // --- ensure all intermediate groups exist (mkdir -p) INCLUDING leaf group ---
    {
        H5E_auto2_t old_func = nullptr;
        void*       old_client_data = nullptr;
        H5Eget_auto2(H5E_DEFAULT, &old_func, &old_client_data);
        H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr);

        std::string gpath = "/" + grp_norm;

        size_t pos = 0;
        while (true) 
        {
            pos = gpath.find('/', pos + 1);
            std::string cur = (pos == std::string::npos) ? gpath : gpath.substr(0, pos);

            hid_t gid = H5Gopen2(file_id, cur.c_str(), H5P_DEFAULT);
            if (gid >= 0)
                H5Gclose(gid);
            else 
            {
                gid = H5Gcreate2(file_id, cur.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
                if (gid >= 0) H5Gclose(gid);
            }

            if (pos == std::string::npos) break;
        }

        H5Eset_auto2(H5E_DEFAULT, old_func, old_client_data);
    }

    #ifdef USING_PARALLEL_HDF5
        MPI_Barrier(comm);
    #endif

    // --- create collectively ---
    hid_t dset = H5Dcreate2(file_id, dname.c_str(), H5T_NATIVE_FLOAT, filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    if (dset < 0) 
    {
        if (memspace  >= 0) H5Sclose(memspace);
        if (filespace >= 0) H5Sclose(filespace);
        return 1;
    }

    // --- select hyperslab (or none for empty ranks) ---
    if (empty) 
    {
        H5Sselect_none(filespace);
        H5Sselect_none(memspace);
    } 
    else 
    {
        if (H5Sselect_hyperslab(filespace, H5S_SELECT_SET, start, NULL, count, NULL) < 0) 
        {
            H5Dclose(dset);
            H5Sclose(memspace);
            H5Sclose(filespace);
            return 1;
        }
    }

    hid_t xfer = H5Pcreate(H5P_DATASET_XFER);
    #ifdef USING_PARALLEL_HDF5
        if (using_mpio_vfd)
            H5Pset_dxpl_mpio(xfer, H5FD_MPIO_COLLECTIVE);
    #endif

    herr_t err = H5Dwrite(dset, H5T_NATIVE_FLOAT, memspace, filespace, xfer, empty ? nullptr : buffer);

    H5Pclose(xfer);
    H5Dclose(dset);
    H5Sclose(memspace);
    H5Sclose(filespace);

    return (err < 0) ? 1 : 0;
}

int PHDF5fileClass::WritePHDF5dataset_particles_f64(const std::string& grp, const std::string& name, const double* buffer,
                                                    const hsize_t gdim[2], const hsize_t start[2], const hsize_t count[2])
{
    const bool empty = (count[0] == 0 || count[1] == 0);

    hid_t filespace = H5Screate_simple(2, gdim, NULL);
    hid_t memspace  = empty ? H5Screate(H5S_NULL) : H5Screate_simple(2, count, NULL);

    std::string grp_norm = grp;
    if (!grp_norm.empty() && grp_norm[0] == '/') grp_norm.erase(0, 1);
    while (!grp_norm.empty() && grp_norm.back() == '/') grp_norm.pop_back();

    std::string dname = "/" + grp_norm + "/" + name;

    {
        H5E_auto2_t old_func = nullptr;
        void*       old_client_data = nullptr;
        H5Eget_auto2(H5E_DEFAULT, &old_func, &old_client_data);
        H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr);

        std::string gpath = "/" + grp_norm;

        size_t pos = 0;
        while (true) 
        {
            pos = gpath.find('/', pos + 1);
            std::string cur = (pos == std::string::npos) ? gpath : gpath.substr(0, pos);

            hid_t gid = H5Gopen2(file_id, cur.c_str(), H5P_DEFAULT);
            if (gid >= 0)
                H5Gclose(gid);
            else 
            {
                gid = H5Gcreate2(file_id, cur.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
                if (gid >= 0) H5Gclose(gid);
            }

            if (pos == std::string::npos) break;
        }

        H5Eset_auto2(H5E_DEFAULT, old_func, old_client_data);
    }

    #ifdef USING_PARALLEL_HDF5
        MPI_Barrier(comm);
    #endif

    hid_t dset = H5Dcreate2(file_id, dname.c_str(), H5T_NATIVE_DOUBLE, filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    if (dset < 0) 
    {
        if (memspace  >= 0) H5Sclose(memspace);
        if (filespace >= 0) H5Sclose(filespace);
        return 1;
    }

    if (empty) 
    {
        H5Sselect_none(filespace);
        H5Sselect_none(memspace);
    } 
    else 
    {
        if (H5Sselect_hyperslab(filespace, H5S_SELECT_SET, start, NULL, count, NULL) < 0) 
        {
            H5Dclose(dset);
            H5Sclose(memspace);
            H5Sclose(filespace);
            return 1;
        }
    }

    hid_t xfer = H5Pcreate(H5P_DATASET_XFER);
    #ifdef USING_PARALLEL_HDF5
        if (using_mpio_vfd)
            H5Pset_dxpl_mpio(xfer, H5FD_MPIO_COLLECTIVE);
    #endif

    herr_t err = H5Dwrite(dset, H5T_NATIVE_DOUBLE, memspace, filespace, xfer, empty ? nullptr : buffer);

    H5Pclose(xfer);
    H5Dclose(dset);
    H5Sclose(memspace);
    H5Sclose(filespace);

    return (err < 0) ? 1 : 0;
}

int PHDF5fileClass::WritePHDF5dataset_nodes_f32(const std::string& grp, const std::string& name, const float* buffer,
                                                const hsize_t gdim[3], const hsize_t start[3], const hsize_t count[3])
{
    const bool empty = (count[0] == 0 || count[1] == 0 || count[2] == 0);

    hid_t filespace = H5Screate_simple(3, gdim, NULL);
    hid_t memspace  = empty ? H5Screate(H5S_NULL) : H5Screate_simple(3, count, NULL);

    // --- normalize grp (strip leading '/', strip trailing '/') ---
    std::string grp_norm = grp;
    if (!grp_norm.empty() && grp_norm[0] == '/') grp_norm.erase(0, 1);
    while (!grp_norm.empty() && grp_norm.back() == '/') grp_norm.pop_back();

    std::string dname = "/" + grp_norm + "/" + name;

    // --- ensure all intermediate groups exist (mkdir -p) INCLUDING leaf group ---
    {
        H5E_auto2_t old_func = nullptr;
        void*       old_client_data = nullptr;
        H5Eget_auto2(H5E_DEFAULT, &old_func, &old_client_data);
        H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr);

        // Build "/Moments/species_0" etc.
        std::string gpath = "/" + grp_norm;

        // Create each prefix group: "/Moments", "/Moments/species_0"
        size_t pos = 0;
        while (true) {
            pos = gpath.find('/', pos + 1);
            std::string cur = (pos == std::string::npos) ? gpath : gpath.substr(0, pos);

            hid_t gid = H5Gopen2(file_id, cur.c_str(), H5P_DEFAULT);
            if (gid >= 0) {
                H5Gclose(gid);
            } else {
                gid = H5Gcreate2(file_id, cur.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
                if (gid >= 0) H5Gclose(gid);
            }

            if (pos == std::string::npos) break;
        }

        H5Eset_auto2(H5E_DEFAULT, old_func, old_client_data);
    }

    #ifdef USING_PARALLEL_HDF5
        MPI_Barrier(comm);
    #endif

    // --- dataset does not exist: create collectively on ALL ranks ---
    hid_t dset = H5Dcreate2(file_id, dname.c_str(), H5T_NATIVE_FLOAT, filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    
    if (dset < 0) 
    {
        if (memspace  >= 0) H5Sclose(memspace);
        if (filespace >= 0) H5Sclose(filespace);
        return 1;
    }

    // --- select hyperslab (or none for empty ranks) ---
    if (empty) 
    {
        H5Sselect_none(filespace);
        H5Sselect_none(memspace);
    } 
    else 
    {
        if (H5Sselect_hyperslab(filespace, H5S_SELECT_SET, start, NULL, count, NULL) < 0) 
        {
            H5Dclose(dset);
            H5Sclose(memspace);
            H5Sclose(filespace);
            return 1;
        }
    }

    hid_t xfer = H5Pcreate(H5P_DATASET_XFER);
    
    #ifdef USING_PARALLEL_HDF5
        if (using_mpio_vfd)
            H5Pset_dxpl_mpio(xfer, H5FD_MPIO_COLLECTIVE);
    #endif

    herr_t err = H5Dwrite(dset, H5T_NATIVE_FLOAT, memspace, filespace, xfer, empty ? nullptr : buffer);

    H5Pclose(xfer);
    H5Dclose(dset);
    H5Sclose(memspace);
    H5Sclose(filespace);

    return (err < 0) ? 1 : 0;
}

int PHDF5fileClass::WritePHDF5dataset_nodes_f64(const std::string& grp, const std::string& name, const double* buffer,
                                                const hsize_t gdim[3], const hsize_t start[3], const hsize_t count[3])
{
    const bool empty = (count[0] == 0 || count[1] == 0 || count[2] == 0);

    hid_t filespace = H5Screate_simple(3, gdim, NULL);
    hid_t memspace  = empty ? H5Screate(H5S_NULL) : H5Screate_simple(3, count, NULL);

    if (filespace < 0 || memspace < 0) {
        if (memspace  >= 0) H5Sclose(memspace);
        if (filespace >= 0) H5Sclose(filespace);
        return 1;
    }

    // --- normalize grp (strip leading '/', strip trailing '/') ---
    std::string grp_norm = grp;
    if (!grp_norm.empty() && grp_norm[0] == '/') grp_norm.erase(0, 1);
    while (!grp_norm.empty() && grp_norm.back() == '/') grp_norm.pop_back();

    std::string dname = "/" + grp_norm + "/" + name;

    // --- ensure all intermediate groups exist (mkdir -p) INCLUDING leaf group ---
    {
        H5E_auto2_t old_func = nullptr;
        void*       old_client_data = nullptr;
        H5Eget_auto2(H5E_DEFAULT, &old_func, &old_client_data);
        H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr);

        std::string gpath = "/" + grp_norm;

        // Create each prefix group: "/Moments", "/Moments/species_0", ...
        size_t pos = 0;
        while (true) {
            pos = gpath.find('/', pos + 1);
            std::string cur = (pos == std::string::npos) ? gpath : gpath.substr(0, pos);

            hid_t gid = H5Gopen2(file_id, cur.c_str(), H5P_DEFAULT);
            if (gid >= 0) {
                H5Gclose(gid);
            } else {
                gid = H5Gcreate2(file_id, cur.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
                if (gid >= 0) H5Gclose(gid);
            }

            if (pos == std::string::npos) break;
        }

        H5Eset_auto2(H5E_DEFAULT, old_func, old_client_data);
    }

    #ifdef USING_PARALLEL_HDF5
        MPI_Barrier(comm);
    #endif

    // --- dataset does not exist: create collectively on ALL ranks ---
    hid_t dset = H5Dcreate2(file_id, dname.c_str(), H5T_NATIVE_DOUBLE,
                            filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    if (dset < 0) {
        H5Sclose(memspace);
        H5Sclose(filespace);
        return 1;
    }

    // --- select hyperslab (or none for empty ranks) ---
    if (empty) {
        H5Sselect_none(filespace);
        H5Sselect_none(memspace);
    } else {
        if (H5Sselect_hyperslab(filespace, H5S_SELECT_SET, start, NULL, count, NULL) < 0) {
            H5Dclose(dset);
            H5Sclose(memspace);
            H5Sclose(filespace);
            return 1;
        }
    }

    hid_t xfer = H5Pcreate(H5P_DATASET_XFER);
    if (xfer < 0) {
        H5Dclose(dset);
        H5Sclose(memspace);
        H5Sclose(filespace);
        return 1;
    }

    #ifdef USING_PARALLEL_HDF5
        if (using_mpio_vfd)
            H5Pset_dxpl_mpio(xfer, H5FD_MPIO_COLLECTIVE);
    #endif

    herr_t err = H5Dwrite(dset, H5T_NATIVE_DOUBLE, memspace, filespace, xfer,
                          empty ? nullptr : buffer);

    H5Pclose(xfer);
    H5Dclose(dset);
    H5Sclose(memspace);
    H5Sclose(filespace);

    return (err < 0) ? 1 : 0;
}

int PHDF5fileClass::getPHDF5ncx()
{
    return (int)dim[0];
}

int PHDF5fileClass::getPHDF5ncy()
{
    if (ndim<2) return 1;
    return (int)dim[1];
}

int PHDF5fileClass::getPHDF5ncz()
{
    if (ndim<3) return 1;
    return (int)dim[2];
}

int PHDF5fileClass::getPHDF5ndim()
{
    return ndim;
}

#endif
