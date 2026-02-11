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
#include <fstream>

#include "phdf5.h"
#include "ParallelIO.h"
#include "MPIdata.h"
#include "TimeTasks.h"
#include "Collective.h"
#include "Grid3DCU.h"
#include "VCtopology3D.h"
#include "Particles3D.h"
#include "EMfields3D.h"
#include "math.h"
#include <algorithm>
#include <iostream>
#include <sstream>
#include <iomanip>
#include "Collective.h"
#include <filesystem>
#include <vector>

bool contains_tag(const std::string& taglist, const std::string& target) 
{
    std::istringstream iss(taglist);
    std::string token;
    while (iss >> token) 
    {
        if (token == target) 
        {
            return true;
        }
    }
    return false;
}

static inline void block_dist_1d(hsize_t G, int P, int c, hsize_t &start, hsize_t &count)
{
    const hsize_t base = G / (hsize_t)P;
    const hsize_t rem  = G % (hsize_t)P;
    count = base + ((hsize_t)c < rem ? 1 : 0);
    start = (hsize_t)c * base + (hsize_t)std::min<int>(c, (int)rem);
}

template <typename GetF, typename T>
static inline void pack_sca_nodes(T* out, int lx, int ly, int lz, GetF getf)
{
    size_t p = 0;
    for (int i = 1; i <= lx; ++i)
        for (int j = 1; j <= ly; ++j)
            for (int k = 1; k <= lz; ++k)
                out[p++] = static_cast<T>(getf(i, j, k));
}

template <typename T>
static inline void pack_xyz_from_aos(T* out, long long nop, const Particles3D& P)
{
    for (long long p = 0; p < nop; ++p)
    {
        out[3*p + 0] = (T)P.getX((int)p);
        out[3*p + 1] = (T)P.getY((int)p);
        out[3*p + 2] = (T)P.getZ((int)p);
    }
}

template <typename T>
static inline void pack_uvw_from_aos(T* out, long long nop, const Particles3D& P)
{
    for (long long p = 0; p < nop; ++p)
    {
        out[3*p + 0] = (T)P.getU((int)p);
        out[3*p + 1] = (T)P.getV((int)p);
        out[3*p + 2] = (T)P.getW((int)p);
    }
}

template <typename T>
static inline void pack_q_from_aos(T* out, long long nop, const Particles3D& P)
{
    out[0] = (T)P.getQ((int)0);
}

//*! Function used to write the FIELDS and MOMENTS using the parallel HDF5 library
void WriteFieldsParallel(Grid3DCU *grid, EMfields3D *EMf, CollectiveIO *col, VCtopology3D *vct, int cycle)
{
#ifdef PHDF5

    if (vct->getCartesian_rank() == 0)
        cout << endl << "Writing FIELD data at cycle " << cycle << endl;

    std::stringstream filenmbr;
    filenmbr << std::setfill('0') << std::setw(5) << cycle;
    const std::string fields_folder = col->getSaveDirName() + "/Fields_" + filenmbr.str() + "/";
    const std::string moments_folder = col->getSaveDirName() + "/Moments_" + filenmbr.str() + "/";

    if (contains_tag(col->getFieldOutputTag(), "E") || contains_tag(col->getFieldOutputTag(), "B"))
    {        
        if (MPIdata::get_rank() == 0)
            std::filesystem::create_directories(fields_folder);

        MPI_Barrier(vct->getFieldComm());
    }

    if (contains_tag(col->getFieldOutputTag(), "rho_s")     || contains_tag(col->getFieldOutputTag(), "J_s")    ||
        contains_tag(col->getFieldOutputTag(), "pressure")  || contains_tag(col->getFieldOutputTag(), "H_flux") ||
        contains_tag(col->getFieldOutputTag(), "E_flux"))
    {
        if (MPIdata::get_rank() == 0)
            std::filesystem::create_directories(moments_folder);

        MPI_Barrier(vct->getFieldComm());
    }

    // Global NODE dims (no ghosts)
    const hsize_t Gc[3] = { (hsize_t)col->getNxc(),
                            (hsize_t)col->getNyc(),
                            (hsize_t)col->getNzc() };

    const hsize_t Gn[3] = { Gc[0] + 1, Gc[1] + 1, Gc[2] + 1 };

    // cell distribution (non-overlapping)
    hsize_t cstart[3], ccount[3];
    block_dist_1d(Gc[0], vct->getXLEN(), vct->getCoordinates(0), cstart[0], ccount[0]);
    block_dist_1d(Gc[1], vct->getYLEN(), vct->getCoordinates(1), cstart[1], ccount[1]);
    block_dist_1d(Gc[2], vct->getZLEN(), vct->getCoordinates(2), cstart[2], ccount[2]);

    // convert to node hyperslab (unique nodes)
    hsize_t nstart[3] = { cstart[0], cstart[1], cstart[2] };
    hsize_t ncount[3] = {ccount[0] + (vct->getCoordinates(0) == vct->getXLEN()-1 ? 1 : 0),
                         ccount[1] + (vct->getCoordinates(1) == vct->getYLEN()-1 ? 1 : 0),
                         ccount[2] + (vct->getCoordinates(2) == vct->getZLEN()-1 ? 1 : 0)};

    const int lx = (int)ncount[0]; const int ly = (int)ncount[1]; const int lz = (int)ncount[2];
    const hsize_t Gx = (hsize_t)(col->getNxc() + 1); const hsize_t Gy = (hsize_t)(col->getNyc() + 1); const hsize_t Gz = (hsize_t)(col->getNzc() + 1);

    double L[3] = {col->getLx(), col->getLy(), col->getLz()};
    int dglob_tmp[3] = {(int)Gx, (int)Gy, (int)Gz};
    int dlocl_tmp[3] = {lx, ly, lz};

    //? Buffer vector
    const size_t nloc = (size_t)lx * (size_t)ly * (size_t)lz;
    static std::vector<float> buf_f(nloc);
    static std::vector<double> buf_d(nloc);
    string precision = col->get_output_data_precision();

    //* E field (defined at nodes)
    if (contains_tag(col->getFieldOutputTag(), "E"))
    {
        //* Create a file to store E
        const std::string filename = fields_folder + "E_" + filenmbr.str() + ".h5";
        PHDF5fileClass outputfile(filename, 3, vct->getCoordinates(), vct->getFieldComm());
        outputfile.CreatePHDF5file(L, dglob_tmp, dlocl_tmp, "Fields");

        //* Write data to file
        if (precision == "SINGLE")
        {
            pack_sca_nodes(buf_f.data(), lx, ly, lz, [&](int i,int j,int k){ return EMf->getEx(i,j,k); });
            outputfile.WritePHDF5dataset_nodes_f32("Fields", "Ex", buf_f.data(), Gn, nstart, ncount);
            pack_sca_nodes(buf_f.data(), lx, ly, lz, [&](int i,int j,int k){ return EMf->getEy(i,j,k); });
            outputfile.WritePHDF5dataset_nodes_f32("Fields", "Ey", buf_f.data(), Gn, nstart, ncount);
            pack_sca_nodes(buf_f.data(), lx, ly, lz, [&](int i,int j,int k){ return EMf->getEz(i,j,k); });
            outputfile.WritePHDF5dataset_nodes_f32("Fields", "Ez", buf_f.data(), Gn, nstart, ncount);
        }
        else if (precision == "DOUBLE")
        {
            pack_sca_nodes(buf_d.data(), lx, ly, lz, [&](int i,int j,int k){ return EMf->getEx(i,j,k); });
            outputfile.WritePHDF5dataset_nodes_f64("Fields", "Ex", buf_d.data(), Gn, nstart, ncount);
            pack_sca_nodes(buf_d.data(), lx, ly, lz, [&](int i,int j,int k){ return EMf->getEy(i,j,k); });
            outputfile.WritePHDF5dataset_nodes_f64("Fields", "Ey", buf_d.data(), Gn, nstart, ncount);
            pack_sca_nodes(buf_d.data(), lx, ly, lz, [&](int i,int j,int k){ return EMf->getEz(i,j,k); });
            outputfile.WritePHDF5dataset_nodes_f64("Fields", "Ez", buf_d.data(), Gn, nstart, ncount);
        }

        //* Close file
        outputfile.ClosePHDF5file();
    }

    //* B field (defined at nodes)
    if (contains_tag(col->getFieldOutputTag(), "B"))
    {
        //* Create a file to store B
        const std::string filename = fields_folder + "B_" + filenmbr.str() + ".h5";
        PHDF5fileClass outputfile(filename, 3, vct->getCoordinates(), vct->getFieldComm());
        outputfile.CreatePHDF5file(L, dglob_tmp, dlocl_tmp, "Fields");

        //* Write data to file
        if (precision == "SINGLE")
        {
            pack_sca_nodes(buf_f.data(), lx, ly, lz, [&](int i,int j,int k){ return EMf->getBx(i,j,k); });
            outputfile.WritePHDF5dataset_nodes_f32("Fields", "Bx", buf_f.data(), Gn, nstart, ncount);
            pack_sca_nodes(buf_f.data(), lx, ly, lz, [&](int i,int j,int k){ return EMf->getBy(i,j,k); });
            outputfile.WritePHDF5dataset_nodes_f32("Fields", "By", buf_f.data(), Gn, nstart, ncount);
            pack_sca_nodes(buf_f.data(), lx, ly, lz, [&](int i,int j,int k){ return EMf->getBz(i,j,k); });
            outputfile.WritePHDF5dataset_nodes_f32("Fields", "Bz", buf_f.data(), Gn, nstart, ncount);
        }
        else if (precision == "DOUBLE")
        {
            pack_sca_nodes(buf_d.data(), lx, ly, lz, [&](int i,int j,int k){ return EMf->getBx(i,j,k); });
            outputfile.WritePHDF5dataset_nodes_f64("Fields", "Bx", buf_d.data(), Gn, nstart, ncount);
            pack_sca_nodes(buf_d.data(), lx, ly, lz, [&](int i,int j,int k){ return EMf->getBy(i,j,k); });
            outputfile.WritePHDF5dataset_nodes_f64("Fields", "By", buf_d.data(), Gn, nstart, ncount);
            pack_sca_nodes(buf_d.data(), lx, ly, lz, [&](int i,int j,int k){ return EMf->getBz(i,j,k); });
            outputfile.WritePHDF5dataset_nodes_f64("Fields", "Bz", buf_d.data(), Gn, nstart, ncount);
        }

        //* Close file
        outputfile.ClosePHDF5file();
    }

    //* Moments for each species (defined at nodes)
    for (int is = 0; is < col->getNs(); ++is) 
    {
        stringstream ii;
		ii << is;

        //* rhos
        if (contains_tag(col->getFieldOutputTag(), "rho_s"))
        {
            //* Create a file to store rho
            const std::string filename = moments_folder + "rho_species_" + ii.str() + "_" + filenmbr.str() + ".h5";
            PHDF5fileClass outputfile(filename, 3, vct->getCoordinates(), vct->getFieldComm());
            outputfile.CreatePHDF5file(L, dglob_tmp, dlocl_tmp, "/Moments");

            //* Write data to file
            if (precision == "SINGLE")
            {
                pack_sca_nodes(buf_f.data(), lx, ly, lz, [&](int i,int j,int k){ return EMf->getRHOns(i, j, k, is); });
                outputfile.WritePHDF5dataset_nodes_f32("Moments/species_" + ii.str(), "rho" , buf_f.data(), Gn, nstart, ncount);
            }
            else if (precision == "DOUBLE")
            {
                pack_sca_nodes(buf_d.data(), lx, ly, lz, [&](int i,int j,int k){ return EMf->getRHOns(i, j, k, is); });
                outputfile.WritePHDF5dataset_nodes_f64("Moments/species_" + ii.str(), "rho" , buf_d.data(), Gn, nstart, ncount);
            }

            //* Close file
            outputfile.ClosePHDF5file();
        }

        //* Js
        if (contains_tag(col->getFieldOutputTag(), "J_s"))
        {
            //* Create a file to store current
            const std::string filename = moments_folder + "J_species_" + ii.str() + "_" + filenmbr.str() + ".h5";
            PHDF5fileClass outputfile(filename, 3, vct->getCoordinates(), vct->getFieldComm());
            outputfile.CreatePHDF5file(L, dglob_tmp, dlocl_tmp, "Moments");

            //* Write data to file
            if (precision == "SINGLE")
            {
                pack_sca_nodes(buf_f.data(), lx, ly, lz, [&](int i,int j,int k){ return EMf->getJxs(i, j, k, is); });
                outputfile.WritePHDF5dataset_nodes_f32("Moments/species_" + ii.str(), "Jx" , buf_f.data(), Gn, nstart, ncount);
                pack_sca_nodes(buf_f.data(), lx, ly, lz, [&](int i,int j,int k){ return EMf->getJys(i, j, k, is); });
                outputfile.WritePHDF5dataset_nodes_f32("Moments/species_" + ii.str(), "Jy" , buf_f.data(), Gn, nstart, ncount);
                pack_sca_nodes(buf_f.data(), lx, ly, lz, [&](int i,int j,int k){ return EMf->getJzs(i, j, k, is); });
                outputfile.WritePHDF5dataset_nodes_f32("Moments/species_" + ii.str(), "Jz" , buf_f.data(), Gn, nstart, ncount);
            }
            else if (precision == "DOUBLE")
            {
                pack_sca_nodes(buf_d.data(), lx, ly, lz, [&](int i,int j,int k){ return EMf->getJxs(i, j, k, is); });
                outputfile.WritePHDF5dataset_nodes_f64("Moments/species_" + ii.str(), "Jx" , buf_d.data(), Gn, nstart, ncount);
                pack_sca_nodes(buf_d.data(), lx, ly, lz, [&](int i,int j,int k){ return EMf->getJys(i, j, k, is); });
                outputfile.WritePHDF5dataset_nodes_f64("Moments/species_" + ii.str(), "Jy" , buf_d.data(), Gn, nstart, ncount);
                pack_sca_nodes(buf_d.data(), lx, ly, lz, [&](int i,int j,int k){ return EMf->getJzs(i, j, k, is); });
                outputfile.WritePHDF5dataset_nodes_f64("Moments/species_" + ii.str(), "Jz" , buf_d.data(), Gn, nstart, ncount);
            }

            //* Close file
            outputfile.ClosePHDF5file();
        }

        //* Pressure tensor
        if (contains_tag(col->getFieldOutputTag(), "pressure"))
        {
            //* Create a file to store pressure tensor
            const std::string filename = moments_folder + "Pressure_species_" + ii.str() + "_" + filenmbr.str() + ".h5";
            PHDF5fileClass outputfile(filename, 3, vct->getCoordinates(), vct->getFieldComm());
            outputfile.CreatePHDF5file(L, dglob_tmp, dlocl_tmp, "Moments");

            //* Write data to file
            if (precision == "SINGLE")
            {
                pack_sca_nodes(buf_f.data(), lx, ly, lz, [&](int i,int j,int k){ return EMf->getpXXsn(i, j, k, is); });
                outputfile.WritePHDF5dataset_nodes_f32("Moments/species_" + ii.str(), "pXX" , buf_f.data(), Gn, nstart, ncount);
                pack_sca_nodes(buf_f.data(), lx, ly, lz, [&](int i,int j,int k){ return EMf->getpXYsn(i, j, k, is); });
                outputfile.WritePHDF5dataset_nodes_f32("Moments/species_" + ii.str(), "pXY" , buf_f.data(), Gn, nstart, ncount);
                pack_sca_nodes(buf_f.data(), lx, ly, lz, [&](int i,int j,int k){ return EMf->getpXZsn(i, j, k, is); });
                outputfile.WritePHDF5dataset_nodes_f32("Moments/species_" + ii.str(), "pXZ" , buf_f.data(), Gn, nstart, ncount);
                pack_sca_nodes(buf_f.data(), lx, ly, lz, [&](int i,int j,int k){ return EMf->getpYYsn(i, j, k, is); });
                outputfile.WritePHDF5dataset_nodes_f32("Moments/species_" + ii.str(), "pYY" , buf_f.data(), Gn, nstart, ncount);
                pack_sca_nodes(buf_f.data(), lx, ly, lz, [&](int i,int j,int k){ return EMf->getpYZsn(i, j, k, is); });
                outputfile.WritePHDF5dataset_nodes_f32("Moments/species_" + ii.str(), "pYZ" , buf_f.data(), Gn, nstart, ncount);
                pack_sca_nodes(buf_f.data(), lx, ly, lz, [&](int i,int j,int k){ return EMf->getpZZsn(i, j, k, is); });
                outputfile.WritePHDF5dataset_nodes_f32("Moments/species_" + ii.str(), "pZZ" , buf_f.data(), Gn, nstart, ncount);
            }
            else if (precision == "DOUBLE")
            {
                pack_sca_nodes(buf_d.data(), lx, ly, lz, [&](int i,int j,int k){ return EMf->getpXXsn(i, j, k, is); });
                outputfile.WritePHDF5dataset_nodes_f64("Moments/species_" + ii.str(), "pXX" , buf_d.data(), Gn, nstart, ncount);
                pack_sca_nodes(buf_d.data(), lx, ly, lz, [&](int i,int j,int k){ return EMf->getpXYsn(i, j, k, is); });
                outputfile.WritePHDF5dataset_nodes_f64("Moments/species_" + ii.str(), "pXY" , buf_d.data(), Gn, nstart, ncount);
                pack_sca_nodes(buf_d.data(), lx, ly, lz, [&](int i,int j,int k){ return EMf->getpXZsn(i, j, k, is); });
                outputfile.WritePHDF5dataset_nodes_f64("Moments/species_" + ii.str(), "pXZ" , buf_d.data(), Gn, nstart, ncount);
                pack_sca_nodes(buf_d.data(), lx, ly, lz, [&](int i,int j,int k){ return EMf->getpYYsn(i, j, k, is); });
                outputfile.WritePHDF5dataset_nodes_f64("Moments/species_" + ii.str(), "pYY" , buf_d.data(), Gn, nstart, ncount);
                pack_sca_nodes(buf_d.data(), lx, ly, lz, [&](int i,int j,int k){ return EMf->getpYZsn(i, j, k, is); });
                outputfile.WritePHDF5dataset_nodes_f64("Moments/species_" + ii.str(), "pYZ" , buf_d.data(), Gn, nstart, ncount);
                pack_sca_nodes(buf_d.data(), lx, ly, lz, [&](int i,int j,int k){ return EMf->getpZZsn(i, j, k, is); });
                outputfile.WritePHDF5dataset_nodes_f64("Moments/species_" + ii.str(), "pZZ" , buf_d.data(), Gn, nstart, ncount);
            }

            //* Close file
            outputfile.ClosePHDF5file();
        }

        //* Energy Flux
        if (contains_tag(col->getFieldOutputTag(), "E_flux"))
        {
            //* Create a file to store energy flux
            const std::string filename = moments_folder + "E_flux_species_" + ii.str() + "_" + filenmbr.str() + ".h5";
            PHDF5fileClass outputfile(filename, 3, vct->getCoordinates(), vct->getFieldComm());
            outputfile.CreatePHDF5file(L, dglob_tmp, dlocl_tmp, "Moments");

            //* Write data to file
            if (precision == "SINGLE")
            {
                pack_sca_nodes(buf_f.data(), lx, ly, lz, [&](int i,int j,int k){ return EMf->getEFxs(i, j, k, is); });
                outputfile.WritePHDF5dataset_nodes_f32("Moments/species_" + ii.str(), "EFx" , buf_f.data(), Gn, nstart, ncount);
                pack_sca_nodes(buf_f.data(), lx, ly, lz, [&](int i,int j,int k){ return EMf->getEFys(i, j, k, is); });
                outputfile.WritePHDF5dataset_nodes_f32("Moments/species_" + ii.str(), "EFy" , buf_f.data(), Gn, nstart, ncount);
                pack_sca_nodes(buf_f.data(), lx, ly, lz, [&](int i,int j,int k){ return EMf->getEFzs(i, j, k, is); });
                outputfile.WritePHDF5dataset_nodes_f32("Moments/species_" + ii.str(), "EFz" , buf_f.data(), Gn, nstart, ncount);
            }
            else if (precision == "DOUBLE")
            {
                pack_sca_nodes(buf_d.data(), lx, ly, lz, [&](int i,int j,int k){ return EMf->getEFxs(i, j, k, is); });
                outputfile.WritePHDF5dataset_nodes_f64("Moments/species_" + ii.str(), "EFx" , buf_d.data(), Gn, nstart, ncount);
                pack_sca_nodes(buf_d.data(), lx, ly, lz, [&](int i,int j,int k){ return EMf->getEFys(i, j, k, is); });
                outputfile.WritePHDF5dataset_nodes_f64("Moments/species_" + ii.str(), "EFy" , buf_d.data(), Gn, nstart, ncount);
                pack_sca_nodes(buf_d.data(), lx, ly, lz, [&](int i,int j,int k){ return EMf->getEFzs(i, j, k, is); });
                outputfile.WritePHDF5dataset_nodes_f64("Moments/species_" + ii.str(), "EFz" , buf_d.data(), Gn, nstart, ncount);
            }

            //* Close file
            outputfile.ClosePHDF5file();
        }

        //* Heat Flux
        if (contains_tag(col->getFieldOutputTag(), "H_flux"))
        {
            //* Create a file to store energy flux
            const std::string filename = moments_folder + "H_flux_species_" + ii.str() + "_" + filenmbr.str() + ".h5";
            PHDF5fileClass outputfile(filename, 3, vct->getCoordinates(), vct->getFieldComm());
            outputfile.CreatePHDF5file(L, dglob_tmp, dlocl_tmp, "Moments");

            //* Write data to file
            if (precision == "SINGLE")
            {
                pack_sca_nodes(buf_f.data(), lx, ly, lz, [&](int i,int j,int k){ return EMf->getQxxxs(i, j, k, is); });
                outputfile.WritePHDF5dataset_nodes_f32("Moments/species_" + ii.str(), "Qxxxs" , buf_f.data(), Gn, nstart, ncount);
                pack_sca_nodes(buf_f.data(), lx, ly, lz, [&](int i,int j,int k){ return EMf->getQxxys(i, j, k, is); });
                outputfile.WritePHDF5dataset_nodes_f32("Moments/species_" + ii.str(), "Qxxys" , buf_f.data(), Gn, nstart, ncount);
                pack_sca_nodes(buf_f.data(), lx, ly, lz, [&](int i,int j,int k){ return EMf->getQxyys(i, j, k, is); });
                outputfile.WritePHDF5dataset_nodes_f32("Moments/species_" + ii.str(), "Qxyys" , buf_f.data(), Gn, nstart, ncount);
                pack_sca_nodes(buf_f.data(), lx, ly, lz, [&](int i,int j,int k){ return EMf->getQxzzs(i, j, k, is); });
                outputfile.WritePHDF5dataset_nodes_f32("Moments/species_" + ii.str(), "Qxzzs" , buf_f.data(), Gn, nstart, ncount);
                pack_sca_nodes(buf_f.data(), lx, ly, lz, [&](int i,int j,int k){ return EMf->getQyyys(i, j, k, is); });
                outputfile.WritePHDF5dataset_nodes_f32("Moments/species_" + ii.str(), "Qyyys" , buf_f.data(), Gn, nstart, ncount);
                pack_sca_nodes(buf_f.data(), lx, ly, lz, [&](int i,int j,int k){ return EMf->getQyzzs(i, j, k, is); });
                outputfile.WritePHDF5dataset_nodes_f32("Moments/species_" + ii.str(), "Qyzzs" , buf_f.data(), Gn, nstart, ncount);
                pack_sca_nodes(buf_f.data(), lx, ly, lz, [&](int i,int j,int k){ return EMf->getQzzzs(i, j, k, is); });
                outputfile.WritePHDF5dataset_nodes_f32("Moments/species_" + ii.str(), "Qzzzs" , buf_f.data(), Gn, nstart, ncount);
                pack_sca_nodes(buf_f.data(), lx, ly, lz, [&](int i,int j,int k){ return EMf->getQxyzs(i, j, k, is); });
                outputfile.WritePHDF5dataset_nodes_f32("Moments/species_" + ii.str(), "Qxyzs" , buf_f.data(), Gn, nstart, ncount);
                pack_sca_nodes(buf_f.data(), lx, ly, lz, [&](int i,int j,int k){ return EMf->getQxxzs(i, j, k, is); });
                outputfile.WritePHDF5dataset_nodes_f32("Moments/species_" + ii.str(), "Qxxzs" , buf_f.data(), Gn, nstart, ncount);
                pack_sca_nodes(buf_f.data(), lx, ly, lz, [&](int i,int j,int k){ return EMf->getQyyzs(i, j, k, is); });
                outputfile.WritePHDF5dataset_nodes_f32("Moments/species_" + ii.str(), "Qyyzs" , buf_f.data(), Gn, nstart, ncount);
            }
            else if (precision == "DOUBLE")
            {
                pack_sca_nodes(buf_d.data(), lx, ly, lz, [&](int i,int j,int k){ return EMf->getQxxxs(i, j, k, is); });
                outputfile.WritePHDF5dataset_nodes_f64("Moments/species_" + ii.str(), "Qxxxs" , buf_d.data(), Gn, nstart, ncount);
                pack_sca_nodes(buf_d.data(), lx, ly, lz, [&](int i,int j,int k){ return EMf->getQxxys(i, j, k, is); });
                outputfile.WritePHDF5dataset_nodes_f64("Moments/species_" + ii.str(), "Qxxys" , buf_d.data(), Gn, nstart, ncount);
                pack_sca_nodes(buf_d.data(), lx, ly, lz, [&](int i,int j,int k){ return EMf->getQxyys(i, j, k, is); });
                outputfile.WritePHDF5dataset_nodes_f64("Moments/species_" + ii.str(), "Qxyys" , buf_d.data(), Gn, nstart, ncount);
                pack_sca_nodes(buf_d.data(), lx, ly, lz, [&](int i,int j,int k){ return EMf->getQxzzs(i, j, k, is); });
                outputfile.WritePHDF5dataset_nodes_f64("Moments/species_" + ii.str(), "Qxzzs" , buf_d.data(), Gn, nstart, ncount);
                pack_sca_nodes(buf_d.data(), lx, ly, lz, [&](int i,int j,int k){ return EMf->getQyyys(i, j, k, is); });
                outputfile.WritePHDF5dataset_nodes_f64("Moments/species_" + ii.str(), "Qyyys" , buf_d.data(), Gn, nstart, ncount);
                pack_sca_nodes(buf_d.data(), lx, ly, lz, [&](int i,int j,int k){ return EMf->getQyzzs(i, j, k, is); });
                outputfile.WritePHDF5dataset_nodes_f64("Moments/species_" + ii.str(), "Qyzzs" , buf_d.data(), Gn, nstart, ncount);
                pack_sca_nodes(buf_d.data(), lx, ly, lz, [&](int i,int j,int k){ return EMf->getQzzzs(i, j, k, is); });
                outputfile.WritePHDF5dataset_nodes_f64("Moments/species_" + ii.str(), "Qzzzs" , buf_d.data(), Gn, nstart, ncount);
                pack_sca_nodes(buf_d.data(), lx, ly, lz, [&](int i,int j,int k){ return EMf->getQxyzs(i, j, k, is); });
                outputfile.WritePHDF5dataset_nodes_f64("Moments/species_" + ii.str(), "Qxyzs" , buf_d.data(), Gn, nstart, ncount);
                pack_sca_nodes(buf_d.data(), lx, ly, lz, [&](int i,int j,int k){ return EMf->getQxxzs(i, j, k, is); });
                outputfile.WritePHDF5dataset_nodes_f64("Moments/species_" + ii.str(), "Qxxzs" , buf_d.data(), Gn, nstart, ncount);
                pack_sca_nodes(buf_d.data(), lx, ly, lz, [&](int i,int j,int k){ return EMf->getQyyzs(i, j, k, is); });
                outputfile.WritePHDF5dataset_nodes_f64("Moments/species_" + ii.str(), "Qyyzs" , buf_d.data(), Gn, nstart, ncount);
            }

            //* Close file
            outputfile.ClosePHDF5file();
        }
    }
#else
    eprintf("Parallel HDF5 requested but iPIC3D has compiled without parallel HDF5 library.");
#endif
}

void WriteParticlesParallel(Particles3D* particles, CollectiveIO *col, VCtopology3D *vct, int cycle)
{
#ifdef PHDF5

    if (vct->getCartesian_rank() == 0)
        cout << endl << "Writing PARTICLE data at cycle " << cycle << endl;

    std::stringstream filenmbr;
    filenmbr << std::setfill('0') << std::setw(5) << cycle;
    const std::string particles_folder = col->getSaveDirName() + "/Particles_" + filenmbr.str() + "/";

    if (contains_tag(col->getPclOutputTag(), "position") || contains_tag(col->getPclOutputTag(), "velocity")  || contains_tag(col->getPclOutputTag(), "q"))
    {        
        if (MPIdata::get_rank() == 0)
            std::filesystem::create_directories(particles_folder);

        // MPI_Barrier(MPI_COMM_WORLD);
    }

    //? Buffer vector
    static std::vector<float> particle_single;
    static std::vector<double> particle_double;
    string precision = col->get_output_data_precision();

    //* Particle data for each species
    for (int is = 0; is < col->getNs(); ++is) 
    {
        size_t nop = particles[is].getNOP();
        if (nop < 0) abort();

        stringstream ii;
		ii << is;

        const hsize_t nlocal = static_cast<hsize_t>(nop);

        //* Sizes of arrays where position, velocity, and q are to be written
        hsize_t Nglobal = 0;
        MPI_Allreduce(&nlocal, &Nglobal, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, vct->getFieldComm());

        hsize_t offset = 0;
        MPI_Exscan(&nlocal, &offset, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, vct->getFieldComm());

        hsize_t gdim[2]  = { Nglobal, 3 }; hsize_t gdim_q[2]  = { 1, 1 };
        hsize_t start[2] = { offset, 0 }; hsize_t start_q[2] = { 0, 0 }; 
        const hsize_t one_or_zero = (MPIdata::get_rank() == 0) ? hsize_t(1) : hsize_t(0);
        hsize_t count[2] = { nlocal, 3 }; hsize_t count_q[2] = { one_or_zero, 1 };

        //* Create a file to store postions, velocities, and charges
        const std::string filename = particles_folder + "species_" + ii.str() + "_" + filenmbr.str() + ".h5";
        PHDF5fileClass outputfile(filename, 3, vct->getCoordinates(), vct->getFieldComm());
        outputfile.CreatePHDF5fileParticles("Particles");

        //* position
        if (contains_tag(col->getPclOutputTag(), "position"))
        {
            //* Write data to file
            if (precision == "SINGLE")
            {
                particle_single.resize(3*nlocal);
                pack_xyz_from_aos<float>(particle_single.data(), nop, particles[is]);
                outputfile.WritePHDF5dataset_particles_f32("Particles/species_" + ii.str(), "position", particle_single.data(), gdim, start, count);
            }
            else if (precision == "DOUBLE")
            {
                particle_double.resize(3*nlocal);
                pack_xyz_from_aos<double>(particle_double.data(), nop, particles[is]);
                outputfile.WritePHDF5dataset_particles_f64("Particles/species_" + ii.str(), "position", particle_double.data(), gdim, start, count);
            }
        }

        //* velocity
        if (contains_tag(col->getPclOutputTag(), "velocity"))
        {
            //* Write data to file
            if (precision == "SINGLE")
            {
                particle_single.resize(3*nlocal);
                pack_uvw_from_aos<float>(particle_single.data(), nop, particles[is]);
                outputfile.WritePHDF5dataset_particles_f32("Particles/species_" + ii.str(), "velocity", particle_single.data(), gdim, start, count);
            }
            else if (precision == "DOUBLE")
            {
                particle_double.resize(3*nlocal);
                pack_uvw_from_aos<double>(particle_double.data(), nop, particles[is]);
                outputfile.WritePHDF5dataset_particles_f64("Particles/species_" + ii.str(), "velocity", particle_double.data(), gdim, start, count);
            }
        }

        //* charge
        if (contains_tag(col->getPclOutputTag(), "q"))
        {
            //* Write data to file
            if (precision == "SINGLE")
            {
                particle_single.resize(1);
                pack_q_from_aos<float>(particle_single.data(), nop, particles[is]);
                outputfile.WritePHDF5dataset_particles_f32("Particles/species_" + ii.str(), "q", particle_single.data(), gdim_q, start_q, count_q);
            }
            else if (precision == "DOUBLE")
            {
                particle_double.resize(1);
                pack_q_from_aos<double>(particle_double.data(), nop, particles[is]);
                outputfile.WritePHDF5dataset_particles_f64("Particles/species_" + ii.str(), "q", particle_double.data(), gdim_q, start_q, count_q);
            }
        }

        //* Close file
        outputfile.ClosePHDF5file();
    }

#else
    eprintf("Parallel HDF5 requested but iPIC3D has compiled without parallel HDF5 library.");
#endif
}


//template<typename T, int sz>
//int size(T(&)[sz])
//{
//    return sz;
//}

void WriteFieldsVTK(Grid3DCU *grid, EMfields3D *EMf, CollectiveIO *col, VCtopology3D *vct, const string & outputTag ,int cycle)
{
	//All VTK output at grid cells excluding ghost cells
	const int nxn  =grid->getNXN(),nyn  = grid->getNYN(),nzn =grid->getNZN();
	const int dimX =col->getNxc() ,dimY = col->getNyc(), dimZ=col->getNzc();
	const double spaceX = dimX>1 ?col->getLx()/(dimX-1) :col->getLx();
	const double spaceY = dimY>1 ?col->getLy()/(dimY-1) :col->getLy();
	const double spaceZ = dimZ>1 ?col->getLz()/(dimZ-1) :col->getLz();
	const int    nPoints = dimX*dimY*dimZ;
	MPI_File     fh;
	MPI_Status   status;

	if (outputTag.find("B", 0) != string::npos || outputTag.find("E", 0) != string::npos
			 || outputTag.find("Je", 0) != string::npos || outputTag.find("Ji", 0) != string::npos){

		const string tags0[]={"B", "E", "Je", "Ji"};
		float writebuffer[nzn-3][nyn-3][nxn-3][3];
		float tmpX, tmpY, tmpZ;

		for(int tagid=0;tagid<4;tagid++){
		 if (outputTag.find(tags0[tagid], 0) == string::npos) continue;

		 char   header[1024];
		 if (tags0[tagid].compare("B") == 0){
			 for(int iz=0;iz<nzn-3;iz++)
				  for(int iy=0;iy<nyn-3;iy++)
					  for(int ix= 0;ix<nxn-3;ix++){
						  tmpX = (float)EMf->getBxTot(ix+1, iy+1, iz+1);
						  tmpY = (float)EMf->getByTot(ix+1, iy+1, iz+1);
						  tmpZ = (float)EMf->getBzTot(ix+1, iy+1, iz+1);

						  writebuffer[iz][iy][ix][0] =  (fabs(tmpX) < 1E-16) ?0.0 : tmpX;
						  writebuffer[iz][iy][ix][1] =  (fabs(tmpY) < 1E-16) ?0.0 : tmpY;
						  writebuffer[iz][iy][ix][2] =  (fabs(tmpZ) < 1E-16) ?0.0 : tmpZ;
					  }

	      //Write VTK header
		  sprintf(header, "# vtk DataFile Version 2.0\n"
						   "Magnetic Field from iPIC3D\n"
						   "BINARY\n"
						   "DATASET STRUCTURED_POINTS\n"
						   "DIMENSIONS %d %d %d\n"
						   "ORIGIN 0 0 0\n"
						   "SPACING %f %f %f\n"
						   "POINT_DATA %d\n"
						   "VECTORS B float\n", dimX,dimY,dimZ, spaceX,spaceY,spaceZ, nPoints);

		 }else if (tags0[tagid].compare("E") == 0){
			 for(int iz=0;iz<nzn-3;iz++)
				  for(int iy=0;iy<nyn-3;iy++)
					  for(int ix= 0;ix<nxn-3;ix++){
						  writebuffer[iz][iy][ix][0] = (float)EMf->getEx(ix+1, iy+1, iz+1);
						  writebuffer[iz][iy][ix][1] = (float)EMf->getEy(ix+1, iy+1, iz+1);
						  writebuffer[iz][iy][ix][2] = (float)EMf->getEz(ix+1, iy+1, iz+1);
					  }

		  //Write VTK header
		   sprintf(header, "# vtk DataFile Version 2.0\n"
						   "Electric Field from iPIC3D\n"
						   "BINARY\n"
						   "DATASET STRUCTURED_POINTS\n"
						   "DIMENSIONS %d %d %d\n"
						   "ORIGIN 0 0 0\n"
						   "SPACING %f %f %f\n"
						   "POINT_DATA %d \n"
						   "VECTORS E float\n", dimX,dimY,dimZ, spaceX,spaceY,spaceZ, nPoints);

		 }else if (tags0[tagid].compare("Je") == 0){
			 for(int iz=0;iz<nzn-3;iz++)
				  for(int iy=0;iy<nyn-3;iy++)
					  for(int ix= 0;ix<nxn-3;ix++){
						  writebuffer[iz][iy][ix][0] = (float)EMf->getJxs(ix+1, iy+1, iz+1, 0);
						  writebuffer[iz][iy][ix][1] = (float)EMf->getJys(ix+1, iy+1, iz+1, 0);
						  writebuffer[iz][iy][ix][2] = (float)EMf->getJzs(ix+1, iy+1, iz+1, 0);
					  }

		  //Write VTK header
		   sprintf(header, "# vtk DataFile Version 2.0\n"
						   "Electron current from iPIC3D\n"
						   "BINARY\n"
						   "DATASET STRUCTURED_POINTS\n"
						   "DIMENSIONS %d %d %d\n"
						   "ORIGIN 0 0 0\n"
						   "SPACING %f %f %f\n"
						   "POINT_DATA %d \n"
						   "VECTORS Je float\n", dimX,dimY,dimZ, spaceX,spaceY,spaceZ, nPoints);

		 }else if (tags0[tagid].compare("Ji") == 0){
			 for(int iz=0;iz<nzn-3;iz++)
				  for(int iy=0;iy<nyn-3;iy++)
					  for(int ix= 0;ix<nxn-3;ix++){
						  writebuffer[iz][iy][ix][0] = (float)EMf->getJxs(ix+1, iy+1, iz+1, 1);
						  writebuffer[iz][iy][ix][1] = (float)EMf->getJys(ix+1, iy+1, iz+1, 1);
						  writebuffer[iz][iy][ix][2] = (float)EMf->getJzs(ix+1, iy+1, iz+1, 1);
					  }

		  //Write VTK header
		   sprintf(header, "# vtk DataFile Version 2.0\n"
						   "Ion current from iPIC3D\n"
						   "BINARY\n"
						   "DATASET STRUCTURED_POINTS\n"
						   "DIMENSIONS %d %d %d\n"
						   "ORIGIN 0 0 0\n"
						   "SPACING %f %f %f\n"
						   "POINT_DATA %d \n"
						   "VECTORS Ji float\n", dimX,dimY,dimZ, spaceX,spaceY,spaceZ, nPoints);
		 }


		 if(EMf->isLittleEndian()){

			 for(int iz=0;iz<nzn-3;iz++)
				  for(int iy=0;iy<nyn-3;iy++)
					  for(int ix= 0;ix<nxn-3;ix++){
						  ByteSwap((unsigned char*) &writebuffer[iz][iy][ix][0],4);
						  ByteSwap((unsigned char*) &writebuffer[iz][iy][ix][1],4);
						  ByteSwap((unsigned char*) &writebuffer[iz][iy][ix][2],4);
					  }
		 }

		  int nelem = strlen(header);
		  int charsize=sizeof(char);
		  MPI_Offset disp = nelem*charsize;

		  ostringstream filename;
		  filename << col->getSaveDirName() << "/" << col->getSimName() << "_"<< tags0[tagid] << "_" << cycle << ".vtk";
		  MPI_File_open(vct->getFieldComm(),filename.str().c_str(), MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);

		  MPI_File_set_view(fh, 0, MPI_BYTE, MPI_BYTE, "native", MPI_INFO_NULL);
		  if (vct->getCartesian_rank()==0){
			  MPI_File_write(fh, header, nelem, MPI_BYTE, &status);
		  }

	      int err = MPI_File_set_view(fh, disp, EMf->getXYZeType(), EMf->getProcviewXYZ(), "native", MPI_INFO_NULL);
	      if(err){
	          dprintf("Error in MPI_File_set_view\n");

	          int error_code = status.MPI_ERROR;
	          if (error_code != MPI_SUCCESS) {
	              char error_string[100];
	              int length_of_error_string, error_class;

	              MPI_Error_class(error_code, &error_class);
	              MPI_Error_string(error_class, error_string, &length_of_error_string);
	              dprintf("Error %s\n", error_string);
	          }
	      }

	      err = MPI_File_write_all(fh, writebuffer[0][0][0],3*(nxn-3)*(nyn-3)*(nzn-3),MPI_FLOAT, &status);
	      if(err){
		      int tcount=0;
		      MPI_Get_count(&status, MPI_DOUBLE, &tcount);
			  dprintf(" wrote %i",  tcount);
	          dprintf("Error in write1\n");
	          int error_code = status.MPI_ERROR;
	          if (error_code != MPI_SUCCESS) {
	              char error_string[100];
	              int length_of_error_string, error_class;

	              MPI_Error_class(error_code, &error_class);
	              MPI_Error_string(error_class, error_string, &length_of_error_string);
	              dprintf("Error %s\n", error_string);
	          }
	      }
	      MPI_File_close(&fh);
		}
	}

	if (outputTag.find("rho", 0) != string::npos){

		float writebufferRhoe[nzn-3][nyn-3][nxn-3];
		float writebufferRhoi[nzn-3][nyn-3][nxn-3];
		char   headerRhoe[1024];
		char   headerRhoi[1024];

		for(int iz=0;iz<nzn-3;iz++)
		  for(int iy=0;iy<nyn-3;iy++)
			  for(int ix= 0;ix<nxn-3;ix++){
				  writebufferRhoe[iz][iy][ix] = (float)EMf->getRHOns(ix+1, iy+1, iz+1, 0)*4*3.1415926535897;
				  writebufferRhoi[iz][iy][ix] = (float)EMf->getRHOns(ix+1, iy+1, iz+1, 1)*4*3.1415926535897;
			  }

		//Write VTK header
		sprintf(headerRhoe, "# vtk DataFile Version 2.0\n"
						   "Electron density from iPIC3D\n"
						   "BINARY\n"
						   "DATASET STRUCTURED_POINTS\n"
						   "DIMENSIONS %d %d %d\n"
						   "ORIGIN 0 0 0\n"
						   "SPACING %f %f %f\n"
						   "POINT_DATA %d \n"
						   "SCALARS rhoe float\n"
						   "LOOKUP_TABLE default\n",  dimX,dimY,dimZ, spaceX,spaceY,spaceZ, nPoints);

		sprintf(headerRhoi, "# vtk DataFile Version 2.0\n"
						   "Ion density from iPIC3D\n"
						   "BINARY\n"
						   "DATASET STRUCTURED_POINTS\n"
						   "DIMENSIONS %d %d %d\n"
						   "ORIGIN 0 0 0\n"
						   "SPACING %f %f %f\n"
						   "POINT_DATA %d \n"
						   "SCALARS rhoi float\n"
						   "LOOKUP_TABLE default\n",  dimX,dimY,dimZ, spaceX,spaceY,spaceZ, nPoints);

		 if(EMf->isLittleEndian()){
			 for(int iz=0;iz<nzn-3;iz++)
				  for(int iy=0;iy<nyn-3;iy++)
					  for(int ix= 0;ix<nxn-3;ix++){
						  ByteSwap((unsigned char*) &writebufferRhoe[iz][iy][ix],4);
						  ByteSwap((unsigned char*) &writebufferRhoi[iz][iy][ix],4);
					  }
		 }

		  int nelem = strlen(headerRhoe);
		  int charsize=sizeof(char);
		  MPI_Offset disp = nelem*charsize;

		  ostringstream filename;
		  filename << col->getSaveDirName() << "/" << col->getSimName() << "_rhoe_" << cycle << ".vtk";
		  MPI_File_open(vct->getFieldComm(),filename.str().c_str(), MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);

		  MPI_File_set_view(fh, 0, MPI_BYTE, MPI_BYTE, "native", MPI_INFO_NULL);
		  if (vct->getCartesian_rank()==0){
			  MPI_File_write(fh, headerRhoe, nelem, MPI_BYTE, &status);
		  }

	      int err = MPI_File_set_view(fh, disp, MPI_FLOAT, EMf->getProcview(), "native", MPI_INFO_NULL);
	      if(err){
	          dprintf("Error in MPI_File_set_view\n");

	          int error_code = status.MPI_ERROR;
	          if (error_code != MPI_SUCCESS) {
	              char error_string[100];
	              int length_of_error_string, error_class;

	              MPI_Error_class(error_code, &error_class);
	              MPI_Error_string(error_class, error_string, &length_of_error_string);
	              dprintf("Error %s\n", error_string);
	          }
	      }

	      err = MPI_File_write_all(fh, writebufferRhoe[0][0],(nxn-3)*(nyn-3)*(nzn-3),MPI_FLOAT, &status);
	      if(err){
		      int tcount=0;
		      MPI_Get_count(&status, MPI_DOUBLE, &tcount);
			  dprintf(" wrote %i",  tcount);
	          dprintf("Error in write1\n");
	          int error_code = status.MPI_ERROR;
	          if (error_code != MPI_SUCCESS) {
	              char error_string[100];
	              int length_of_error_string, error_class;

	              MPI_Error_class(error_code, &error_class);
	              MPI_Error_string(error_class, error_string, &length_of_error_string);
	              dprintf("Error %s\n", error_string);
	          }
	      }
	      MPI_File_close(&fh);

	      nelem = strlen(headerRhoi);
	      disp  = nelem*charsize;

	      filename.str("");
	      filename << col->getSaveDirName() << "/" << col->getSimName() << "_rhoi_" << cycle << ".vtk";
	      MPI_File_open(vct->getFieldComm(),filename.str().c_str(), MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);

	      MPI_File_set_view(fh, 0, MPI_BYTE, MPI_BYTE, "native", MPI_INFO_NULL);
		  if (vct->getCartesian_rank()==0){
			  MPI_File_write(fh, headerRhoi, nelem, MPI_BYTE, &status);
		  }

	      err = MPI_File_set_view(fh, disp, MPI_FLOAT, EMf->getProcview(), "native", MPI_INFO_NULL);
		  if(err){
			  dprintf("Error in MPI_File_set_view\n");

			  int error_code = status.MPI_ERROR;
			  if (error_code != MPI_SUCCESS) {
				  char error_string[100];
				  int length_of_error_string, error_class;

				  MPI_Error_class(error_code, &error_class);
				  MPI_Error_string(error_class, error_string, &length_of_error_string);
				  dprintf("Error %s\n", error_string);
			  }
		  }

		  err = MPI_File_write_all(fh, writebufferRhoi[0][0],(nxn-3)*(nyn-3)*(nzn-3),MPI_FLOAT, &status);
		  if(err){
			  int tcount=0;
			  MPI_Get_count(&status, MPI_DOUBLE, &tcount);
			  dprintf(" wrote %i",  tcount);
			  dprintf("Error in write1\n");
			  int error_code = status.MPI_ERROR;
			  if (error_code != MPI_SUCCESS) {
				  char error_string[100];
				  int length_of_error_string, error_class;

				  MPI_Error_class(error_code, &error_class);
				  MPI_Error_string(error_class, error_string, &length_of_error_string);
				  dprintf("Error %s\n", error_string);
			  }
		  }
		  MPI_File_close(&fh);
	}
}

void WriteFieldsVTK(Grid3DCU *grid, EMfields3D *EMf, CollectiveIO *col, VCtopology3D *vct, const string & outputTag ,int cycle,float**** fieldwritebuffer){

	//All VTK output at grid cells excluding ghost cells
	const int nxn  =grid->getNXN(),nyn  = grid->getNYN(),nzn =grid->getNZN();
	const int dimX =col->getNxc() ,dimY = col->getNyc(), dimZ=col->getNzc();
	const double spaceX = dimX>1 ?col->getLx()/(dimX-1) :col->getLx();
	const double spaceY = dimY>1 ?col->getLy()/(dimY-1) :col->getLy();
	const double spaceZ = dimZ>1 ?col->getLz()/(dimZ-1) :col->getLz();
	const int    nPoints = dimX*dimY*dimZ;
	MPI_File     fh;
	MPI_Status   status;
	const string fieldtags[]={"B", "E", "Je", "Ji","Je2", "Ji3"};
	const int    tagsize = size(fieldtags);
	const string outputtag = col->getFieldOutputTag();

	for(int tagid=0; tagid<tagsize; tagid++){

	 if (outputTag.find(fieldtags[tagid], 0) == string::npos) continue;

	 char   header[1024];
	 if (fieldtags[tagid].compare("B") == 0){
		 for(int iz=0;iz<nzn-3;iz++)
			  for(int iy=0;iy<nyn-3;iy++)
				  for(int ix= 0;ix<nxn-3;ix++){
					  fieldwritebuffer[iz][iy][ix][0] =  (float)EMf->getBxTot(ix+1, iy+1, iz+1);
					  fieldwritebuffer[iz][iy][ix][1] =  (float)EMf->getByTot(ix+1, iy+1, iz+1);
					  fieldwritebuffer[iz][iy][ix][2] =  (float)EMf->getBzTot(ix+1, iy+1, iz+1);
				  }

		 //Write VTK header
		 sprintf(header, "# vtk DataFile Version 2.0\n"
					   "Magnetic Field from iPIC3D\n"
					   "BINARY\n"
					   "DATASET STRUCTURED_POINTS\n"
					   "DIMENSIONS %d %d %d\n"
					   "ORIGIN 0 0 0\n"
					   "SPACING %f %f %f\n"
					   "POINT_DATA %d\n"
					   "VECTORS B float\n", dimX,dimY,dimZ, spaceX,spaceY,spaceZ, nPoints);

	 }else if (fieldtags[tagid].compare("E") == 0){
		 for(int iz=0;iz<nzn-3;iz++)
			  for(int iy=0;iy<nyn-3;iy++)
				  for(int ix= 0;ix<nxn-3;ix++){
					  fieldwritebuffer[iz][iy][ix][0] = (float)EMf->getEx(ix+1, iy+1, iz+1);
					  fieldwritebuffer[iz][iy][ix][1] = (float)EMf->getEy(ix+1, iy+1, iz+1);
					  fieldwritebuffer[iz][iy][ix][2] = (float)EMf->getEz(ix+1, iy+1, iz+1);
				  }

		 //Write VTK header
		 sprintf(header, "# vtk DataFile Version 2.0\n"
					   "Electric Field from iPIC3D\n"
					   "BINARY\n"
					   "DATASET STRUCTURED_POINTS\n"
					   "DIMENSIONS %d %d %d\n"
					   "ORIGIN 0 0 0\n"
					   "SPACING %f %f %f\n"
					   "POINT_DATA %d \n"
					   "VECTORS E float\n", dimX,dimY,dimZ, spaceX,spaceY,spaceZ, nPoints);

	 }else if (fieldtags[tagid].compare("Je") == 0){
		 for(int iz=0;iz<nzn-3;iz++)
			  for(int iy=0;iy<nyn-3;iy++)
				  for(int ix= 0;ix<nxn-3;ix++){
					  fieldwritebuffer[iz][iy][ix][0] = (float)EMf->getJxs(ix+1, iy+1, iz+1, 0);
					  fieldwritebuffer[iz][iy][ix][1] = (float)EMf->getJys(ix+1, iy+1, iz+1, 0);
					  fieldwritebuffer[iz][iy][ix][2] = (float)EMf->getJzs(ix+1, iy+1, iz+1, 0);
				  }

		 //Write VTK header
		 sprintf(header, "# vtk DataFile Version 2.0\n"
					   "Electron current from iPIC3D\n"
					   "BINARY\n"
					   "DATASET STRUCTURED_POINTS\n"
					   "DIMENSIONS %d %d %d\n"
					   "ORIGIN 0 0 0\n"
					   "SPACING %f %f %f\n"
					   "POINT_DATA %d \n"
					   "VECTORS Je float\n", dimX,dimY,dimZ, spaceX,spaceY,spaceZ, nPoints);

	 }else if (fieldtags[tagid].compare("Ji") == 0){
		 for(int iz=0;iz<nzn-3;iz++)
			  for(int iy=0;iy<nyn-3;iy++)
				  for(int ix= 0;ix<nxn-3;ix++){
					  fieldwritebuffer[iz][iy][ix][0] = (float)EMf->getJxs(ix+1, iy+1, iz+1, 1);
					  fieldwritebuffer[iz][iy][ix][1] = (float)EMf->getJys(ix+1, iy+1, iz+1, 1);
					  fieldwritebuffer[iz][iy][ix][2] = (float)EMf->getJzs(ix+1, iy+1, iz+1, 1);
				  }

		 //Write VTK header
		 sprintf(header, "# vtk DataFile Version 2.0\n"
					   "Ion current from iPIC3D\n"
					   "BINARY\n"
					   "DATASET STRUCTURED_POINTS\n"
					   "DIMENSIONS %d %d %d\n"
					   "ORIGIN 0 0 0\n"
					   "SPACING %f %f %f\n"
					   "POINT_DATA %d \n"
					   "VECTORS Ji float\n", dimX,dimY,dimZ, spaceX,spaceY,spaceZ, nPoints);
	 }else if (fieldtags[tagid].compare("Je2") == 0){
		 for(int iz=0;iz<nzn-3;iz++)
			  for(int iy=0;iy<nyn-3;iy++)
				  for(int ix= 0;ix<nxn-3;ix++){
					  fieldwritebuffer[iz][iy][ix][0] = (float)EMf->getJxs(ix+1, iy+1, iz+1, 2);
					  fieldwritebuffer[iz][iy][ix][1] = (float)EMf->getJys(ix+1, iy+1, iz+1, 2);
					  fieldwritebuffer[iz][iy][ix][2] = (float)EMf->getJzs(ix+1, iy+1, iz+1, 2);
				  }

		 //Write VTK header
		 sprintf(header, "# vtk DataFile Version 2.0\n"
				 "Electron2 current from iPIC3D\n"
					   "BINARY\n"
					   "DATASET STRUCTURED_POINTS\n"
					   "DIMENSIONS %d %d %d\n"
					   "ORIGIN 0 0 0\n"
					   "SPACING %f %f %f\n"
					   "POINT_DATA %d \n"
					   "VECTORS Je float\n", dimX,dimY,dimZ, spaceX,spaceY,spaceZ, nPoints);

	 }else if (fieldtags[tagid].compare("Ji3") == 0){
		 for(int iz=0;iz<nzn-3;iz++)
			  for(int iy=0;iy<nyn-3;iy++)
				  for(int ix= 0;ix<nxn-3;ix++){
					  fieldwritebuffer[iz][iy][ix][0] = (float)EMf->getJxs(ix+1, iy+1, iz+1, 3);
					  fieldwritebuffer[iz][iy][ix][1] = (float)EMf->getJys(ix+1, iy+1, iz+1, 3);
					  fieldwritebuffer[iz][iy][ix][2] = (float)EMf->getJzs(ix+1, iy+1, iz+1, 3);
				  }

		 //Write VTK header
		 sprintf(header, "# vtk DataFile Version 2.0\n"
				 "Ion3 current from iPIC3D\n"
					   "BINARY\n"
					   "DATASET STRUCTURED_POINTS\n"
					   "DIMENSIONS %d %d %d\n"
					   "ORIGIN 0 0 0\n"
					   "SPACING %f %f %f\n"
					   "POINT_DATA %d \n"
					   "VECTORS Ji float\n", dimX,dimY,dimZ, spaceX,spaceY,spaceZ, nPoints);
	 }

	 if(EMf->isLittleEndian()){
		 for(int iz=0;iz<nzn-3;iz++)
			  for(int iy=0;iy<nyn-3;iy++)
				  for(int ix= 0;ix<nxn-3;ix++){
					  ByteSwap((unsigned char*) &fieldwritebuffer[iz][iy][ix][0],4);
					  ByteSwap((unsigned char*) &fieldwritebuffer[iz][iy][ix][1],4);
					  ByteSwap((unsigned char*) &fieldwritebuffer[iz][iy][ix][2],4);
				  }
	 }

	  int nelem = strlen(header);
	  int charsize=sizeof(char);
	  MPI_Offset disp = nelem*charsize;

	  ostringstream filename;
	  filename << col->getSaveDirName() << "/" << col->getSimName() << "_"<< fieldtags[tagid] << "_" << cycle << ".vtk";
	  MPI_File_open(vct->getFieldComm(),filename.str().c_str(), MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);

	  if (vct->getCartesian_rank()==0){
		  MPI_File_write(fh, header, nelem, MPI_BYTE, &status);
	  }

      int error_code = MPI_File_set_view(fh, disp, EMf->getXYZeType(), EMf->getProcviewXYZ(), "native", MPI_INFO_NULL);
      if (error_code != MPI_SUCCESS) {
		char error_string[100];
		int length_of_error_string, error_class;

		MPI_Error_class(error_code, &error_class);
		MPI_Error_string(error_class, error_string, &length_of_error_string);
		dprintf("Error in MPI_File_set_view: %s\n", error_string);
	  }

      error_code = MPI_File_write_all(fh, fieldwritebuffer[0][0][0],(nxn-3)*(nyn-3)*(nzn-3),EMf->getXYZeType(), &status);
      if(error_code != MPI_SUCCESS){
	      int tcount=0;
	      MPI_Get_count(&status, EMf->getXYZeType(), &tcount);
          char error_string[100];
          int length_of_error_string, error_class;
          MPI_Error_class(error_code, &error_class);
          MPI_Error_string(error_class, error_string, &length_of_error_string);
          dprintf("Error in MPI_File_write_all: %s, wrote %d EMf->getXYZeType()\n", error_string,tcount);
      }
      MPI_File_close(&fh);
	}
}

void WriteMomentsVTK(Grid3DCU *grid, EMfields3D *EMf, CollectiveIO *col, VCtopology3D *vct, const string & outputTag ,int cycle, float*** momentswritebuffer){

	//All VTK output at grid cells excluding ghost cells
	const int nxn  =grid->getNXN(),nyn  = grid->getNYN(),nzn =grid->getNZN();
	const int dimX =col->getNxc() ,dimY = col->getNyc(), dimZ=col->getNzc();
	const double spaceX = dimX>1 ?col->getLx()/(dimX-1) :col->getLx();
	const double spaceY = dimY>1 ?col->getLy()/(dimY-1) :col->getLy();
	const double spaceZ = dimZ>1 ?col->getLz()/(dimZ-1) :col->getLz();
	const int    nPoints = dimX*dimY*dimZ;
	MPI_File     fh;
	MPI_Status   status;
	const string momentstags[]={"rho", "PXX", "PXY", "PXZ", "PYY", "PYZ", "PZZ"};
	const int    tagsize = size(momentstags);
	const string outputtag = col->getMomentsOutputTag();
	const int ns = col->getNs();

	for(int tagid=0; tagid<tagsize; tagid++){
		 if (outputtag.find(momentstags[tagid], 0) == string::npos) continue;

		 for(int si=0;si<ns;si++){
			 char  header[1024];
			 if (momentstags[tagid].compare("rho") == 0){
				for(int iz=0;iz<nzn-3;iz++)
				  for(int iy=0;iy<nyn-3;iy++)
					  for(int ix= 0;ix<nxn-3;ix++)
						  momentswritebuffer[iz][iy][ix] = (float)EMf->getRHOns(ix+1, iy+1, iz+1, si)*4*3.1415926535897;

				//Write VTK header
				sprintf(header, "# vtk DataFile Version 2.0\n"
								   "%s%d density from iPIC3D\n"
								   "BINARY\n"
								   "DATASET STRUCTURED_POINTS\n"
								   "DIMENSIONS %d %d %d\n"
								   "ORIGIN 0 0 0\n"
								   "SPACING %f %f %f\n"
								   "POINT_DATA %d \n"
								   "SCALARS %s float\n"
					"LOOKUP_TABLE default\n",(si%2==0)?"Electron":"Ion",si,dimX,dimY,dimZ, spaceX,spaceY,spaceZ, nPoints,(si%2==0)?"rhoe":"rhoi");
		 }else if(momentstags[tagid].compare("PXX") == 0){
				for(int iz=0;iz<nzn-3;iz++)
				  for(int iy=0;iy<nyn-3;iy++)
				    for(int ix= 0;ix<nxn-3;ix++)
				      momentswritebuffer[iz][iy][ix] = (float)EMf->getpXXsn(ix+1, iy+1, iz+1, si);

				//Write VTK header
				sprintf(header, "# vtk DataFile Version 2.0\n"
						"%s%d pressure PXX from iPIC3D\n"
								   "BINARY\n"
								   "DATASET STRUCTURED_POINTS\n"
								   "DIMENSIONS %d %d %d\n"
								   "ORIGIN 0 0 0\n"
								   "SPACING %f %f %f\n"
								   "POINT_DATA %d \n"
								   "SCALARS %s float\n"
					"LOOKUP_TABLE default\n",(si%2==0)?"Electron":"Ion",si,dimX,dimY,dimZ, spaceX,spaceY,spaceZ, nPoints,(si%2==0)?"PXXe":"PXXi");
		}else if(momentstags[tagid].compare("PXY") == 0){

			for(int iz=0;iz<nzn-3;iz++)
			  for(int iy=0;iy<nyn-3;iy++)
				  for(int ix= 0;ix<nxn-3;ix++)
					  momentswritebuffer[iz][iy][ix] = (float)EMf->getpXYsn(ix+1, iy+1, iz+1, si);

			//Write VTK header
			sprintf(header, "# vtk DataFile Version 2.0\n"
					"%s%d pressure PXY from iPIC3D\n"
							   "BINARY\n"
							   "DATASET STRUCTURED_POINTS\n"
							   "DIMENSIONS %d %d %d\n"
							   "ORIGIN 0 0 0\n"
							   "SPACING %f %f %f\n"
							   "POINT_DATA %d \n"
							   "SCALARS %s float\n"
				"LOOKUP_TABLE default\n",(si%2==0)?"Electron":"Ion",si,dimX,dimY,dimZ, spaceX,spaceY,spaceZ, nPoints,(si%2==0)?"PXYe":"PXYi");
		}else if(momentstags[tagid].compare("PXZ") == 0){

			for(int iz=0;iz<nzn-3;iz++)
			  for(int iy=0;iy<nyn-3;iy++)
				  for(int ix= 0;ix<nxn-3;ix++)
					  momentswritebuffer[iz][iy][ix] = (float)EMf->getpXZsn(ix+1, iy+1, iz+1, si);

			//Write VTK header
			sprintf(header, "# vtk DataFile Version 2.0\n"
							   "%s%d pressure PXZ from iPIC3D\n"
							   "BINARY\n"
							   "DATASET STRUCTURED_POINTS\n"
							   "DIMENSIONS %d %d %d\n"
							   "ORIGIN 0 0 0\n"
							   "SPACING %f %f %f\n"
							   "POINT_DATA %d \n"
							   "SCALARS %s float\n"
				"LOOKUP_TABLE default\n",(si%2==0)?"Electron":"Ion",si,dimX,dimY,dimZ, spaceX,spaceY,spaceZ, nPoints,(si%2==0)?"PXZe":"PXZi");
		}else if(momentstags[tagid].compare("PYY") == 0){

			for(int iz=0;iz<nzn-3;iz++)
			  for(int iy=0;iy<nyn-3;iy++)
				  for(int ix= 0;ix<nxn-3;ix++)
					  momentswritebuffer[iz][iy][ix] = (float)EMf->getpYYsn(ix+1, iy+1, iz+1, si);

			//Write VTK header
			sprintf(header, "# vtk DataFile Version 2.0\n"
							   "%s%d pressure PYY from iPIC3D\n"
							   "BINARY\n"
							   "DATASET STRUCTURED_POINTS\n"
							   "DIMENSIONS %d %d %d\n"
							   "ORIGIN 0 0 0\n"
							   "SPACING %f %f %f\n"
							   "POINT_DATA %d \n"
							   "SCALARS %s float\n"
				"LOOKUP_TABLE default\n",(si%2==0)?"Electron":"Ion",si,dimX,dimY,dimZ, spaceX,spaceY,spaceZ, nPoints,(si%2==0)?"PYYe":"PYYi");
		}else if(momentstags[tagid].compare("PYZ") == 0){

			for(int iz=0;iz<nzn-3;iz++)
			  for(int iy=0;iy<nyn-3;iy++)
				  for(int ix= 0;ix<nxn-3;ix++)
					  momentswritebuffer[iz][iy][ix] = (float)EMf->getpYZsn(ix+1, iy+1, iz+1, si);

			//Write VTK header
			sprintf(header, "# vtk DataFile Version 2.0\n"
							   "%s%d pressure PYZ from iPIC3D\n"
							   "BINARY\n"
							   "DATASET STRUCTURED_POINTS\n"
							   "DIMENSIONS %d %d %d\n"
							   "ORIGIN 0 0 0\n"
							   "SPACING %f %f %f\n"
							   "POINT_DATA %d \n"
							   "SCALARS %s float\n"
				"LOOKUP_TABLE default\n",(si%2==0)?"Electron":"Ion",si,dimX,dimY,dimZ, spaceX,spaceY,spaceZ, nPoints,(si%2==0)?"PYZe":"PYZi");
		}else if(momentstags[tagid].compare("PZZ") == 0){

			for(int iz=0;iz<nzn-3;iz++)
			  for(int iy=0;iy<nyn-3;iy++)
				  for(int ix= 0;ix<nxn-3;ix++)
					  momentswritebuffer[iz][iy][ix] = (float)EMf->getpZZsn(ix+1, iy+1, iz+1, si);

			//Write VTK header
			sprintf(header, "# vtk DataFile Version 2.0\n"
							   "%s%d pressure PZZ from iPIC3D\n"
							   "BINARY\n"
							   "DATASET STRUCTURED_POINTS\n"
							   "DIMENSIONS %d %d %d\n"
							   "ORIGIN 0 0 0\n"
							   "SPACING %f %f %f\n"
							   "POINT_DATA %d \n"
							   "SCALARS %s float\n"
				"LOOKUP_TABLE default\n",(si%2==0)?"Electron":"Ion",si,dimX,dimY,dimZ, spaceX,spaceY,spaceZ, nPoints,(si%2==0)?"PZZe":"PZZi");
		}

		 if(EMf->isLittleEndian()){
			 for(int iz=0;iz<nzn-3;iz++)
				  for(int iy=0;iy<nyn-3;iy++)
					  for(int ix= 0;ix<nxn-3;ix++){
						  ByteSwap((unsigned char*) &momentswritebuffer[iz][iy][ix],4);
					  }
		 }

		  int nelem = strlen(header);
		  int charsize=sizeof(char);
		  MPI_Offset disp = nelem*charsize;

		  ostringstream filename;
		  filename << col->getSaveDirName() << "/" << col->getSimName() << "_" << momentstags[tagid] << ((si%2==0)?"e":"i")<< si  << "_" << cycle << ".vtk";
		  MPI_File_open(vct->getFieldComm(),filename.str().c_str(), MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);

		  if (vct->getCartesian_rank()==0){
			  MPI_File_write(fh, header, nelem, MPI_BYTE, &status);
		  }

	      int error_code = MPI_File_set_view(fh, disp, MPI_FLOAT, EMf->getProcview(), "native", MPI_INFO_NULL);
	      if (error_code != MPI_SUCCESS) {
			char error_string[100];
			int length_of_error_string, error_class;
			MPI_Error_class(error_code, &error_class);
			MPI_Error_string(error_class, error_string, &length_of_error_string);
			dprintf("Error in MPI_File_set_view: %s\n", error_string);
		  }

	      error_code = MPI_File_write_all(fh, momentswritebuffer[0][0],(nxn-3)*(nyn-3)*(nzn-3),MPI_FLOAT, &status);
	      if(error_code != MPI_SUCCESS){
		      int tcount=0;
		      MPI_Get_count(&status, MPI_FLOAT, &tcount);
	          char error_string[100];
	          int length_of_error_string, error_class;
	          MPI_Error_class(error_code, &error_class);
	          MPI_Error_string(error_class, error_string, &length_of_error_string);
	          dprintf("Error in MPI_File_write_all: %s, wrote %d MPI_FLOAT\n", error_string,tcount);
	      }
	      MPI_File_close(&fh);
		 }//END OF SPECIES
	}//END OF TAGS
}


int WriteFieldsVTKNonblk(Grid3DCU *grid, EMfields3D *EMf, CollectiveIO *col, VCtopology3D *vct,int cycle,
		                float**** fieldwritebuffer,MPI_Request requestArr[4],MPI_File fhArr[4])
{

	//All VTK output at grid cells excluding ghost cells
	const int nxn  =grid->getNXN(),nyn  = grid->getNYN(),nzn =grid->getNZN();
	const int dimX =col->getNxc() ,dimY = col->getNyc(), dimZ=col->getNzc();
	const double spaceX = dimX>1 ?col->getLx()/(dimX-1) :col->getLx();
	const double spaceY = dimY>1 ?col->getLy()/(dimY-1) :col->getLy();
	const double spaceZ = dimZ>1 ?col->getLz()/(dimZ-1) :col->getLz();
	const int    nPoints = dimX*dimY*dimZ;
	const string fieldtags[]={"B", "E", "Je", "Ji"};
	const int fieldtagsize = size(fieldtags);
	const string fieldoutputtag = col->getFieldOutputTag();
	int counter=0,error_code;

	for(int tagid=0; tagid<fieldtagsize; tagid++){
		 if (fieldoutputtag.find(fieldtags[tagid], 0) == string::npos) continue;

		 fieldwritebuffer = &(fieldwritebuffer[counter]);
		 char  header[1024];
		 if (fieldtags[tagid].compare("B") == 0){
			 for(int iz=0;iz<nzn-3;iz++)
				  for(int iy=0;iy<nyn-3;iy++)
					  for(int ix= 0;ix<nxn-3;ix++){
						  fieldwritebuffer[iz][iy][ix][0] =  (float)EMf->getBxTot(ix+1, iy+1, iz+1);
						  fieldwritebuffer[iz][iy][ix][1] =  (float)EMf->getByTot(ix+1, iy+1, iz+1);
						  fieldwritebuffer[iz][iy][ix][2] =  (float)EMf->getBzTot(ix+1, iy+1, iz+1);
					  }

			 //Write VTK header
			 sprintf(header, "# vtk DataFile Version 2.0\n"
						   "Magnetic Field from iPIC3D\n"
						   "BINARY\n"
						   "DATASET STRUCTURED_POINTS\n"
						   "DIMENSIONS %d %d %d\n"
						   "ORIGIN 0 0 0\n"
						   "SPACING %f %f %f\n"
						   "POINT_DATA %d\n"
						   "VECTORS B float\n", dimX,dimY,dimZ, spaceX,spaceY,spaceZ, nPoints);

		 }else if (fieldtags[tagid].compare("E") == 0){
			 for(int iz=0;iz<nzn-3;iz++)
				  for(int iy=0;iy<nyn-3;iy++)
					  for(int ix= 0;ix<nxn-3;ix++){
						  fieldwritebuffer[iz][iy][ix][0] = (float)EMf->getEx(ix+1, iy+1, iz+1);
						  fieldwritebuffer[iz][iy][ix][1] = (float)EMf->getEy(ix+1, iy+1, iz+1);
						  fieldwritebuffer[iz][iy][ix][2] = (float)EMf->getEz(ix+1, iy+1, iz+1);
					  }

			 //Write VTK header
			 sprintf(header, "# vtk DataFile Version 2.0\n"
						   "Electric Field from iPIC3D\n"
						   "BINARY\n"
						   "DATASET STRUCTURED_POINTS\n"
						   "DIMENSIONS %d %d %d\n"
						   "ORIGIN 0 0 0\n"
						   "SPACING %f %f %f\n"
						   "POINT_DATA %d \n"
						   "VECTORS E float\n", dimX,dimY,dimZ, spaceX,spaceY,spaceZ, nPoints);

		 }else if (fieldtags[tagid].compare("Je") == 0){
			 for(int iz=0;iz<nzn-3;iz++)
				  for(int iy=0;iy<nyn-3;iy++)
					  for(int ix= 0;ix<nxn-3;ix++){
						  fieldwritebuffer[iz][iy][ix][0] = (float)EMf->getJxs(ix+1, iy+1, iz+1, 0);
						  fieldwritebuffer[iz][iy][ix][1] = (float)EMf->getJys(ix+1, iy+1, iz+1, 0);
						  fieldwritebuffer[iz][iy][ix][2] = (float)EMf->getJzs(ix+1, iy+1, iz+1, 0);
					  }

			 //Write VTK header
			 sprintf(header, "# vtk DataFile Version 2.0\n"
						   "Electron current from iPIC3D\n"
						   "BINARY\n"
						   "DATASET STRUCTURED_POINTS\n"
						   "DIMENSIONS %d %d %d\n"
						   "ORIGIN 0 0 0\n"
						   "SPACING %f %f %f\n"
						   "POINT_DATA %d \n"
						   "VECTORS Je float\n", dimX,dimY,dimZ, spaceX,spaceY,spaceZ, nPoints);

		 }else if (fieldtags[tagid].compare("Ji") == 0){
			 for(int iz=0;iz<nzn-3;iz++)
				  for(int iy=0;iy<nyn-3;iy++)
					  for(int ix= 0;ix<nxn-3;ix++){
						  fieldwritebuffer[iz][iy][ix][0] = (float)EMf->getJxs(ix+1, iy+1, iz+1, 1);
						  fieldwritebuffer[iz][iy][ix][1] = (float)EMf->getJys(ix+1, iy+1, iz+1, 1);
						  fieldwritebuffer[iz][iy][ix][2] = (float)EMf->getJzs(ix+1, iy+1, iz+1, 1);
					  }

			 //Write VTK header
			 sprintf(header, "# vtk DataFile Version 2.0\n"
						   "Ion current from iPIC3D\n"
						   "BINARY\n"
						   "DATASET STRUCTURED_POINTS\n"
						   "DIMENSIONS %d %d %d\n"
						   "ORIGIN 0 0 0\n"
						   "SPACING %f %f %f\n"
						   "POINT_DATA %d \n"
						   "VECTORS Ji float\n", dimX,dimY,dimZ, spaceX,spaceY,spaceZ, nPoints);
		 }

		 if(EMf->isLittleEndian()){
			 for(int iz=0;iz<nzn-3;iz++)
				  for(int iy=0;iy<nyn-3;iy++)
					  for(int ix= 0;ix<nxn-3;ix++){
						  ByteSwap((unsigned char*) &fieldwritebuffer[iz][iy][ix][0],4);
						  ByteSwap((unsigned char*) &fieldwritebuffer[iz][iy][ix][1],4);
						  ByteSwap((unsigned char*) &fieldwritebuffer[iz][iy][ix][2],4);
					  }
		 }

		  int nelem = strlen(header);
		  int charsize=sizeof(char);
		  MPI_Offset disp = nelem*charsize;

		  ostringstream filename;
		  filename << col->getSaveDirName() << "/" << col->getSimName() << "_"<< fieldtags[tagid] << "_" << cycle << ".vtk";
		  MPI_File_open(vct->getFieldComm(),filename.str().c_str(), MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &(fhArr[counter]));

		  if (vct->getCartesian_rank()==0){
			  MPI_Status   status;
			  MPI_File_write(fhArr[counter], header, nelem, MPI_BYTE, &status);
		  }

		  error_code = MPI_File_set_view(fhArr[counter], disp, EMf->getXYZeType(), EMf->getProcviewXYZ(), "native", MPI_INFO_NULL);
	      if(error_code!= MPI_SUCCESS){
              char error_string[100];
              int length_of_error_string, error_class;
              MPI_Error_class(error_code, &error_class);
              MPI_Error_string(error_class, error_string, &length_of_error_string);
              dprintf("Error in MPI_File_set_view: %s\n", error_string);
	      }

	      error_code = MPI_File_write_all_begin(fhArr[counter],fieldwritebuffer[0][0][0],(nxn-3)*(nyn-3)*(nzn-3),EMf->getXYZeType());
	      //error_code = MPI_File_iwrite(fhArr[counter], fieldwritebuffer[0][0][0],(nxn-3)*(nyn-3)*(nzn-3),EMf->getXYZeType(), &(requestArr[counter]));
	      if(error_code!= MPI_SUCCESS){
              char error_string[100];
              int length_of_error_string, error_class;
              MPI_Error_class(error_code, &error_class);
              MPI_Error_string(error_class, error_string, &length_of_error_string);
              dprintf("Error in MPI_File_iwrite: %s", error_string);
	      }
	      counter ++;
		}
	return counter;
}


int  WriteMomentsVTKNonblk(Grid3DCU *grid, EMfields3D *EMf, CollectiveIO *col, VCtopology3D *vct,int cycle,
		float*** momentswritebuffer,MPI_Request requestArr[14],MPI_File fhArr[14])
{
	//All VTK output at grid cells excluding ghost cells
	const int nxn  =grid->getNXN(),nyn  = grid->getNYN(),nzn =grid->getNZN();
	const int dimX =col->getNxc() ,dimY = col->getNyc(), dimZ=col->getNzc();
	const double spaceX = dimX>1 ?col->getLx()/(dimX-1) :col->getLx();
	const double spaceY = dimY>1 ?col->getLy()/(dimY-1) :col->getLy();
	const double spaceZ = dimZ>1 ?col->getLz()/(dimZ-1) :col->getLz();
	const int    nPoints = dimX*dimY*dimZ;
	const string momentstags[]={"rho", "PXX", "PXY", "PXZ", "PYY", "PYZ", "PZZ"};
	const int    tagsize = size(momentstags);
	const string outputtag = col->getMomentsOutputTag();
	int counter=0,err;

	for(int tagid=0; tagid<tagsize; tagid++){
		 if (outputtag.find(momentstags[tagid], 0) == string::npos) continue;

		 for(int si=0;si<=1;si++){
			 momentswritebuffer = &(momentswritebuffer[counter]);
			 char  header[1024];
			 if (momentstags[tagid].compare("rho") == 0){

				for(int iz=0;iz<nzn-3;iz++)
				  for(int iy=0;iy<nyn-3;iy++)
					  for(int ix= 0;ix<nxn-3;ix++)
						  momentswritebuffer[iz][iy][ix] = (float)EMf->getRHOns(ix+1, iy+1, iz+1, si)*4*3.1415926535897;

				//Write VTK header
				sprintf(header, "# vtk DataFile Version 2.0\n"
								   "%s density from iPIC3D\n"
								   "BINARY\n"
								   "DATASET STRUCTURED_POINTS\n"
								   "DIMENSIONS %d %d %d\n"
								   "ORIGIN 0 0 0\n"
								   "SPACING %f %f %f\n"
								   "POINT_DATA %d \n"
								   "SCALARS %s float\n"
								   "LOOKUP_TABLE default\n",(si%2==0)?"Electron":"Ion",dimX,dimY,dimZ, spaceX,spaceY,spaceZ, nPoints,(si%2==0)?"rhoe":"rhoi");
			 }else if(momentstags[tagid].compare("PXX") == 0){

				for(int iz=0;iz<nzn-3;iz++)
				  for(int iy=0;iy<nyn-3;iy++)
					  for(int ix= 0;ix<nxn-3;ix++)
						  momentswritebuffer[iz][iy][ix] = (float)EMf->getpXXsn(ix+1, iy+1, iz+1, si);

				//Write VTK header
				sprintf(header, "# vtk DataFile Version 2.0\n"
								   "%s pressure PXX from iPIC3D\n"
								   "BINARY\n"
								   "DATASET STRUCTURED_POINTS\n"
								   "DIMENSIONS %d %d %d\n"
								   "ORIGIN 0 0 0\n"
								   "SPACING %f %f %f\n"
								   "POINT_DATA %d \n"
								   "SCALARS %s float\n"
								   "LOOKUP_TABLE default\n",(si%2==0)?"Electron":"Ion",dimX,dimY,dimZ, spaceX,spaceY,spaceZ, nPoints,(si%2==0)?"PXXe":"PXXi");
			 }else if(momentstags[tagid].compare("PXY") == 0){

				for(int iz=0;iz<nzn-3;iz++)
				  for(int iy=0;iy<nyn-3;iy++)
					  for(int ix= 0;ix<nxn-3;ix++)
						  momentswritebuffer[iz][iy][ix] = (float)EMf->getpXYsn(ix+1, iy+1, iz+1, si);

				//Write VTK header
				sprintf(header, "# vtk DataFile Version 2.0\n"
								   "%s pressure PXY from iPIC3D\n"
								   "BINARY\n"
								   "DATASET STRUCTURED_POINTS\n"
								   "DIMENSIONS %d %d %d\n"
								   "ORIGIN 0 0 0\n"
								   "SPACING %f %f %f\n"
								   "POINT_DATA %d \n"
								   "SCALARS %s float\n"
								   "LOOKUP_TABLE default\n",(si%2==0)?"Electron":"Ion",dimX,dimY,dimZ, spaceX,spaceY,spaceZ, nPoints,(si%2==0)?"PXYe":"PXYi");
			}else if(momentstags[tagid].compare("PXZ") == 0){

				for(int iz=0;iz<nzn-3;iz++)
				  for(int iy=0;iy<nyn-3;iy++)
					  for(int ix= 0;ix<nxn-3;ix++)
						  momentswritebuffer[iz][iy][ix] = (float)EMf->getpXZsn(ix+1, iy+1, iz+1, si);

				//Write VTK header
				sprintf(header, "# vtk DataFile Version 2.0\n"
								   "%s pressure PXZ from iPIC3D\n"
								   "BINARY\n"
								   "DATASET STRUCTURED_POINTS\n"
								   "DIMENSIONS %d %d %d\n"
								   "ORIGIN 0 0 0\n"
								   "SPACING %f %f %f\n"
								   "POINT_DATA %d \n"
								   "SCALARS %s float\n"
								   "LOOKUP_TABLE default\n",(si%2==0)?"Electron":"Ion",dimX,dimY,dimZ, spaceX,spaceY,spaceZ, nPoints,(si%2==0)?"PXZe":"PXZi");
			}else if(momentstags[tagid].compare("PYY") == 0){

				for(int iz=0;iz<nzn-3;iz++)
				  for(int iy=0;iy<nyn-3;iy++)
					  for(int ix= 0;ix<nxn-3;ix++)
						  momentswritebuffer[iz][iy][ix] = (float)EMf->getpYYsn(ix+1, iy+1, iz+1, si);

				//Write VTK header
				sprintf(header, "# vtk DataFile Version 2.0\n"
								   "%s pressure PYY from iPIC3D\n"
								   "BINARY\n"
								   "DATASET STRUCTURED_POINTS\n"
								   "DIMENSIONS %d %d %d\n"
								   "ORIGIN 0 0 0\n"
								   "SPACING %f %f %f\n"
								   "POINT_DATA %d \n"
								   "SCALARS %s float\n"
								   "LOOKUP_TABLE default\n",(si%2==0)?"Electron":"Ion",dimX,dimY,dimZ, spaceX,spaceY,spaceZ, nPoints,(si%2==0)?"PYYe":"PYYi");
			}else if(momentstags[tagid].compare("PYZ") == 0){

				for(int iz=0;iz<nzn-3;iz++)
				  for(int iy=0;iy<nyn-3;iy++)
					  for(int ix= 0;ix<nxn-3;ix++)
						  momentswritebuffer[iz][iy][ix] = (float)EMf->getpYZsn(ix+1, iy+1, iz+1, si);

				//Write VTK header
				sprintf(header, "# vtk DataFile Version 2.0\n"
								   "%s pressure PYZ from iPIC3D\n"
								   "BINARY\n"
								   "DATASET STRUCTURED_POINTS\n"
								   "DIMENSIONS %d %d %d\n"
								   "ORIGIN 0 0 0\n"
								   "SPACING %f %f %f\n"
								   "POINT_DATA %d \n"
								   "SCALARS %s float\n"
								   "LOOKUP_TABLE default\n",(si%2==0)?"Electron":"Ion",dimX,dimY,dimZ, spaceX,spaceY,spaceZ, nPoints,(si%2==0)?"PYZe":"PYZi");
			}else if(momentstags[tagid].compare("PZZ") == 0){

				for(int iz=0;iz<nzn-3;iz++)
				  for(int iy=0;iy<nyn-3;iy++)
					  for(int ix= 0;ix<nxn-3;ix++)
						  momentswritebuffer[iz][iy][ix] = (float)EMf->getpZZsn(ix+1, iy+1, iz+1, si);

				//Write VTK header
				sprintf(header, "# vtk DataFile Version 2.0\n"
								   "%s pressure PZZ from iPIC3D\n"
								   "BINARY\n"
								   "DATASET STRUCTURED_POINTS\n"
								   "DIMENSIONS %d %d %d\n"
								   "ORIGIN 0 0 0\n"
								   "SPACING %f %f %f\n"
								   "POINT_DATA %d \n"
								   "SCALARS %s float\n"
								   "LOOKUP_TABLE default\n",(si%2==0)?"Electron":"Ion",dimX,dimY,dimZ, spaceX,spaceY,spaceZ, nPoints,(si%2==0)?"PZZe":"PZZi");
			}

			 if(EMf->isLittleEndian()){
				 for(int iz=0;iz<nzn-3;iz++)
					  for(int iy=0;iy<nyn-3;iy++)
						  for(int ix= 0;ix<nxn-3;ix++){
							  ByteSwap((unsigned char*) &momentswritebuffer[iz][iy][ix],4);
						  }
			 }

		  int nelem = strlen(header);
		  int charsize=sizeof(char);
		  MPI_Offset disp = nelem*charsize;

		  ostringstream filename;
		  filename << col->getSaveDirName() << "/" << col->getSimName() << "_" << momentstags[tagid] << ((si%2==0)?"e":"i") << "_" << cycle << ".vtk";
		  MPI_File_open(vct->getFieldComm(),filename.str().c_str(), MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &fhArr[counter]);

		  if (vct->getCartesian_rank()==0){
			  MPI_Status   status;
			  MPI_File_write(fhArr[counter], header, nelem, MPI_BYTE, &status);
		  }

	      int error_code = MPI_File_set_view(fhArr[counter], disp, MPI_FLOAT, EMf->getProcview(), "native", MPI_INFO_NULL);
	      if (error_code != MPI_SUCCESS) {
			char error_string[100];
			int length_of_error_string, error_class;
			MPI_Error_class(error_code, &error_class);
			MPI_Error_string(error_class, error_string, &length_of_error_string);
			dprintf("Error in MPI_File_set_view: %s\n", error_string);
		  }

	      error_code = MPI_File_write_all_begin(fhArr[counter],momentswritebuffer[0][0],(nxn-3)*(nyn-3)*(nzn-3),MPI_FLOAT);
	      //error_code = MPI_File_iwrite(fhArr[counter], momentswritebuffer[0][0],(nxn-3)*(nyn-3)*(nzn-3),MPI_FLOAT, &(requestArr[counter]));
	      if(error_code != MPI_SUCCESS){
	          char error_string[100];
	          int length_of_error_string, error_class;
	          MPI_Error_class(error_code, &error_class);
	          MPI_Error_string(error_class, error_string, &length_of_error_string);
	          dprintf("Error in MPI_File_iwrite: %s \n", error_string);
	      }
	      counter ++;
		 }//END OF SPECIES
	}
	return counter;
}


void ByteSwap(unsigned char * b, int n)
{
   int i = 0;
   int j = n-1;
   while (i<j)
   {
      std::swap(b[i], b[j]);
      i++, j--;
   }
}