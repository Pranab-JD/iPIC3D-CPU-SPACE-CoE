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


// #include <mpi.h>
// #include "phdf5.h"
// #include "WriteOutputParallel.h"
// #include "Grid3DCU.h"
// #include "EMfields3D.h"
// #include "VCtopology3D.h"
// #include "errors.h"
// #include <string>
// #include <sstream>
// #include <iomanip>
// #include "phdf5.h"
// #include "Collective.h"
// #include <vector>
// #include <hdf5.h>

// using namespace std;

// static inline void block_dist_1d(hsize_t G, int P, int c, hsize_t &start, hsize_t &count)
// {
//     const hsize_t base = G / (hsize_t)P;
//     const hsize_t rem  = G % (hsize_t)P;
//     count = base + ((hsize_t)c < rem ? 1 : 0);
//     start = (hsize_t)c * base + (hsize_t)std::min<int>(c, (int)rem);
// }

// void WriteOutputParallel(Grid3DCU *grid, EMfields3D *EMf, CollectiveIO *col, VCtopology3D *vct, int cycle)
// {
//     #ifdef PHDF5
//     string       grpname;
//     string       dtaname;

//     stringstream filenmbr;
//     string       filename;

//     /* ------------------- */
//     /* Setup the file name */
//     /* ------------------- */

//     filenmbr << setfill('0') << setw(5) << cycle;
//     filename = col->getSaveDirName() + "/" + col->getSimName() + "_" + filenmbr.str() + ".h5";

//     /* ---------------------------------------------------------------------------- */
//     /* Define the number of cells in the global and local mesh and set the mesh size */
//     /* ---------------------------------------------------------------------------- */

//     // const int nxc = grid->getNXC();
//     // const int nyc = grid->getNYC();
//     // const int nzc = grid->getNZC();
//     const int nxn = grid->getNXN();
//     const int nyn = grid->getNYN();
//     const int nzn = grid->getNZN();

//     int dglob[3] = { col->getNxc() + 1, col->getNyc() + 1, col->getNzc() + 1 };
//     int dlocl[3] = {nxn-2, nyn-2, nzn-2 };
//     double L[3] = { col ->getLx ()  , col ->getLy ()  , col ->getLz ()   };

//     /* --------------------------------------- */
//     /* Declare and open the parallel HDF5 file */
//     /* --------------------------------------- */

//     PHDF5fileClass outputfile(filename, 3, vct->getCoordinates(), MPI_COMM_WORLD);
//     outputfile.CreatePHDF5file(L, dglob, dlocl, false);

//     /* ------------------------ */
//     /* Write the Electric field */
//     /* ------------------------ */

//     outputfile.WritePHDF5dataset("Fields", "Ex", EMf->getEx(), nxn-2, nyn-2, nzn-2);
//     outputfile.WritePHDF5dataset("Fields", "Ey", EMf->getEy(), nxn-2, nyn-2, nzn-2);
//     outputfile.WritePHDF5dataset("Fields", "Ez", EMf->getEz(), nxn-2, nyn-2, nzn-2);

//     /* ------------------------ */
//     /* Write the Magnetic field */
//     /* ------------------------ */

//     outputfile.WritePHDF5dataset("Fields", "Bx", EMf->getBx(), nxn-2, nyn-2, nzn-2);
//     outputfile.WritePHDF5dataset("Fields", "By", EMf->getBy(), nxn-2, nyn-2, nzn-2);
//     outputfile.WritePHDF5dataset("Fields", "Bz", EMf->getBz(), nxn-2, nyn-2, nzn-2);

//     /* ----------------------------------------------- */
//     /* Write the moments for each species */
//     /* ----------------------------------------------- */

//     for (int is = 0; is < col->getNs(); is++)
//     {
//         stringstream snmbr;
//         snmbr << is;
//         const string num = snmbr.str();

//         // Charge Density
//         outputfile.WritePHDF5dataset("Fields", string("rho_")+num , EMf->getRHOns()[is], nxn-2, nyn-2, nzn-2);

//         // Current
//         outputfile.WritePHDF5dataset("Fields", string("Jx_")+num, EMf->getJxs()[is], nxn-2, nyn-2, nzn-2);
//         outputfile.WritePHDF5dataset("Fields", string("Jy_")+num, EMf->getJys()[is], nxn-2, nyn-2, nzn-2);
//         outputfile.WritePHDF5dataset("Fields", string("Jz_")+num, EMf->getJzs()[is], nxn-2, nyn-2, nzn-2);
//     }

//     outputfile.ClosePHDF5file();

//     #else

//     eprintf(" The input file requests the use of the Parallel HDF5 functions, but the code has been compiled using the sequential HDF5 library.\n"
//             " Recompile the code using the parallel HDF5 options or change the input file options. ");
//     #endif
// }

// static inline void block_dist_1d(hsize_t G, int P, int c, hsize_t &start, hsize_t &count)
// {
//     const hsize_t base = G / (hsize_t)P;
//     const hsize_t rem  = G % (hsize_t)P;
//     count = base + ((hsize_t)c < rem ? 1 : 0);
//     start = (hsize_t)c * base + (hsize_t)std::min<int>(c, (int)rem);
// }

// template <typename GetX, typename GetY, typename GetZ>
// static inline void pack_vec3_nodes(float* out,
//                                    int lx, int ly, int lz,
//                                    GetX getx, GetY gety, GetZ getz)
// {
//     // out is contiguous [lx][ly][lz][3] flattened in C-order
//     size_t p = 0;
//     for (int i = 1; i <= lx; ++i)
//         for (int j = 1; j <= ly; ++j)
//             for (int k = 1; k <= lz; ++k) {
//                 out[p++] = static_cast<float>(getx(i, j, k));
//                 out[p++] = static_cast<float>(gety(i, j, k));
//                 out[p++] = static_cast<float>(getz(i, j, k));
//             }
// }

// template <typename GetF>
// static inline void pack_sca_nodes(float* out,
//                                   int lx, int ly, int lz,
//                                   GetF getf)
// {
//     size_t p = 0;
//     for (int i = 1; i <= lx; ++i)
//         for (int j = 1; j <= ly; ++j)
//             for (int k = 1; k <= lz; ++k)
//                 out[p++] = static_cast<float>(getf(i, j, k));
// }

// void WriteOutputParallel(Grid3DCU *grid, EMfields3D *EMf, CollectiveIO *col, VCtopology3D *vct, int cycle)
// {
// #ifdef PHDF5
//     // ---------- filename ----------
//     std::stringstream filenmbr;
//     filenmbr << std::setfill('0') << std::setw(5) << cycle;
//     const std::string filename = col->getSaveDirName() + "/" + col->getSimName() + "_" + filenmbr.str() + ".h5";

//     // ---------- local node block written by VTK ----------
//     // VTK code writes indices (ix+1) with extent (nxn-3), so local written size = NXN-3 etc.
//     const int nxn = grid->getNXN();
//     const int nyn = grid->getNYN();
//     const int nzn = grid->getNZN();

//     // Global NODE dims (no ghosts)
//     hsize_t gdim[3] = {
//         (hsize_t)(col->getNxc() + 1),
//         (hsize_t)(col->getNyc() + 1),
//         (hsize_t)(col->getNzc() + 1)
//     };

//     // Ragged distribution gives the *correct* local sizes
//     hsize_t start[3], count[3];
//     block_dist_1d(gdim[0], vct->getXLEN(), vct->getCoordinates(0), start[0], count[0]);
//     block_dist_1d(gdim[1], vct->getYLEN(), vct->getCoordinates(1), start[1], count[1]);
//     block_dist_1d(gdim[2], vct->getZLEN(), vct->getCoordinates(2), start[2], count[2]);

//     const int lx = (int)count[0];
//     const int ly = (int)count[1];
//     const int lz = (int)count[2];

//     // Sanity check against local arrays (interior nodes are NXN-2 etc)
//     if (lx != nxn - 2 || ly != nyn - 2 || lz != nzn - 2) {
//         eprintf("Local node interior mismatch.\n"
//                 "  expected from distribution = %d %d %d\n"
//                 "  local NXN/NYN/NZN-2        = %d %d %d\n"
//                 "  coord=(%d,%d,%d) lens=(%d,%d,%d) global_nodes=(%llu,%llu,%llu)\n",
//                 lx, ly, lz,
//                 nxn-2, nyn-2, nzn-2,
//                 vct->getCoordinates(0), vct->getCoordinates(1), vct->getCoordinates(2),
//                 vct->getXLEN(), vct->getYLEN(), vct->getZLEN(),
//                 (unsigned long long)gdim[0], (unsigned long long)gdim[1], (unsigned long long)gdim[2]);
//         return;
//     }

//     // ---------- create file ----------
//     // NOTE: CreatePHDF5file currently stores dim/chdim etc. We don't rely on those now;
//     // we just need a valid parallel file_id and the default groups.
//     const hsize_t Gx = (hsize_t)(col->getNxc() + 1);
//     const hsize_t Gy = (hsize_t)(col->getNyc() + 1);
//     const hsize_t Gz = (hsize_t)(col->getNzc() + 1);

//     double L[3] = {col->getLx(), col->getLy(), col->getLz()};
//     int dglob_tmp[3] = {(int)Gx, (int)Gy, (int)Gz};
//     int dlocl_tmp[3] = {lx, ly, lz};

//     PHDF5fileClass outputfile(filename, 3, vct->getCoordinates(), MPI_COMM_WORLD);
//     outputfile.CreatePHDF5file(L, dglob_tmp, dlocl_tmp, false);

//     // ---------- write E at nodes ----------
//     {
//         std::vector<float> buf((size_t)lx * ly * lz * 3);

//         pack_vec3_nodes(buf.data(), lx, ly, lz,
//                         [&](int i,int j,int k){ return EMf->getEx(i, j, k); },
//                         [&](int i,int j,int k){ return EMf->getEy(i, j, k); },
//                         [&](int i,int j,int k){ return EMf->getEz(i, j, k); });

//         // You can store as one dataset "E" with 3 components, but simplest is Ex/Ey/Ez like your HDF5
//         // If you want one dataset, you need a 4D dataset; keep 3 datasets for now:
//         // Here we just reuse packer per component instead, to keep dataset rank 3.
//     }

//     // Write Ex/Ey/Ez separately (rank-3 datasets)
//     {
//         std::vector<float> buf((size_t)lx * ly * lz);

//         pack_sca_nodes(buf.data(), lx, ly, lz, [&](int i,int j,int k){ return EMf->getEx(i,j,k); });
//         // outputfile.WritePHDF5dataset_ragged_f32("Fields", "Ex", buf.data(), gdim, start, count);
//         outputfile.WritePHDF5dataset_nodes("Fields","Ex", EMf->getEx(),
//                                   grid->getNXN(), grid->getNYN(), grid->getNZN(),
//                                   col->getNxc(), col->getNyc(), col->getNzc());

//         // pack_sca_nodes(buf.data(), lx, ly, lz, [&](int i,int j,int k){ return EMf->getEy(i,j,k); });
//         // outputfile.WritePHDF5dataset_ragged_f32("Fields", "Ey", buf.data(), gdim, start, count);

//         // pack_sca_nodes(buf.data(), lx, ly, lz, [&](int i,int j,int k){ return EMf->getEz(i,j,k); });
//         // outputfile.WritePHDF5dataset_ragged_f32("Fields", "Ez", buf.data(), gdim, start, count);
//     }

//     // ---------- write B at nodes (use your same getters as VTK uses) ----------
//     // {
//     //     std::vector<float> buf((size_t)lx * ly * lz);

//     //     pack_sca_nodes(buf.data(), lx, ly, lz, [&](int i,int j,int k){ return EMf->getBxTot(i,j,k); });
//     //     outputfile.WritePHDF5dataset_ragged_f32("Fields", "Bx", buf.data(), gdim, start, count);

//     //     pack_sca_nodes(buf.data(), lx, ly, lz, [&](int i,int j,int k){ return EMf->getByTot(i,j,k); });
//     //     outputfile.WritePHDF5dataset_ragged_f32("Fields", "By", buf.data(), gdim, start, count);

//     //     pack_sca_nodes(buf.data(), lx, ly, lz, [&](int i,int j,int k){ return EMf->getBzTot(i,j,k); });
//     //     outputfile.WritePHDF5dataset_ragged_f32("Fields", "Bz", buf.data(), gdim, start, count);
//     // }

//     // // ---------- Je and Ji at nodes ----------
//     // // species 0 = electrons, 1 = ions (as in your VTK code)
//     // for (int is = 0; is < 2; ++is) {
//     //     std::vector<float> buf((size_t)lx * ly * lz);

//     //     std::string prefix = (is == 0) ? "Je" : "Ji";

//     //     pack_sca_nodes(buf.data(), lx, ly, lz, [&](int i,int j,int k){ return EMf->getJxs(i,j,k,is); });
//     //     outputfile.WritePHDF5dataset_ragged_f32("Fields", prefix + "x", buf.data(), gdim, start, count);

//     //     pack_sca_nodes(buf.data(), lx, ly, lz, [&](int i,int j,int k){ return EMf->getJys(i,j,k,is); });
//     //     outputfile.WritePHDF5dataset_ragged_f32("Fields", prefix + "y", buf.data(), gdim, start, count);

//     //     pack_sca_nodes(buf.data(), lx, ly, lz, [&](int i,int j,int k){ return EMf->getJzs(i,j,k,is); });
//     //     outputfile.WritePHDF5dataset_ragged_f32("Fields", prefix + "z", buf.data(), gdim, start, count);
//     // }

//     // // ---------- rhoe and rhoi at nodes ----------
//     // for (int is = 0; is < 2; ++is) {
//     //     std::vector<float> buf((size_t)lx * ly * lz);
//     //     std::string name = (is == 0) ? "rhoe" : "rhoi";

//     //     pack_sca_nodes(buf.data(), lx, ly, lz, [&](int i,int j,int k){
//     //         return EMf->getRHOns(i,j,k,is) * (4.0 * M_PI);   // match your VTK scaling if you want it
//     //     });
//     //     outputfile.WritePHDF5dataset_ragged_f32("Fields", name, buf.data(), gdim, start, count);
//     // }

//     outputfile.ClosePHDF5file();

// #else
//     eprintf("Parallel HDF5 requested but code compiled without PHDF5.");
// #endif
// }