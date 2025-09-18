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
#include "ipichdf5.h"
#include "EMfields3D.h"
#include "Collective.h"
#include "Basic.h"
#include "Com3DNonblk.h"
#include "VCtopology3D.h"
#include "Grid3DCU.h"
#include "CG.h"
#include "GMRES.h"
#include "Particles3Dcomm.h"
#include "Moments.h"
#include "Parameters.h"
#include "ompdefs.h"
#include "debug.h"
#include "string.h"
#include "mic_particles.h"
#include "TimeTasks.h"
#include "ipicmath.h" // for roundup_to_multiple
#include "Alloc.h"
#include "asserts.h"
#include <iomanip>
#include <iostream>
#include <fstream>
#include "../LeXInt_Timer.hpp"
#include <filesystem>
#include <cstring>

#ifndef NO_HDF5
#endif

using std::cout;
using std::endl;
using namespace iPic3D;

#define NE_MASS 14      //* Used in mass matrix

/*! constructor */
//
// We rely on the following rule from the C++ standard, section 12.6.2.5:
//
//   nonstatic data members shall be initialized in the order
//   they were declared in the class definition
//
// in particular, nxc, nyc, nzc and nxn, nyn, nzn are assumed
// initialized when subsequently used.

EMfields3D::EMfields3D(Collective * col, Grid * grid, VirtualTopology3D *vct) : 
    _col(*col),
    _grid(*grid),
    _vct(*vct),
    nxc(grid->getNXC()),
    nxn(grid->getNXN()),
    nyc(grid->getNYC()),
    nyn(grid->getNYN()),
    nzc(grid->getNZC()),
    nzn(grid->getNZN()),
    dx(grid->getDX()),
    dy(grid->getDY()),
    dz(grid->getDZ()),
    invVOL(grid->getInvVOL()),
    xStart(grid->getXstart()),
    xEnd(grid->getXend()),
    yStart(grid->getYstart()),
    yEnd(grid->getYend()),
    zStart(grid->getZstart()),
    zEnd(grid->getZend()),
    Lx(col->getLx()),
    Ly(col->getLy()),
    Lz(col->getLz()),
    ns(col->getNs()),
    c(col->getC()),
    dt(col->getDt()),
    th(col->getTh()),
    ue0(col->getU0(0)),
    ve0(col->getV0(0)),
    we0(col->getW0(0)),
    x_center(col->getx_center()),
    y_center(col->gety_center()),
    z_center(col->getz_center()),
    L_square(col->getL_square()),
    delt (c*th*dt), // declared after these

    //! Allocate arrays for data on nodes !//
    //? (nxn, nyn, nzn) --> nodes 
    //? (nxc, nyc, nzc) --> cell centres
    fieldForPcls  (nxn, nyn, nzn, 2*DFIELD_3or4),
    Ex      (nxn, nyn, nzn),
    Ey      (nxn, nyn, nzn),
    Ez      (nxn, nyn, nzn),
    Exth    (nxn, nyn, nzn),
    Eyth    (nxn, nyn, nzn),
    Ezth    (nxn, nyn, nzn),

    Bxn     (nxn, nyn, nzn),
    Byn     (nxn, nyn, nzn),
    Bzn     (nxn, nyn, nzn),
    Bxc     (nxc, nyc, nzc),
    Byc     (nxc, nyc, nzc),
    Bzc     (nxc, nyc, nzc),
    Bx_tot  (nxn, nyn, nzn),
    By_tot  (nxn, nyn, nzn),
    Bz_tot  (nxn, nyn, nzn),

    //! E_ext, B_ext, and J_ext are not used (must be allocated even if memory intensive :( )
    //! When external forces are implemented, assign proper memory; currently set to (1, 1, 1) to save on memory
    Bxc_ext  (1, 1, 1),
    Byc_ext  (1, 1, 1),
    Bzc_ext  (1, 1, 1),
    Bx_ext   (1, 1, 1),
    By_ext   (1, 1, 1),
    Bz_ext   (1, 1, 1),
    Jx_ext   (1, 1, 1),
    Jy_ext   (1, 1, 1),
    Jz_ext   (1, 1, 1),
    Ex_ext   (1, 1, 1),
    Ey_ext   (1, 1, 1),
    Ez_ext   (1, 1, 1),
    
    Jx      (nxn, nyn, nzn),
    Jy      (nxn, nyn, nzn),
    Jz      (nxn, nyn, nzn),
    Jxh     (nxn, nyn, nzn),
    Jyh     (nxn, nyn, nzn),
    Jzh     (nxn, nyn, nzn),

    //! Mass matrix quantities
    Mxx (NE_MASS, nxn, nyn, nzn),
    Myy (NE_MASS, nxn, nyn, nzn),
    Mzz (NE_MASS, nxn, nyn, nzn),
    Mxy (NE_MASS, nxn, nyn, nzn),
    Myx (NE_MASS, nxn, nyn, nzn),
    Mxz (NE_MASS, nxn, nyn, nzn),
    Mzx (NE_MASS, nxn, nyn, nzn),
    Myz (NE_MASS, nxn, nyn, nzn),
    Mzy (NE_MASS, nxn, nyn, nzn),

    //! Species-specific quantities
    rhocs_avg (ns, nxc, nyc, nzc),
    rhons     (ns, nxn, nyn, nzn),
    rhocs     (ns, nxc, nyc, nzc),              //* Data defined on cell centres
    Jxs       (ns, nxn, nyn, nzn),
    Jys       (ns, nxn, nyn, nzn),
    Jzs       (ns, nxn, nyn, nzn),
    Jxhs      (ns, nxn, nyn, nzn),
    Jyhs      (ns, nxn, nyn, nzn),
    Jzhs      (ns, nxn, nyn, nzn),
    E_flux_xs      (ns, nxn, nyn, nzn),
    E_flux_ys      (ns, nxn, nyn, nzn),
    E_flux_zs      (ns, nxn, nyn, nzn),
    Nns       (ns, nxn, nyn, nzn),
    residual_divergence (ns, nxc, nyc, nzc),    //* Data defined on cell centres

    pXXsn (ns, nxn, nyn, nzn),
    pXYsn (ns, nxn, nyn, nzn),
    pXZsn (ns, nxn, nyn, nzn),
    pYYsn (ns, nxn, nyn, nzn),
    pYZsn (ns, nxn, nyn, nzn),
    pZZsn (ns, nxn, nyn, nzn),
    
    //? Other arrays
    PHI      (nxc, nyc, nzc),
    rhoc_avg (nxc, nyc, nzc),
    rhoc     (nxc, nyc, nzc),
    rhon     (nxn, nyn, nzn),
    Phic     (nxc, nyc, nzc),

    //? Divergence
    divC        (nxc, nyc, nzc),
    divE        (nxc, nyc, nzc),
    divB        (nxn, nyn, nzn),
    divE_average(nxc, nyc, nzc),
    
    //? Temporary arrays
    tempC   (nxc, nyc, nzc),
    tempXC  (nxc, nyc, nzc),
    tempYC  (nxc, nyc, nzc),
    tempZC  (nxc, nyc, nzc),
    tempXC2 (nxc, nyc, nzc),
    tempYC2 (nxc, nyc, nzc),
    tempZC2 (nxc, nyc, nzc),

    tempX   (nxn, nyn, nzn),
    tempY   (nxn, nyn, nzn),
    tempZ   (nxn, nyn, nzn),
    temp2X  (nxn, nyn, nzn),
    temp2Y  (nxn, nyn, nzn),
    temp2Z  (nxn, nyn, nzn),
    temp3X  (nxn, nyn, nzn),
    temp3Y  (nxn, nyn, nzn),
    temp3Z  (nxn, nyn, nzn),
    tempXN  (nxn, nyn, nzn),
    tempYN  (nxn, nyn, nzn),
    tempZN  (nxn, nyn, nzn),
    smooth_temp(nxn, nyn, nzn),
    
    imageX  (nxn, nyn, nzn),
    imageY  (nxn, nyn, nzn),
    imageZ  (nxn, nyn, nzn),
    Dx      (nxn, nyn, nzn),
    Dy      (nxn, nyn, nzn),
    Dz      (nxn, nyn, nzn),
    vectX   (nxn, nyn, nzn),
    vectY   (nxn, nyn, nzn),
    vectZ   (nxn, nyn, nzn)

{
    //! =============== Constructor =============== !//
  
    //? External fields
    B1x = col->getB1x();
    B1y = col->getB1y();
    B1z = col->getB1z();

    Bx_ext.setall(0.);
    By_ext.setall(0.);
    Bz_ext.setall(0.);
    Bx_tot.setall(0.);
    By_tot.setall(0.);
    Bz_tot.setall(0.);

    GMREStol = col->getGMREStol();
    zeroCurrent = (col->getZeroCurrent() == 1 ? 1 : 0);
    
    qom = new double[ns];
    for (int i = 0; i < ns; i++)
        qom[i] = col->getQOM(i);
    
    //? boundary conditions: PHI and EM fields
    bcPHIfaceXright = col->getBcPHIfaceXright();
    bcPHIfaceXleft  = col->getBcPHIfaceXleft();
    bcPHIfaceYright = col->getBcPHIfaceYright();
    bcPHIfaceYleft  = col->getBcPHIfaceYleft();
    bcPHIfaceZright = col->getBcPHIfaceZright();
    bcPHIfaceZleft  = col->getBcPHIfaceZleft();

    bcEMfaceXright  = col->getBcEMfaceXright();
    bcEMfaceXleft   = col->getBcEMfaceXleft();
    bcEMfaceYright  = col->getBcEMfaceYright();
    bcEMfaceYleft   = col->getBcEMfaceYleft();
    bcEMfaceZright  = col->getBcEMfaceZright();
    bcEMfaceZleft   = col->getBcEMfaceZleft();

    B0x = col->getB0x();
    B0y = col->getB0y();
    B0z = col->getB0z();
    Smooth = col->getSmooth();
    smooth_cycle = col->getSmoothCycle();
    num_smoothings = col->getNumSmoothings();
    SaveHeatFluxTensor = col->getSaveHeatFluxTensor();
    
    rhoINIT = new double[ns];               //* Background density
    DriftSpecies = new bool[ns];
    for (int i = 0; i < ns; i++) 
    {
        rhoINIT[i] = col->getRHOinit(i);
        if ((fabs(col->getW0(i)) != 0) || (fabs(col->getU0(i)) != 0)) //* GEM and LHDI
            DriftSpecies[i] = true;
        else
            DriftSpecies[i] = false;
    }

    FourPI = 16 * atan(1.0);
    restart_status = col->getRestart_status();

    //! Mass Matrix (dyadic product of a vector)
    mass_matrix = new double[3];
    mass_matrix[0] = 0.0;
    mass_matrix[1] = 1.0;
    mass_matrix[2] = 0.0;

    //* Custom input parameters
    nparam = col->getNparam();
    if (nparam > 0) 
    {
        input_param = new double[nparam];
        
        for (int ip=0; ip<nparam; ip++) 
            input_param[ip] = col->getInputParam(ip);
    }

    //* Set all memory allocated to zero
    setAllzero();


    if(Parameters::get_VECTORIZE_MOMENTS())
    {
        //* In this case particles are sorted and there is no need for each thread to sum moments in a separate array.
        sizeMomentsArray = 1;
    }
    else
    {
        sizeMomentsArray = omp_get_max_threads();
    }
    
    moments10Array = new Moments10*[sizeMomentsArray];
    ecsim_moments13Array = new ECSIM_Moments13*[sizeMomentsArray];
    for(int i = 0; i < sizeMomentsArray; i++)
    {
        moments10Array[i] = new Moments10(nxn, nyn, nzn);
        ecsim_moments13Array[i] = new ECSIM_Moments13(nxn, nyn, nzn);
    }

    if (col->getSaveHeatFluxTensor()) 
    {
        Qxxxs = newArr4(double, ns, nxn, nyn, nzn);
        Qxxys = newArr4(double, ns, nxn, nyn, nzn);
        Qxyys = newArr4(double, ns, nxn, nyn, nzn);
        Qxzzs = newArr4(double, ns, nxn, nyn, nzn);
        Qyyys = newArr4(double, ns, nxn, nyn, nzn);
        Qyzzs = newArr4(double, ns, nxn, nyn, nzn);
        Qzzzs = newArr4(double, ns, nxn, nyn, nzn);
        Qxyzs = newArr4(double, ns, nxn, nyn, nzn);
        Qxxzs = newArr4(double, ns, nxn, nyn, nzn);
        Qyyzs = newArr4(double, ns, nxn, nyn, nzn);
    }

    //! Define MPI Derived Data types for Center Halo Exchange
    //? For face exchange on X dir
    MPI_Type_vector((nyc-2),(nzc-2),nzc, MPI_DOUBLE, &yzFacetypeC);
    MPI_Type_commit(&yzFacetypeC);
    
    //? For face exchange on Y dir
    MPI_Type_create_hvector((nxc-2),(nzc-2),(nzc*nyc*sizeof(double)), MPI_DOUBLE, &xzFacetypeC);
    MPI_Type_commit(&xzFacetypeC);

    MPI_Type_vector((nyc-2), 1, nzc, MPI_DOUBLE, &yEdgetypeC);
    MPI_Type_commit(&yEdgetypeC);
    
    //? For face exchangeg on Z dir
    MPI_Type_create_hvector((nxc-2), 1, (nzc*nyc*sizeof(double)), yEdgetypeC, &xyFacetypeC);
    MPI_Type_commit(&xyFacetypeC);
    
    //? 2 yEdgeType can be merged into one message
    MPI_Type_create_hvector(2, 1,(nzc-1)*sizeof(double), yEdgetypeC, &yEdgetypeC2);
    MPI_Type_commit(&yEdgetypeC2);
    
    MPI_Type_contiguous((nzc-2),MPI_DOUBLE, &zEdgetypeC);
    MPI_Type_commit(&zEdgetypeC);
    
    MPI_Type_create_hvector(2, (nzc-2),(nxc-1)*(nyc*nzc)*sizeof(double), MPI_DOUBLE, &zEdgetypeC2);
    MPI_Type_commit(&zEdgetypeC2);
    
    MPI_Type_vector((nxc-2), 1, nyc*nzc, MPI_DOUBLE, &xEdgetypeC);
    MPI_Type_commit(&xEdgetypeC);
    MPI_Type_create_hvector(2, 1, (nyc-1)*nzc*sizeof(double), xEdgetypeC, &xEdgetypeC2);
    MPI_Type_commit(&xEdgetypeC2);
    
    //* corner used to communicate in x direction
    int blocklengthC[]={1,1,1,1};
    int displacementsC[]={0,nzc-1,(nyc-1)*nzc,nyc*nzc-1};
    MPI_Type_indexed(4, blocklengthC, displacementsC, MPI_DOUBLE, &cornertypeC);
    MPI_Type_commit(&cornertypeC);

    //! Define MPI Derived Data types for Node Halo Exchange
    //? For face exchange on X dir
    MPI_Type_vector((nyn-2),(nzn-2),nzn, MPI_DOUBLE, &yzFacetypeN);
    MPI_Type_commit(&yzFacetypeN);

    //? For face exchange on Y dir
    MPI_Type_create_hvector((nxn-2),(nzn-2),(nzn*nyn*sizeof(double)), MPI_DOUBLE, &xzFacetypeN);
    MPI_Type_commit(&xzFacetypeN);

    MPI_Type_vector((nyn-2), 1, nzn, MPI_DOUBLE, &yEdgetypeN);
    MPI_Type_commit(&yEdgetypeN);

    //? For face exchangeg on Z dir
    MPI_Type_create_hvector((nxn-2), 1, (nzn*nyn*sizeof(double)), yEdgetypeN, &xyFacetypeN);
    MPI_Type_commit(&xyFacetypeN);

    //? 2 yEdgeType can be merged into one message
    MPI_Type_create_hvector(2, 1,(nzn-1)*sizeof(double), yEdgetypeN, &yEdgetypeN2);
    MPI_Type_commit(&yEdgetypeN2);

    MPI_Type_contiguous((nzn-2),MPI_DOUBLE, &zEdgetypeN);
    MPI_Type_commit(&zEdgetypeN);

    MPI_Type_create_hvector(2, (nzn-2),(nxn-1)*(nyn*nzn)*sizeof(double), MPI_DOUBLE, &zEdgetypeN2);
    MPI_Type_commit(&zEdgetypeN2);

    MPI_Type_vector((nxn-2), 1, nyn*nzn, MPI_DOUBLE, &xEdgetypeN);
    MPI_Type_commit(&xEdgetypeN);
    MPI_Type_create_hvector(2, 1, (nyn-1)*nzn*sizeof(double), xEdgetypeN, &xEdgetypeN2);
    MPI_Type_commit(&xEdgetypeN2);

    //* corner used to communicate in x direction
    int blocklengthN[]={1,1,1,1};
    int displacementsN[]={0,nzn-1,(nyn-1)*nzn,nyn*nzn-1};
    MPI_Type_indexed(4, blocklengthN, displacementsN, MPI_DOUBLE, &cornertypeN);
    MPI_Type_commit(&cornertypeN);

    //! Write data to files
    if (col->getWriteMethod() == "pvtk" || col->getWriteMethod() == "nbcvtk")
    {
    	//* Test Endian
    	int TestEndian = 1;
    	lEndFlag =*(char*)&TestEndian;

        //* Create process file view
        int  size[3], subsize[3], start[3];

        //* 3D subarray - reverse X, Z
        subsize[0] = nzc-2; subsize[1] = nyc-2; subsize[2] = nxc-2;
        size[0] = (nzc-2)*vct->getZLEN();size[1] = (nyc-2)*vct->getYLEN();size[2] = (nxc-2)*vct->getXLEN();
        start[0]= vct->getCoordinates(2)*subsize[0];
        start[1]= vct->getCoordinates(1)*subsize[1];
        start[2]= vct->getCoordinates(0)*subsize[2];

        MPI_Type_contiguous(3,MPI_FLOAT, &xyzcomp);
        MPI_Type_commit(&xyzcomp);

        MPI_Type_create_subarray(3, size, subsize, start,MPI_ORDER_C, xyzcomp, &procviewXYZ);
        MPI_Type_commit(&procviewXYZ);

        MPI_Type_create_subarray(3, size, subsize, start,MPI_ORDER_C, MPI_FLOAT, &procview);
        MPI_Type_commit(&procview);

        subsize[0] = nxc-2; subsize[1] =nyc-2; subsize[2] = nzc-2;
        size[0]    = nxc;	  size[1] 	 =nyc;	 size[2] 	= nzc;
        start[0]	 = 1;	  start[1]	 =1;	 start[2]	= 1;
        MPI_Type_create_subarray(3, size, subsize, start,MPI_ORDER_C, MPI_FLOAT, &ghosttype);
        MPI_Type_commit(&ghosttype);
    }
}

void EMfields3D::setAllzero()
{
    // Fext = 1;
    // Fzro = 1;

    //* 3D arrays defined on nodes
    for (int ii = 0; ii < nxn; ii++)
        for (int jj = 0; jj < nyn; jj++)
            for (int kk = 0; kk < nzn; kk++)
            {
                Ex.fetch(ii, jj, kk)        = 0.0;
                Ey.fetch(ii, jj, kk)        = 0.0;
                Ez.fetch(ii, jj, kk)        = 0.0;
                Exth.fetch(ii, jj, kk)      = 0.0;
                Eyth.fetch(ii, jj, kk)      = 0.0;
                Ezth.fetch(ii, jj, kk)      = 0.0;
                Bxn.fetch(ii, jj, kk)       = 0.0;
                Byn.fetch(ii, jj, kk)       = 0.0;
                Bzn.fetch(ii, jj, kk)       = 0.0;
                Bx_tot.fetch(ii, jj, kk)    = 0.0;
                By_tot.fetch(ii, jj, kk)    = 0.0;
                Bz_tot.fetch(ii, jj, kk)    = 0.0;
                Jxh.fetch(ii, jj, kk)       = 0.0;
                Jyh.fetch(ii, jj, kk)       = 0.0;
                Jzh.fetch(ii, jj, kk)       = 0.0;
                
                rhon.fetch(ii, jj, kk)      = 0.0;
                divB.fetch(ii, jj, kk)      = 0.0;

                //! E_ext, B_ext, and J_ext are not used
                // Ex_ext.fetch(ii, jj, kk)    = 0.0;
                // Ey_ext.fetch(ii, jj, kk)    = 0.0;
                // Ez_ext.fetch(ii, jj, kk)    = 0.0;
                // Bx_ext.fetch(ii, jj, kk)    = 0.0;
                // By_ext.fetch(ii, jj, kk)    = 0.0;
                // Bz_ext.fetch(ii, jj, kk)    = 0.0;
                // Jx_ext.fetch(ii, jj, kk)    = 0.0;
                // Jy_ext.fetch(ii, jj, kk)    = 0.0;
                // Jz_ext.fetch(ii, jj, kk)    = 0.0;

                tempX.fetch(ii, jj, kk)     = 0.0;
                tempY.fetch(ii, jj, kk)     = 0.0;
                tempZ.fetch(ii, jj, kk)     = 0.0;
                temp2X.fetch(ii, jj, kk)    = 0.0;
                temp2Y.fetch(ii, jj, kk)    = 0.0;
                temp2Z.fetch(ii, jj, kk)    = 0.0;
                temp3X.fetch(ii, jj, kk)    = 0.0;
                temp3Y.fetch(ii, jj, kk)    = 0.0;
                temp3Z.fetch(ii, jj, kk)    = 0.0;
                tempXN.fetch(ii, jj, kk)    = 0.0;
                tempYN.fetch(ii, jj, kk)    = 0.0;
                tempZN.fetch(ii, jj, kk)    = 0.0;
                imageX.fetch(ii, jj, kk)    = 0.0;
                imageY.fetch(ii, jj, kk)    = 0.0;
                imageZ.fetch(ii, jj, kk)    = 0.0;
                vectX.fetch(ii, jj, kk)     = 0.0;
                vectY.fetch(ii, jj, kk)     = 0.0;
                vectZ.fetch(ii, jj, kk)     = 0.0;
                Dx.fetch(ii, jj, kk)        = 0.0;
                Dy.fetch(ii, jj, kk)        = 0.0;
                Dz.fetch(ii, jj, kk)        = 0.0;
            }

    //* 3D arrays defined at cell centres
    for (int ii = 0; ii < nxc; ii++)
        for (int jj = 0; jj < nyc; jj++)
            for (int kk = 0; kk < nzc; kk++)
            {
                Bxc.fetch(ii, jj, kk)           = 0.0;
                Byc.fetch(ii, jj, kk)           = 0.0;
                Bzc.fetch(ii, jj, kk)           = 0.0;
                // Bxc_ext.fetch(ii, jj, kk)       = 0.0;
                // Byc_ext.fetch(ii, jj, kk)       = 0.0;
                // Bzc_ext.fetch(ii, jj, kk)       = 0.0;
                divE.fetch(ii, jj, kk)          = 0.0;
                divE_average.fetch(ii, jj, kk)  = 0.0;
                rhoc.fetch(ii, jj, kk)          = 0.0;
                rhoc_avg.fetch(ii, jj, kk)      = 0.0;
                tempXC.fetch(ii, jj, kk)        = 0.0;
                tempYC.fetch(ii, jj, kk)        = 0.0;
                tempZC.fetch(ii, jj, kk)        = 0.0;
                tempXC2.fetch(ii, jj, kk)       = 0.0;
                tempYC2.fetch(ii, jj, kk)       = 0.0;
                tempZC2.fetch(ii, jj, kk)       = 0.0;
                tempC.fetch(ii, jj, kk)         = 0.0;
            }


    //* 4D arrays defined on nodes
    for (int is = 0; is < ns; is ++)
        for (int ii = 0; ii < nxn; ii++)
            for (int jj = 0; jj < nyn; jj++)
                for (int kk = 0; kk < nzn; kk++)
                {
                    Jxs.fetch(is, ii, jj, kk)   = 0.0;
                    Jys.fetch(is, ii, jj, kk)   = 0.0;
                    Jzs.fetch(is, ii, jj, kk)   = 0.0;
                    Jxhs.fetch(is, ii, jj, kk)  = 0.0;
                    Jyhs.fetch(is, ii, jj, kk)  = 0.0;
                    Jzhs.fetch(is, ii, jj, kk)  = 0.0;
                    E_flux_xs.fetch(is, ii, jj, kk)  = 0.0;
                    E_flux_ys.fetch(is, ii, jj, kk)  = 0.0;
                    E_flux_zs.fetch(is, ii, jj, kk)  = 0.0;
                    rhons.fetch(is, ii, jj, kk) = 0.0;
                }

    for (int is = 0; is < NE_MASS; is ++)
        for (int ii = 0; ii < nxn; ii++)
            for (int jj = 0; jj < nyn; jj++)
                    for (int kk = 0; kk < nzn; kk++)
                    {
                        Mxx.fetch(is, ii, jj, kk) = 0.0;
                        Mxy.fetch(is, ii, jj, kk) = 0.0;
                        Mxz.fetch(is, ii, jj, kk) = 0.0;
                        Myx.fetch(is, ii, jj, kk) = 0.0;
                        Myy.fetch(is, ii, jj, kk) = 0.0;
                        Myz.fetch(is, ii, jj, kk) = 0.0;
                        Mzx.fetch(is, ii, jj, kk) = 0.0;
                        Mzy.fetch(is, ii, jj, kk) = 0.0;
                        Mzz.fetch(is, ii, jj, kk) = 0.0;
                    }

    //* 4D arrays defined at cell centres
    for (int is = 0; is < ns; is ++)
        for (int ii = 0; ii < nxc; ii++)
            for (int jj = 0; jj < nyc; jj++)
                for (int kk = 0; kk < nzc; kk++)
                {
                    rhocs.fetch(is, ii, jj, kk) = 0.0;
                    rhocs_avg.fetch(is, ii, jj, kk) = 0.0;
                    residual_divergence.fetch(is, ii, jj, kk) = 0.0;
                }
}


void EMfields3D::freeDataType()
{
    MPI_Type_free(&yzFacetypeC);
    MPI_Type_free(&xzFacetypeC);
    MPI_Type_free(&xyFacetypeC);
    MPI_Type_free(&xEdgetypeC);
    MPI_Type_free(&yEdgetypeC);
    MPI_Type_free(&zEdgetypeC);
    MPI_Type_free(&xEdgetypeC2);
    MPI_Type_free(&yEdgetypeC2);
    MPI_Type_free(&zEdgetypeC2);
    MPI_Type_free(&cornertypeC);

    MPI_Type_free(&yzFacetypeN);
    MPI_Type_free(&xzFacetypeN);
    MPI_Type_free(&xyFacetypeN);
    MPI_Type_free(&xEdgetypeN);
    MPI_Type_free(&yEdgetypeN);
    MPI_Type_free(&zEdgetypeN);
    MPI_Type_free(&xEdgetypeN2);
    MPI_Type_free(&yEdgetypeN2);
    MPI_Type_free(&zEdgetypeN2);
    MPI_Type_free(&cornertypeN);

    if (_col.getWriteMethod() == "pvtk" || _col.getWriteMethod() == "nbcvtk")
    {
        MPI_Type_free(&procview);
        MPI_Type_free(&xyzcomp);
        MPI_Type_free(&procviewXYZ);
        MPI_Type_free(&ghosttype);
    }
}

//! ===================================== Compute Moments ===================================== !//

//? This was Particles3Dcomm::interpP2G()
void EMfields3D::sumMomentsOld(const Particles3Dcomm& pcls)
{
  const Grid *grid = &get_grid();

  const double inv_dx = 1.0 / dx;
  const double inv_dy = 1.0 / dy;
  const double inv_dz = 1.0 / dz;
  const int nxn = grid->getNXN();
  const int nyn = grid->getNYN();
  const int nzn = grid->getNZN();
  const double xstart = grid->getXstart();
  const double ystart = grid->getYstart();
  const double zstart = grid->getZstart();
  double const*const x = pcls.getXall();
  double const*const y = pcls.getYall();
  double const*const z = pcls.getZall();
  double const*const u = pcls.getUall();
  double const*const v = pcls.getVall();
  double const*const w = pcls.getWall();
  double const*const q = pcls.getQall();
  //
  const int is = pcls.get_species_num();

  const int nop = pcls.getNOP();
  // To make memory use scale to a large number of threads, we
  // could first apply an efficient parallel sorting algorithm
  // to the particles and then accumulate moments in smaller
  // subarrays.
  //#ifdef _OPENMP
  TimeTasks timeTasksAcc;
  #pragma omp parallel private(timeTasks)
  {
    int thread_num = omp_get_thread_num();
    Moments10& speciesMoments10 = fetch_moments10Array(thread_num);
    speciesMoments10.set_to_zero();
    arr4_double moments = speciesMoments10.fetch_arr();
    // The following loop is expensive, so it is wise to assume that the
    // compiler is stupid.  Therefore we should on the one hand
    // expand things out and on the other hand avoid repeating computations.
    #pragma omp for
    for (int i = 0; i < nop; i++)
    {
      // compute the quadratic moments of velocity
      //
      const double ui=u[i];
      const double vi=v[i];
      const double wi=w[i];
      const double uui=ui*ui;
      const double uvi=ui*vi;
      const double uwi=ui*wi;
      const double vvi=vi*vi;
      const double vwi=vi*wi;
      const double wwi=wi*wi;
      double velmoments[10];
      velmoments[0] = 1.;
      velmoments[1] = ui;
      velmoments[2] = vi;
      velmoments[3] = wi;
      velmoments[4] = uui;
      velmoments[5] = uvi;
      velmoments[6] = uwi;
      velmoments[7] = vvi;
      velmoments[8] = vwi;
      velmoments[9] = wwi;

      //
      // compute the weights to distribute the moments
      //
      const int ix = 2 + int (floor((x[i] - xstart) * inv_dx));
      const int iy = 2 + int (floor((y[i] - ystart) * inv_dy));
      const int iz = 2 + int (floor((z[i] - zstart) * inv_dz));
      const double xi0   = x[i] - grid->getXN(ix-1);
      const double eta0  = y[i] - grid->getYN(iy-1);
      const double zeta0 = z[i] - grid->getZN(iz-1);
      const double xi1   = grid->getXN(ix) - x[i];
      const double eta1  = grid->getYN(iy) - y[i];
      const double zeta1 = grid->getZN(iz) - z[i];
      const double qi = q[i];
      const double weight000 = qi * xi0 * eta0 * zeta0 * invVOL;
      const double weight001 = qi * xi0 * eta0 * zeta1 * invVOL;
      const double weight010 = qi * xi0 * eta1 * zeta0 * invVOL;
      const double weight011 = qi * xi0 * eta1 * zeta1 * invVOL;
      const double weight100 = qi * xi1 * eta0 * zeta0 * invVOL;
      const double weight101 = qi * xi1 * eta0 * zeta1 * invVOL;
      const double weight110 = qi * xi1 * eta1 * zeta0 * invVOL;
      const double weight111 = qi * xi1 * eta1 * zeta1 * invVOL;
      double weights[8];
      weights[0] = weight000;
      weights[1] = weight001;
      weights[2] = weight010;
      weights[3] = weight011;
      weights[4] = weight100;
      weights[5] = weight101;
      weights[6] = weight110;
      weights[7] = weight111;

      // add particle to moments
      {
        arr1_double_fetch momentsArray[8];
        momentsArray[0] = moments[ix  ][iy  ][iz  ]; // moments000 
        momentsArray[1] = moments[ix  ][iy  ][iz-1]; // moments001 
        momentsArray[2] = moments[ix  ][iy-1][iz  ]; // moments010 
        momentsArray[3] = moments[ix  ][iy-1][iz-1]; // moments011 
        momentsArray[4] = moments[ix-1][iy  ][iz  ]; // moments100 
        momentsArray[5] = moments[ix-1][iy  ][iz-1]; // moments101 
        momentsArray[6] = moments[ix-1][iy-1][iz  ]; // moments110 
        momentsArray[7] = moments[ix-1][iy-1][iz-1]; // moments111 

        for(int m=0; m<10; m++)
        for(int c=0; c<8; c++)
        {
          momentsArray[c][m] += velmoments[m]*weights[c];
        }
      }
    }
    

    // reduction
    

    // reduce arrays
    {
      #pragma omp critical (reduceMoment0)
      for(int i=0;i<nxn;i++){for(int j=0;j<nyn;j++) for(int k=0;k<nzn;k++)
        { rhons[is][i][j][k] += invVOL*moments[i][j][k][0]; }}
      #pragma omp critical (reduceMoment1)
      for(int i=0;i<nxn;i++){for(int j=0;j<nyn;j++) for(int k=0;k<nzn;k++)
        { Jxs  [is][i][j][k] += invVOL*moments[i][j][k][1]; }}
      #pragma omp critical (reduceMoment2)
      for(int i=0;i<nxn;i++){for(int j=0;j<nyn;j++) for(int k=0;k<nzn;k++)
        { Jys  [is][i][j][k] += invVOL*moments[i][j][k][2]; }}
      #pragma omp critical (reduceMoment3)
      for(int i=0;i<nxn;i++){for(int j=0;j<nyn;j++) for(int k=0;k<nzn;k++)
        { Jzs  [is][i][j][k] += invVOL*moments[i][j][k][3]; }}
      #pragma omp critical (reduceMoment4)
      for(int i=0;i<nxn;i++){for(int j=0;j<nyn;j++) for(int k=0;k<nzn;k++)
        { pXXsn[is][i][j][k] += invVOL*moments[i][j][k][4]; }}
      #pragma omp critical (reduceMoment5)
      for(int i=0;i<nxn;i++){for(int j=0;j<nyn;j++) for(int k=0;k<nzn;k++)
        { pXYsn[is][i][j][k] += invVOL*moments[i][j][k][5]; }}
      #pragma omp critical (reduceMoment6)
      for(int i=0;i<nxn;i++){for(int j=0;j<nyn;j++) for(int k=0;k<nzn;k++)
        { pXZsn[is][i][j][k] += invVOL*moments[i][j][k][6]; }}
      #pragma omp critical (reduceMoment7)
      for(int i=0;i<nxn;i++){for(int j=0;j<nyn;j++) for(int k=0;k<nzn;k++)
        { pYYsn[is][i][j][k] += invVOL*moments[i][j][k][7]; }}
      #pragma omp critical (reduceMoment8)
      for(int i=0;i<nxn;i++){for(int j=0;j<nyn;j++) for(int k=0;k<nzn;k++)
        { pYZsn[is][i][j][k] += invVOL*moments[i][j][k][8]; }}
      #pragma omp critical (reduceMoment9)
      for(int i=0;i<nxn;i++){for(int j=0;j<nyn;j++) for(int k=0;k<nzn;k++)
        { pZZsn[is][i][j][k] += invVOL*moments[i][j][k][9]; }}
    }
    
    #pragma omp critical
    timeTasksAcc += timeTasks;
  }
  // reset timeTasks to be its average value for all threads
  timeTasksAcc /= omp_get_max_threads();
  timeTasks = timeTasksAcc;
  communicateGhostP2G(is);
}
//
// Create a vectorized version of this moment accumulator as follows.
//
// A. Moment accumulation
//
// Case 1: Assuming AoS particle layout and using intrinsics vectorization:
//   Process P:=N/4 particles at a time:
//   1. gather position coordinates from P particles and
//      generate Px8 array of weights and P cell indices.
//   2. for each particle, add 10x8 array of moment-weight
//      products to appropriate cell accumulator.
//   Each cell now has a 10x8 array of node-destined moments.
//   (See sumMoments_AoS_intr().)
// Case 2: Assuming SoA particle layout and using trivial vectorization:
//   Process N:=sizeof(vector_unit)/sizeof(double) particles at a time:
//   1. for pcl=1:N: (3) positions -> (8) weights, cell_index
//   2. For each of 10 moments:
//      a. for pcl=1:N: (<=2 of 3) charge velocities -> (1) moment
//      b. for pcl=1:N: (1) moment, (8) weights -> (8) node-destined moments
//      c. transpose 8xN array of node-destined moments to Nx8 array 
//      d. foreach pcl: add node-distined moments to cell of cell_index
//   Each cell now has a 10x8 array of node-destined moments.
//
//   If particles are sorted by mesh cell, then all moments are destined
//   for the same node; in this case, we can simply accumulate an 8xN
//   array of node-destined moments in each mesh cell and at the end
//   gather these moments at the nodes; to help performance and
//   code reuse, we will in each cell first transpose the 8xN array
//   of node-destined moments to an Nx8 array.
//
// B: Moment reduction
//
//   Gather the information from cells to nodes:
//   3. [foreach cell transpose node-destined moments:
//      10x8 -> 8x10, or rather (8+2)x8 -> 8x8 + 8x2]
//   4. at each node gather moments from cells.
//   5. [transpose moments at nodes if step 3 was done.]
//
//   We will likely omit steps 3 and 5; they could help to optimize,
//   but even without these steps, step 4 is not expected to dominate.
//
// Compare the vectorization notes at the top of mover_PC().
//
// This was Particles3Dcomm::interpP2G()
void EMfields3D::sumMoments(const Particles3Dcomm* part)
{
  const Grid *grid = &get_grid();

  const double inv_dx = 1.0 / dx;
  const double inv_dy = 1.0 / dy;
  const double inv_dz = 1.0 / dz;
  const int nxn = grid->getNXN();
  const int nyn = grid->getNYN();
  const int nzn = grid->getNZN();
  const double xstart = grid->getXstart();
  const double ystart = grid->getYstart();
  const double zstart = grid->getZstart();
  // To make memory use scale to a large number of threads, we
  // could first apply an efficient parallel sorting algorithm
  // to the particles and then accumulate moments in smaller
  // subarrays.
  //#ifdef _OPENMP
  #pragma omp parallel
  {
  for (int i = 0; i < ns; i++)
  {
    const Particles3Dcomm& pcls = part[i];
    assert_eq(pcls.get_particleType(), ParticleType::SoA);
    const int is = pcls.get_species_num();
    assert_eq(i,is);

    double const*const x = pcls.getXall();
    double const*const y = pcls.getYall();
    double const*const z = pcls.getZall();
    double const*const u = pcls.getUall();
    double const*const v = pcls.getVall();
    double const*const w = pcls.getWall();
    double const*const q = pcls.getQall();

    const int nop = pcls.getNOP();

    int thread_num = omp_get_thread_num();
    
    Moments10& speciesMoments10 = fetch_moments10Array(thread_num);
    arr4_double moments = speciesMoments10.fetch_arr();
    //
    // moments.setmode(ompmode::mine);
    // moments.setall(0.);
    // 
    double *moments1d = &moments[0][0][0][0];
    int moments1dsize = moments.get_size();
    for(int i=0; i<moments1dsize; i++) moments1d[i]=0;
    //
    // This barrier is not needed
    #pragma omp barrier
    // The following loop is expensive, so it is wise to assume that the
    // compiler is stupid.  Therefore we should on the one hand
    // expand things out and on the other hand avoid repeating computations.
    #pragma omp for // used nowait with the old way
    for (int i = 0; i < nop; i++)
    {
      // compute the quadratic moments of velocity
      //
      const double ui=u[i];
      const double vi=v[i];
      const double wi=w[i];
      const double uui=ui*ui;
      const double uvi=ui*vi;
      const double uwi=ui*wi;
      const double vvi=vi*vi;
      const double vwi=vi*wi;
      const double wwi=wi*wi;
      double velmoments[10];
      velmoments[0] = 1.;
      velmoments[1] = ui;
      velmoments[2] = vi;
      velmoments[3] = wi;
      velmoments[4] = uui;
      velmoments[5] = uvi;
      velmoments[6] = uwi;
      velmoments[7] = vvi;
      velmoments[8] = vwi;
      velmoments[9] = wwi;

      //
      // compute the weights to distribute the moments
      //
      const int ix = 2 + int (floor((x[i] - xstart) * inv_dx));
      const int iy = 2 + int (floor((y[i] - ystart) * inv_dy));
      const int iz = 2 + int (floor((z[i] - zstart) * inv_dz));
      const double xi0   = x[i] - grid->getXN(ix-1);
      const double eta0  = y[i] - grid->getYN(iy-1);
      const double zeta0 = z[i] - grid->getZN(iz-1);
      const double xi1   = grid->getXN(ix) - x[i];
      const double eta1  = grid->getYN(iy) - y[i];
      const double zeta1 = grid->getZN(iz) - z[i];
      const double qi = q[i];
      const double invVOLqi = invVOL*qi;
      const double weight0 = invVOLqi * xi0;
      const double weight1 = invVOLqi * xi1;
      const double weight00 = weight0*eta0;
      const double weight01 = weight0*eta1;
      const double weight10 = weight1*eta0;
      const double weight11 = weight1*eta1;
      double weights[8];
      weights[0] = weight00*zeta0; // weight000
      weights[1] = weight00*zeta1; // weight001
      weights[2] = weight01*zeta0; // weight010
      weights[3] = weight01*zeta1; // weight011
      weights[4] = weight10*zeta0; // weight100
      weights[5] = weight10*zeta1; // weight101
      weights[6] = weight11*zeta0; // weight110
      weights[7] = weight11*zeta1; // weight111
      //weights[0] = xi0 * eta0 * zeta0 * qi * invVOL; // weight000
      //weights[1] = xi0 * eta0 * zeta1 * qi * invVOL; // weight001
      //weights[2] = xi0 * eta1 * zeta0 * qi * invVOL; // weight010
      //weights[3] = xi0 * eta1 * zeta1 * qi * invVOL; // weight011
      //weights[4] = xi1 * eta0 * zeta0 * qi * invVOL; // weight100
      //weights[5] = xi1 * eta0 * zeta1 * qi * invVOL; // weight101
      //weights[6] = xi1 * eta1 * zeta0 * qi * invVOL; // weight110
      //weights[7] = xi1 * eta1 * zeta1 * qi * invVOL; // weight111

      // add particle to moments
      {
        arr1_double_fetch momentsArray[8];
        arr2_double_fetch moments00 = moments[ix  ][iy  ];
        arr2_double_fetch moments01 = moments[ix  ][iy-1];
        arr2_double_fetch moments10 = moments[ix-1][iy  ];
        arr2_double_fetch moments11 = moments[ix-1][iy-1];
        momentsArray[0] = moments00[iz  ]; // moments000 
        momentsArray[1] = moments00[iz-1]; // moments001 
        momentsArray[2] = moments01[iz  ]; // moments010 
        momentsArray[3] = moments01[iz-1]; // moments011 
        momentsArray[4] = moments10[iz  ]; // moments100 
        momentsArray[5] = moments10[iz-1]; // moments101 
        momentsArray[6] = moments11[iz  ]; // moments110 
        momentsArray[7] = moments11[iz-1]; // moments111 

        for(int m=0; m<10; m++)
        for(int c=0; c<8; c++)
        {
          momentsArray[c][m] += velmoments[m]*weights[c];
        }
      }
    }
    

    // reduction
    

    // reduce moments in parallel
    //
    for(int thread_num=0;thread_num<get_sizeMomentsArray();thread_num++)
    {
      arr4_double moments = fetch_moments10Array(thread_num).fetch_arr();
      #pragma omp for collapse(2)
      for(int i=0;i<nxn;i++)
      for(int j=0;j<nyn;j++)
      for(int k=0;k<nzn;k++)
      {
        rhons[is][i][j][k] += invVOL*moments[i][j][k][0];
        Jxs  [is][i][j][k] += invVOL*moments[i][j][k][1];
        Jys  [is][i][j][k] += invVOL*moments[i][j][k][2];
        Jzs  [is][i][j][k] += invVOL*moments[i][j][k][3];
        pXXsn[is][i][j][k] += invVOL*moments[i][j][k][4];
        pXYsn[is][i][j][k] += invVOL*moments[i][j][k][5];
        pXZsn[is][i][j][k] += invVOL*moments[i][j][k][6];
        pYYsn[is][i][j][k] += invVOL*moments[i][j][k][7];
        pYZsn[is][i][j][k] += invVOL*moments[i][j][k][8];
        pZZsn[is][i][j][k] += invVOL*moments[i][j][k][9];
      }
    }
    //
    // This was the old way of reducing;
    // did not scale well to large number of threads
    //{
    //  #pragma omp critical (reduceMoment0)
    //  for(int i=0;i<nxn;i++){for(int j=0;j<nyn;j++) for(int k=0;k<nzn;k++)
    //    { rhons[is][i][j][k] += invVOL*moments[i][j][k][0]; }}
    //  #pragma omp critical (reduceMoment1)
    //  for(int i=0;i<nxn;i++){for(int j=0;j<nyn;j++) for(int k=0;k<nzn;k++)
    //    { Jxs  [is][i][j][k] += invVOL*moments[i][j][k][1]; }}
    //  #pragma omp critical (reduceMoment2)
    //  for(int i=0;i<nxn;i++){for(int j=0;j<nyn;j++) for(int k=0;k<nzn;k++)
    //    { Jys  [is][i][j][k] += invVOL*moments[i][j][k][2]; }}
    //  #pragma omp critical (reduceMoment3)
    //  for(int i=0;i<nxn;i++){for(int j=0;j<nyn;j++) for(int k=0;k<nzn;k++)
    //    { Jzs  [is][i][j][k] += invVOL*moments[i][j][k][3]; }}
    //  #pragma omp critical (reduceMoment4)
    //  for(int i=0;i<nxn;i++){for(int j=0;j<nyn;j++) for(int k=0;k<nzn;k++)
    //    { pXXsn[is][i][j][k] += invVOL*moments[i][j][k][4]; }}
    //  #pragma omp critical (reduceMoment5)
    //  for(int i=0;i<nxn;i++){for(int j=0;j<nyn;j++) for(int k=0;k<nzn;k++)
    //    { pXYsn[is][i][j][k] += invVOL*moments[i][j][k][5]; }}
    //  #pragma omp critical (reduceMoment6)
    //  for(int i=0;i<nxn;i++){for(int j=0;j<nyn;j++) for(int k=0;k<nzn;k++)
    //    { pXZsn[is][i][j][k] += invVOL*moments[i][j][k][6]; }}
    //  #pragma omp critical (reduceMoment7)
    //  for(int i=0;i<nxn;i++){for(int j=0;j<nyn;j++) for(int k=0;k<nzn;k++)
    //    { pYYsn[is][i][j][k] += invVOL*moments[i][j][k][7]; }}
    //  #pragma omp critical (reduceMoment8)
    //  for(int i=0;i<nxn;i++){for(int j=0;j<nyn;j++) for(int k=0;k<nzn;k++)
    //    { pYZsn[is][i][j][k] += invVOL*moments[i][j][k][8]; }}
    //  #pragma omp critical (reduceMoment9)
    //  for(int i=0;i<nxn;i++){for(int j=0;j<nyn;j++) for(int k=0;k<nzn;k++)
    //    { pZZsn[is][i][j][k] += invVOL*moments[i][j][k][9]; }}
    //}
    
    // uncomment this and remove the loop below
    // when we change to use asynchronous communication.
    // communicateGhostP2G(is, vct);
  }
  }
  for (int i = 0; i < ns; i++)
  {
    communicateGhostP2G(i);
  }
}

//! This is the alternative to "computeMoments()" in CalculateMoments(). 
void EMfields3D::sumMoments_AoS(const Particles3Dcomm* part)
{
    cout << "sumMoments_AoS" << endl;

    const Grid *grid = &get_grid();

    const double inv_dx = 1.0 / dx;
    const double inv_dy = 1.0 / dy;
    const double inv_dz = 1.0 / dz;
    const int nxn = grid->getNXN();
    const int nyn = grid->getNYN();
    const int nzn = grid->getNZN();
    const double xstart = grid->getXstart();
    const double ystart = grid->getYstart();
    const double zstart = grid->getZstart();

    #pragma omp parallel
    {
    for (int species_idx = 0; species_idx < ns; species_idx++)
    {
        const Particles3Dcomm& pcls = part[species_idx];
        assert_eq(pcls.get_particleType(), ParticleType::AoS);
        
        //* Get number of species of particles (is)
        const int is = pcls.get_species_num();
        assert_eq(species_idx, is);

        //* Get number of particles 
        const int nop = pcls.getNOP();

        int thread_num = omp_get_thread_num();
                
        ECSIM_Moments13& speciesMoments13 = fetch_moments13Array(thread_num);
        arr4_double moments = speciesMoments13.fetch_arr();

        double *moments1d = &moments[0][0][0][0];
        int moments1dsize = moments.get_size();
        for(int i = 0; i < moments1dsize; i++) moments1d[i] = 0;
        
        #pragma omp barrier
        #pragma omp for
        for (int pidx = 0; pidx < nop; pidx++)
        {
            const SpeciesParticle& pcl = pcls.get_pcl(pidx);
            
            //* Compute quadratic moments of velocity
            const double ui  = pcl.get_u();
            const double vi  = pcl.get_v();
            const double wi  = pcl.get_w();
            const double uui = ui*ui;
            const double uvi = ui*vi;
            const double uwi = ui*wi;
            const double vvi = vi*vi;
            const double vwi = vi*wi;
            const double wwi = wi*wi;
            
            double velmoments[10];
            velmoments[0] = 1.;     //* charge density
            velmoments[1] = ui;     //* momentum density 
            velmoments[2] = vi;
            velmoments[3] = wi;
            velmoments[4] = uui;    //* second time momentum
            velmoments[5] = uvi;
            velmoments[6] = uwi;
            velmoments[7] = vvi;
            velmoments[8] = vwi;
            velmoments[9] = wwi;

            //? Compute weights to distribute the moments
            const int ix = 2 + int (floor((pcl.get_x() - xstart) * inv_dx));
            const int iy = 2 + int (floor((pcl.get_y() - ystart) * inv_dy));
            const int iz = 2 + int (floor((pcl.get_z() - zstart) * inv_dz));
            const double xi0   = pcl.get_x() - grid->getXN(ix-1);
            const double eta0  = pcl.get_y() - grid->getYN(iy-1);
            const double zeta0 = pcl.get_z() - grid->getZN(iz-1);
            const double xi1   = grid->getXN(ix) - pcl.get_x();
            const double eta1  = grid->getYN(iy) - pcl.get_y();
            const double zeta1 = grid->getZN(iz) - pcl.get_z();
            const double qi = pcl.get_q();
            const double invVOLqi = invVOL*qi;
            const double weight0  = invVOLqi*xi0;
            const double weight1  = invVOLqi*xi1;
            const double weight00 = weight0*eta0;
            const double weight01 = weight0*eta1;
            const double weight10 = weight1*eta0;
            const double weight11 = weight1*eta1;
            double weights[8];
            weights[0] = weight00*zeta0; // weight000 = xi0 * eta0 * zeta0 * qi * invVOL
            weights[1] = weight00*zeta1; // weight001 = xi0 * eta0 * zeta1 * qi * invVOL
            weights[2] = weight01*zeta0; // weight010 = xi0 * eta1 * zeta0 * qi * invVOL
            weights[3] = weight01*zeta1; // weight011 = xi0 * eta1 * zeta1 * qi * invVOL
            weights[4] = weight10*zeta0; // weight100 = xi1 * eta0 * zeta0 * qi * invVOL
            weights[5] = weight10*zeta1; // weight101 = xi1 * eta0 * zeta1 * qi * invVOL
            weights[6] = weight11*zeta0; // weight110 = xi1 * eta1 * zeta0 * qi * invVOL
            weights[7] = weight11*zeta1; // weight111 = xi1 * eta1 * zeta1 * qi * invVOL

            //* Add particle to moments
            arr1_double_fetch momentsArray[8];
            arr2_double_fetch moments00 = moments[ix  ][iy  ];
            arr2_double_fetch moments01 = moments[ix  ][iy-1];
            arr2_double_fetch moments10 = moments[ix-1][iy  ];
            arr2_double_fetch moments11 = moments[ix-1][iy-1];
            momentsArray[0] = moments00[iz  ]; // moments000 
            momentsArray[1] = moments00[iz-1]; // moments001 
            momentsArray[2] = moments01[iz  ]; // moments010 
            momentsArray[3] = moments01[iz-1]; // moments011 
            momentsArray[4] = moments10[iz  ]; // moments100 
            momentsArray[5] = moments10[iz-1]; // moments101 
            momentsArray[6] = moments11[iz  ]; // moments110 
            momentsArray[7] = moments11[iz-1]; // moments111 

            //? Iterate over 10 velocity moments and 8 weights 
            for(int m = 0; m < 10; m++)
                for(int c = 0; c < 8; c++)
                    momentsArray[c][m] += velmoments[m]*weights[c];
        }

        // for (int is = 0; is < ns; is++)
        //     part[is].computeMoments(EMf);

        //? Reduction: reduce moments in parallel
        for(int thread_num = 0; thread_num < get_sizeMomentsArray(); thread_num++)
        {
            arr4_double moments = fetch_moments13Array(thread_num).fetch_arr();
            
            #pragma omp for collapse(2)
            for(int i = 0; i < nxn; i++)
                for(int j = 0; j < nyn; j++)
                    for(int k = 0; k < nzn; k++)
                    {
                        rhons[is][i][j][k] += invVOL*moments[i][j][k][0];
                        Jxhs [is][i][j][k] += moments[i][j][k][1];
                        Jyhs [is][i][j][k] += moments[i][j][k][2];
                        Jzhs [is][i][j][k] += moments[i][j][k][3];
                    }

            // #pragma omp for collapse(2)
            // for (int c = 0; c < NE_MASS; c++) 
            //     for (int i = 0; i < nxn; i++)
            //         for (int j = 0; j < nyn; j++)
            //             for (int k = 0; k < nzn; k++) 
            //             {
            //                 Mxx.fetch(c, i, j, k) += moments[i][j][k][4];
            //                 Mxy.fetch(c, i, j, k) += moments[i][j][k][5];
            //                 Mxz.fetch(c, i, j, k) += moments[i][j][k][6];
            //                 Myx.fetch(c, i, j, k) += moments[i][j][k][7];
            //                 Myy.fetch(c, i, j, k) += moments[i][j][k][8];
            //                 Myz.fetch(c, i, j, k) += moments[i][j][k][9];
            //                 Mzx.fetch(c, i, j, k) += moments[i][j][k][10];
            //                 Mzy.fetch(c, i, j, k) += moments[i][j][k][11];
            //                 Mzz.fetch(c, i, j, k) += moments[i][j][k][12];
            //             }
        }
        
    }
    }

    for (int is = 0; is < ns; is++)
    {
        communicateGhostP2G_ecsim(is);
        // communicateGhostP2G_mass_matrix();
    }

    //* Sum all over the species (mass and charge density)
    // sumOverSpecies();

    //* Communicate average densities
    // for (int is = 0; is < ns; is++)
    //     interpolateCenterSpecies(is);
}

#ifdef __MIC__
    //* add moment weights to all ten moments for the cell of the particle
    //* (assumes that particle data is aligned with cache boundary and begins with the velocity components)
    inline void addto_cell_moments(F64vec8* cell_moments, F64vec8 weights, F64vec8 vel)
    {
        // broadcast particle velocities
        const F64vec8 u = F64vec8(vel[0]);
        const F64vec8 v = F64vec8(vel[1]);
        const F64vec8 w = F64vec8(vel[2]);
        // construct kronecker product of moments and weights
        const F64vec8 u_weights = u*weights;
        const F64vec8 v_weights = v*weights;
        const F64vec8 w_weights = w*weights;
        const F64vec8 uu_weights = u*u_weights;
        const F64vec8 uv_weights = u*v_weights;
        const F64vec8 uw_weights = u*w_weights;
        const F64vec8 vv_weights = v*v_weights;
        const F64vec8 vw_weights = v*w_weights;
        const F64vec8 ww_weights = w*w_weights;
        // add moment weights to accumulated moment weights in mesh mesh
        cell_moments[0] += weights;
        cell_moments[1] += u_weights;
        cell_moments[2] += v_weights;
        cell_moments[3] += w_weights;
        cell_moments[4] += uu_weights;
        cell_moments[5] += uv_weights;
        cell_moments[6] += uw_weights;
        cell_moments[7] += vv_weights;
        cell_moments[8] += vw_weights;
        cell_moments[9] += ww_weights;
    }
#endif // __MIC__

// sum moments of AoS using MIC intrinsics
// 
// We could rewrite this without intrinsics also.  The core idea
// of this algorithm is that instead of scattering the data of
// each particle to its nodes, in each cell we accumulate the
// data that would be scattered and then scatter it at the end.
// By waiting to scatter, with each particle we work with an
// aligned 10x8 matrix rather than a 8x10 matrix, which means
// that for each particle we make 10 vector stores rather than
// 8*2=16 or 8*3=24 vector stores (for unaligned data).  This
// also avoids the expense of computing node indices for each
// particle.
//
// 1. compute vector of 8 weights using position
// 2. form kronecker product of weights with moments
//    by scaling the weights by each velocity moment;
//    add each to accumulated weights for this cell
// 3. after sum is complete, transpose weight-moment
//    product in each cell and distribute to its 8 nodes.
//    An optimized way:
//    A. transpose the first 8 weighted moments with fast 8x8
//       matrix transpose.
//    B. transpose 2x8 matrix of the last two weighted moments
//       and then use 8 masked vector adds to accumulate
//       to weights at nodes.
//    But the optimized way might be overkill since distributing
//    the sums from the cells to the nodes should not dominate
//    if the number of particles per mesh cell is large;
//    if the number of particles per mesh cell is small,
//    then a fully vectorized moment sum is hard to justify anyway.
//
// See notes at the top of sumMoments().
//
void EMfields3D::sumMoments_AoS_intr(const Particles3Dcomm* part)
{
#ifndef __MIC__
  eprintf("not implemented");
#else
  const Grid *grid = &get_grid();

  // define global parameters
  //
  const double inv_dx = 1.0 / dx;
  const double inv_dy = 1.0 / dy;
  const double inv_dz = 1.0 / dz;
  const int nxn = grid->getNXN();
  const int nyn = grid->getNYN();
  const int nzn = grid->getNZN();
  const double xstart = grid->getXstart();
  const double ystart = grid->getYstart();
  const double zstart = grid->getZstart();
  // Here and below x stands for all 3 physical position coordinates
  const F64vec8 dx_inv = make_F64vec8(inv_dx, inv_dy, inv_dz);
  // starting physical position of proper subdomain ("pdom", without ghosts)
  const F64vec8 pdom_xlow = make_F64vec8(xstart,ystart, zstart);
  //
  // X = canonical coordinates.
  //
  // starting position of cell in lower corner
  // of proper subdomain (without ghosts);
  // probably this is an integer value, but we won't rely on it.
  const F64vec8 pdom_Xlow = dx_inv*pdom_xlow;
  // g = including ghosts
  // starting position of cell in low corner
  const F64vec8 gdom_Xlow = pdom_Xlow - F64vec8(1.);
  // starting position of cell in high corner of physical domain
  // in canonical coordinates of ghost domain
  const F64vec8 nXcm1 = make_F64vec8(nxc-1,nyc-1,nzc-1);

  // allocate memory per mesh cell for accumulating moments
  //
  const int num_threads = omp_get_max_threads();
  array4<F64vec8>* cell_moments_per_thr
    = (array4<F64vec8>*) malloc(num_threads*sizeof(array4<F64vec8>));
  for(int thread_num=0;thread_num<num_threads;thread_num++)
  {
    // use placement new to allocate array to accumulate moments for thread
    new(&cell_moments_per_thr[thread_num]) array4<F64vec8>(nxc,nyc,nzc,10);
  }
  //
  // allocate memory per mesh node for accumulating moments
  //
  array3<F64vec8>* node_moments_first8_per_thr
    = (array3<F64vec8>*) malloc(num_threads*sizeof(array3<F64vec8>));
  array4<double>* node_moments_last2_per_thr
    = (array4<double>*) malloc(num_threads*sizeof(array4<double>));
  for(int thread_num=0;thread_num<num_threads;thread_num++)
  {
    // use placement new to allocate array to accumulate moments for thread
    new(&node_moments_first8_per_thr[thread_num]) array3<F64vec8>(nxn,nyn,nzn);
    new(&node_moments_last2_per_thr[thread_num]) array4<double>(nxn,nyn,nzn,2);
  }

  // The moments of a particle must be distributed to the 8 nodes of the cell
  // in proportion to the weight of each node.
  //
  // Refer to the kronecker product of weights and moments as
  // "weighted moments" or "moment weights".
  //
  // Each thread accumulates moment weights in cells.
  //
  // Because particles are not assumed to be sorted by mesh cell,
  // we have to wait until all particles have been processed
  // before we transpose moment weights to weighted moments;
  // the memory that we must allocate to sum moments is thus
  // num_thread*8 times as much as if particles were pre-sorted
  // by mesh cell (and num_threads times as much as if particles
  // were sorted by thread subdomain).
  //
  #pragma omp parallel
  {
    // array4<F64vec8> cell_moments(nxc,nyc,nzc,10);
    const int this_thread = omp_get_thread_num();
    assert_lt(this_thread,num_threads);
    array4<F64vec8>& cell_moments = cell_moments_per_thr[this_thread];

    for (int species_idx = 0; species_idx < ns; species_idx++)
    {
      const Particles3Dcomm& pcls = part[species_idx];
      assert_eq(pcls.get_particleType(), ParticleType::AoS);
      const int is = pcls.get_species_num();
      assert_eq(species_idx,is);

      // moments.setmode(ompmode::mine);
      // moments.setall(0.);
      // 
      F64vec8 *cell_moments1d = &cell_moments[0][0][0][0];
      int moments1dsize = cell_moments.get_size();
      for(int i=0; i<moments1dsize; i++) cell_moments1d[i]=F64vec8(0.);
      //
      // number or particles processed at a time
      const int num_pcls_per_loop = 2;
      const vector_SpeciesParticle& pcl_list = pcls.get_pcl_list();
      const int nop = pcl_list.size();
      // if the number of particles is odd, then make
      // sure that the data after the last particle
      // will not contribute to the moments.
      #pragma omp single // the implied omp barrier is needed
      {
        // make sure that we will not overrun the array
        assert_divides(num_pcls_per_loop,pcl_list.capacity());
        // round up number of particles
        int nop_rounded_up = roundup_to_multiple(nop,num_pcls_per_loop);
        for(int pidx=nop; pidx<nop_rounded_up; pidx++)
        {
          // (This is a benign violation of particle
          // encapsulation and requires a cast).
          SpeciesParticle& pcl = (SpeciesParticle&) pcl_list[pidx];
          pcl.set_to_zero();
        }
      }
      #pragma omp for
      for (int pidx = 0; pidx < nop; pidx+=2)
      {
        // cast particles as vectors
        // (assumes each particle exactly fits a cache line)
        const F64vec8& pcl0 = (const F64vec8&)pcl_list[pidx];
        const F64vec8& pcl1 = (const F64vec8&)pcl_list[pidx+1];
        // gather position data from particles
        // (assumes position vectors are in upper half)
        const F64vec8 xpos = cat_hgh_halves(pcl0,pcl1);

        // convert to canonical coordinates relative to subdomain with ghosts
        const F64vec8 gX = dx_inv*xpos - gdom_Xlow;
        F64vec8 cellXstart = floor(gX);
        // all particles at this point should be inside the
        // proper subdomain of this process, but maybe we
        // will need to enforce this because of inconsistency
        // of floating point arithmetic?
        //cellXstart = maximum(cellXstart,F64vec8(1.));
        //cellXstart = minimum(cellXstart,nXcm1);
        assert(!test_lt(cellXstart,F64vec8(1.)));
        assert(!test_gt(cellXstart,nXcm1));

        // get weights for field_components based on particle position
        //
        F64vec8 weights[2];
        const F64vec8 X = gX - cellXstart;
        construct_weights_for_2pcls(weights, X);

        // add scaled weights to all ten moments for the cell of each particle
        //
        // the cell that we will write to
        const I32vec16 cell = round_to_nearest(cellXstart);
        const int* c=(int*)&cell;
        F64vec8* cell_moments0 = &cell_moments[c[0]][c[1]][c[2]][0];
        F64vec8* cell_moments1 = &cell_moments[c[4]][c[5]][c[6]][0];
        addto_cell_moments(cell_moments0, weights[0], pcl0);
        addto_cell_moments(cell_moments1, weights[1], pcl1);
      }
      if(!this_thread) timeTasks_end_task(TimeTasks::MOMENT_ACCUMULATION);

      // reduction
      if(!this_thread) timeTasks_begin_task(TimeTasks::MOMENT_REDUCTION);

      // reduce moments in parallel
      //
      // this code currently makes no sense for multiple threads.
      assert_eq(num_threads,1);
      {
        // For each thread, distribute moments from cells to nodes
        // and then sum moments at each node over all threads.
        //
        // (Alternatively we could sum over all threads and then
        // distribute to nodes; this alternative would be preferable
        // for vectorization efficiency but more difficult to parallelize
        // across threads).

        // initialize moment accumulators
        //
        memset(&node_moments_first8_per_thr[this_thread][0][0][0],
          0, sizeof(F64vec8)*node_moments_first8_per_thr[0].get_size());
        memset(&node_moments_last2_per_thr[this_thread][0][0][0][0],
          0, sizeof(double)*node_moments_last2_per_thr[0].get_size());

        // distribute moments from cells to nodes
        //
        #pragma omp for collapse(2)
        for(int cx=1;cx<nxc;cx++)
        for(int cy=1;cy<nyc;cy++)
        for(int cz=1;cz<nzc;cz++)
        {
          const int ix=cx+1;
          const int iy=cy+1;
          const int iz=cz+1;
          F64vec8* cell_mom = &cell_moments[cx][cy][cz][0];

          // scatter the cell's first 8 moments to its nodes
          // for each thread
          {
            F64vec8* cell_mom_first8 = cell_mom;
            // regard cell_mom_first8 as a pointer to 8x8 data and transpose
            transpose_8x8_double((double(*)[8]) cell_mom_first8);
            // scatter the moment vectors to the nodes
            array3<F64vec8>& node_moments_first8 = node_moments_first8_per_thr[this_thread];
            arr_fetch2(F64vec8) node_moments0 = node_moments_first8[ix];
            arr_fetch2(F64vec8) node_moments1 = node_moments_first8[cx];
            arr_fetch1(F64vec8) node_moments00 = node_moments0[iy];
            arr_fetch1(F64vec8) node_moments01 = node_moments0[cy];
            arr_fetch1(F64vec8) node_moments10 = node_moments1[iy];
            arr_fetch1(F64vec8) node_moments11 = node_moments1[cy];
            node_moments00[iz] += cell_mom_first8[0]; // node_moments_first8[ix][iy][iz]
            node_moments00[cz] += cell_mom_first8[1]; // node_moments_first8[ix][iy][cz]
            node_moments01[iz] += cell_mom_first8[2]; // node_moments_first8[ix][cy][iz]
            node_moments01[cz] += cell_mom_first8[3]; // node_moments_first8[ix][cy][cz]
            node_moments10[iz] += cell_mom_first8[4]; // node_moments_first8[cx][iy][iz]
            node_moments10[cz] += cell_mom_first8[5]; // node_moments_first8[cx][iy][cz]
            node_moments11[iz] += cell_mom_first8[6]; // node_moments_first8[cx][cy][iz]
            node_moments11[cz] += cell_mom_first8[7]; // node_moments_first8[cx][cy][cz]
          }

          // scatter the cell's last 2 moments to its nodes
          {
            array4<double>& node_moments_last2 = node_moments_last2_per_thr[this_thread];
            arr3_double_fetch node_moments0 = node_moments_last2[ix];
            arr3_double_fetch node_moments1 = node_moments_last2[cx];
            arr2_double_fetch node_moments00 = node_moments0[iy];
            arr2_double_fetch node_moments01 = node_moments0[cy];
            arr2_double_fetch node_moments10 = node_moments1[iy];
            arr2_double_fetch node_moments11 = node_moments1[cy];
            double* node_moments000 = node_moments00[iz];
            double* node_moments001 = node_moments00[cz];
            double* node_moments010 = node_moments01[iz];
            double* node_moments011 = node_moments01[cz];
            double* node_moments100 = node_moments10[iz];
            double* node_moments101 = node_moments10[cz];
            double* node_moments110 = node_moments11[iz];
            double* node_moments111 = node_moments11[cz];

            const F64vec8 mom8 = cell_mom[8];
            const F64vec8 mom9 = cell_mom[9];

            bool naive_last2 = true;
            if(naive_last2)
            {
              node_moments000[0] += mom8[0]; node_moments000[1] += mom9[0];
              node_moments001[0] += mom8[1]; node_moments001[1] += mom9[1];
              node_moments010[0] += mom8[2]; node_moments010[1] += mom9[2];
              node_moments011[0] += mom8[3]; node_moments011[1] += mom9[3];
              node_moments100[0] += mom8[4]; node_moments100[1] += mom9[4];
              node_moments101[0] += mom8[5]; node_moments101[1] += mom9[5];
              node_moments110[0] += mom8[6]; node_moments110[1] += mom9[6];
              node_moments111[0] += mom8[7]; node_moments111[1] += mom9[7];
            }
            else
            {
              // Let a=moment#8 and b=moment#9.
              // Number the nodes 0 through 7.
              //
              // This transpose changes data from the form
              //   [a0 a1 a2 a3 a4 a5 a6 a7]=mom8
              //   [b0 b1 b2 b3 b4 b5 b6 b7]=mom9
              // into the form
              //   [a0 b0 a2 b2 a4 b4 a6 b6]=out8
              //   [a1 b1 a3 b3 a5 b5 a7 b7]=out9
              F64vec8 out8, out9;
              trans2x2(out8, out9, mom8, mom9);

              // probably the compiler is not smart enough to recognize that
              // each line can be done with a single vector instruction:
              node_moments000[0] += out8[0]; node_moments000[1] += out8[1];
              node_moments001[0] += out9[0]; node_moments001[1] += out9[1];
              node_moments010[0] += out8[2]; node_moments010[1] += out8[3];
              node_moments011[0] += out9[2]; node_moments011[1] += out9[3];
              node_moments100[0] += out8[4]; node_moments100[1] += out8[5];
              node_moments101[0] += out9[4]; node_moments101[1] += out9[5];
              node_moments110[0] += out8[6]; node_moments110[1] += out8[7];
              node_moments111[0] += out9[6]; node_moments111[1] += out9[7];
            }
          }
        }

        // at each node add moments to moments of first thread
        //
        #pragma omp for collapse(2)
        for(int nx=1;nx<nxn;nx++)
        for(int ny=1;ny<nyn;ny++)
        {
          arr_fetch1(F64vec8) node_moments8_for_master
            = node_moments_first8_per_thr[0][nx][ny];
          arr_fetch2(double) node_moments2_for_master
            = node_moments_last2_per_thr[0][nx][ny];
          for(int thread_num=1;thread_num<num_threads;thread_num++)
          {
            arr_fetch1(F64vec8) node_moments8_for_thr
              = node_moments_first8_per_thr[thread_num][nx][ny];
            arr_fetch2(double) node_moments2_for_thr
              = node_moments_last2_per_thr[thread_num][nx][ny];
            for(int nz=1;nz<nzn;nz++)
            {
              node_moments8_for_master[nz] += node_moments8_for_thr[nz];
              node_moments2_for_master[nz][0] += node_moments2_for_thr[nz][0];
              node_moments2_for_master[nz][1] += node_moments2_for_thr[nz][1];
            }
          }
        }

        // transpose moments for field solver
        //
        #pragma omp for collapse(2)
        for(int nx=1;nx<nxn;nx++)
        for(int ny=1;ny<nyn;ny++)
        {
          arr_fetch1(F64vec8) node_moments8_for_master
            = node_moments_first8_per_thr[0][nx][ny];
          arr_fetch2(double) node_moments2_for_master
            = node_moments_last2_per_thr[0][nx][ny];
          arr_fetch1(double) rho_sxy = rhons[is][nx][ny];
          arr_fetch1(double) Jx__sxy = Jxs  [is][nx][ny];
          arr_fetch1(double) Jy__sxy = Jys  [is][nx][ny];
          arr_fetch1(double) Jz__sxy = Jzs  [is][nx][ny];
          arr_fetch1(double) pXX_sxy = pXXsn[is][nx][ny];
          arr_fetch1(double) pXY_sxy = pXYsn[is][nx][ny];
          arr_fetch1(double) pXZ_sxy = pXZsn[is][nx][ny];
          arr_fetch1(double) pYY_sxy = pYYsn[is][nx][ny];
          arr_fetch1(double) pYZ_sxy = pYZsn[is][nx][ny];
          arr_fetch1(double) pZZ_sxy = pZZsn[is][nx][ny];
          for(int nz=0;nz<nzn;nz++)
          {
            rho_sxy[nz] = invVOL*node_moments8_for_master[nz][0];
            Jx__sxy[nz] = invVOL*node_moments8_for_master[nz][1];
            Jy__sxy[nz] = invVOL*node_moments8_for_master[nz][2];
            Jz__sxy[nz] = invVOL*node_moments8_for_master[nz][3];
            pXX_sxy[nz] = invVOL*node_moments8_for_master[nz][4];
            pXY_sxy[nz] = invVOL*node_moments8_for_master[nz][5];
            pXZ_sxy[nz] = invVOL*node_moments8_for_master[nz][6];
            pYY_sxy[nz] = invVOL*node_moments8_for_master[nz][7];
            pYZ_sxy[nz] = invVOL*node_moments2_for_master[nz][0];
            pZZ_sxy[nz] = invVOL*node_moments2_for_master[nz][1];
          }
        }
      }
      if(!this_thread) timeTasks_end_task(TimeTasks::MOMENT_REDUCTION);
    }
  }

  // deallocate memory per mesh node for accumulating moments
  //
  for(int thread_num=0;thread_num<num_threads;thread_num++)
  {
    // call destructor to deallocate arrays
    node_moments_first8_per_thr[thread_num].~array3<F64vec8>();
    node_moments_last2_per_thr[thread_num].~array4<double>();
  }
  free(node_moments_first8_per_thr);
  free(node_moments_last2_per_thr);

  // deallocate memory for accumulating moments
  //
  for(int thread_num=0;thread_num<num_threads;thread_num++)
  {
    // deallocate array to accumulate moments for thread
    cell_moments_per_thr[thread_num].~array4<F64vec8>();
  }
  free(cell_moments_per_thr);

  for (int i = 0; i < ns; i++)
  {
    communicateGhostP2G(i);
  }
#endif // __MIC__
}

inline void compute_moments(double velmoments[10], double weights[8],
                            int i,
                            double const * const x,
                            double const * const y,
                            double const * const z,
                            double const * const u,
                            double const * const v,
                            double const * const w,
                            double const * const q,
                            double xstart,
                            double ystart,
                            double zstart,
                            double inv_dx,
                            double inv_dy,
                            double inv_dz,
                            int cx,
                            int cy,
                            int cz)
{
    ALIGNED(x);
    ALIGNED(y);
    ALIGNED(z);
    ALIGNED(u);
    ALIGNED(v);
    ALIGNED(w);
    ALIGNED(q);
    // compute the quadratic moments of velocity
    //
    const double ui=u[i];
    const double vi=v[i];
    const double wi=w[i];
    const double uui=ui*ui;
    const double uvi=ui*vi;
    const double uwi=ui*wi;
    const double vvi=vi*vi;
    const double vwi=vi*wi;
    const double wwi=wi*wi;
    //double velmoments[10];
    velmoments[0] = 1.;
    velmoments[1] = ui;
    velmoments[2] = vi;
    velmoments[3] = wi;
    velmoments[4] = uui;
    velmoments[5] = uvi;
    velmoments[6] = uwi;
    velmoments[7] = vvi;
    velmoments[8] = vwi;
    velmoments[9] = wwi;

    // compute the weights to distribute the moments
    //
    //double weights[8];
    const double abs_xpos = x[i];
    const double abs_ypos = y[i];
    const double abs_zpos = z[i];
    const double rel_xpos = abs_xpos - xstart;
    const double rel_ypos = abs_ypos - ystart;
    const double rel_zpos = abs_zpos - zstart;
    const double cxm1_pos = rel_xpos * inv_dx;
    const double cym1_pos = rel_ypos * inv_dy;
    const double czm1_pos = rel_zpos * inv_dz;
    //if(true)
    //{
    //  const int cx_inf = int(floor(cxm1_pos));
    //  const int cy_inf = int(floor(cym1_pos));
    //  const int cz_inf = int(floor(czm1_pos));
    //  assert_eq(cx-1,cx_inf);
    //  assert_eq(cy-1,cy_inf);
    //  assert_eq(cz-1,cz_inf);
    //}
    // fraction of the distance from the right of the cell
    const double w1x = cx - cxm1_pos;
    const double w1y = cy - cym1_pos;
    const double w1z = cz - czm1_pos;
    // fraction of distance from the left
    const double w0x = 1-w1x;
    const double w0y = 1-w1y;
    const double w0z = 1-w1z;
    // we are calculating a charge moment.
    const double qi=q[i];
    const double weight0 = qi*w0x;
    const double weight1 = qi*w1x;
    const double weight00 = weight0*w0y;
    const double weight01 = weight0*w1y;
    const double weight10 = weight1*w0y;
    const double weight11 = weight1*w1y;
    weights[0] = weight00*w0z; // weight000
    weights[1] = weight00*w1z; // weight001
    weights[2] = weight01*w0z; // weight010
    weights[3] = weight01*w1z; // weight011
    weights[4] = weight10*w0z; // weight100
    weights[5] = weight10*w1z; // weight101
    weights[6] = weight11*w0z; // weight110
    weights[7] = weight11*w1z; // weight111
}

//? Add particle to moments
inline void add_moments_for_pcl(double momentsAcc[8][10],
                                int i,
                                double const * const x,
                                double const * const y,
                                double const * const z,
                                double const * const u,
                                double const * const v,
                                double const * const w,
                                double const * const q,
                                double xstart,
                                double ystart,
                                double zstart,
                                double inv_dx,
                                double inv_dy,
                                double inv_dz,
                                int cx,
                                int cy,
                                int cz)
{
    double velmoments[10];
    double weights[8];
    
    compute_moments(velmoments, weights, i, x, y, z, u, v, w, q,
    xstart, ystart, zstart, inv_dx, inv_dy, inv_dz, cx, cy, cz);

    for(int c=0; c<8; c++)
        for(int m=0; m<10; m++)
            momentsAcc[c][m] += velmoments[m]*weights[c];
}


//? Vectorized version of adding particle to moments
inline void add_moments_for_pcl_vec(double momentsAccVec[8][10][8],
                                    double velmoments[10][8], double weights[8][8],
                                    int i,
                                    int imod,
                                    double const * const x,
                                    double const * const y,
                                    double const * const z,
                                    double const * const u,
                                    double const * const v,
                                    double const * const w,
                                    double const * const q,
                                    double xstart,
                                    double ystart,
                                    double zstart,
                                    double inv_dx,
                                    double inv_dy,
                                    double inv_dz,
                                    int cx,
                                    int cy,
                                    int cz)
{
  ALIGNED(x);
  ALIGNED(y);
  ALIGNED(z);
  ALIGNED(u);
  ALIGNED(v);
  ALIGNED(w);
  ALIGNED(q);
  // compute the quadratic moments of velocity
  //
  const double ui=u[i];
  const double vi=v[i];
  const double wi=w[i];
  const double uui=ui*ui;
  const double uvi=ui*vi;
  const double uwi=ui*wi;
  const double vvi=vi*vi;
  const double vwi=vi*wi;
  const double wwi=wi*wi;
  //double velmoments[10];
  velmoments[0][imod] = 1.;
  velmoments[1][imod] = ui;
  velmoments[2][imod] = vi;
  velmoments[3][imod] = wi;
  velmoments[4][imod] = uui;
  velmoments[5][imod] = uvi;
  velmoments[6][imod] = uwi;
  velmoments[7][imod] = vvi;
  velmoments[8][imod] = vwi;
  velmoments[9][imod] = wwi;

  // compute the weights to distribute the moments
  //
  //double weights[8];
  const double abs_xpos = x[i];
  const double abs_ypos = y[i];
  const double abs_zpos = z[i];
  const double rel_xpos = abs_xpos - xstart;
  const double rel_ypos = abs_ypos - ystart;
  const double rel_zpos = abs_zpos - zstart;
  const double cxm1_pos = rel_xpos * inv_dx;
  const double cym1_pos = rel_ypos * inv_dy;
  const double czm1_pos = rel_zpos * inv_dz;
  //if(true)
  //{
  //  const int cx_inf = int(floor(cxm1_pos));
  //  const int cy_inf = int(floor(cym1_pos));
  //  const int cz_inf = int(floor(czm1_pos));
  //  assert_eq(cx-1,cx_inf);
  //  assert_eq(cy-1,cy_inf);
  //  assert_eq(cz-1,cz_inf);
  //}
  // fraction of the distance from the right of the cell
  const double w1x = cx - cxm1_pos;
  const double w1y = cy - cym1_pos;
  const double w1z = cz - czm1_pos;
  // fraction of distance from the left
  const double w0x = 1-w1x;
  const double w0y = 1-w1y;
  const double w0z = 1-w1z;
  // we are calculating a charge moment.
  const double qi=q[i];
  const double weight0 = qi*w0x;
  const double weight1 = qi*w1x;
  const double weight00 = weight0*w0y;
  const double weight01 = weight0*w1y;
  const double weight10 = weight1*w0y;
  const double weight11 = weight1*w1y;
  weights[0][imod] = weight00*w0z; // weight000
  weights[1][imod] = weight00*w1z; // weight001
  weights[2][imod] = weight01*w0z; // weight010
  weights[3][imod] = weight01*w1z; // weight011
  weights[4][imod] = weight10*w0z; // weight100
  weights[5][imod] = weight10*w1z; // weight101
  weights[6][imod] = weight11*w0z; // weight110
  weights[7][imod] = weight11*w1z; // weight111

  // add moments for this particle
  {
    for(int c=0; c<8; c++)
    for(int m=0; m<10; m++)
    {
      momentsAccVec[c][m][imod] += velmoments[m][imod]*weights[c][imod];
    }
  }
}

void EMfields3D::sumMoments_vectorized(const Particles3Dcomm* part)
{
  const Grid *grid = &get_grid();

  const double inv_dx = grid->get_invdx();
  const double inv_dy = grid->get_invdy();
  const double inv_dz = grid->get_invdz();
  const int nxn = grid->getNXN();
  const int nyn = grid->getNYN();
  const int nzn = grid->getNZN();
  const double xstart = grid->getXstart();
  const double ystart = grid->getYstart();
  const double zstart = grid->getZstart();
  #pragma omp parallel
  {
  for (int species_idx = 0; species_idx < ns; species_idx++)
  {
    const Particles3Dcomm& pcls = part[species_idx];
    assert_eq(pcls.get_particleType(), ParticleType::SoA);
    const int is = pcls.get_species_num();
    assert_eq(species_idx,is);

    double const*const x = pcls.getXall();
    double const*const y = pcls.getYall();
    double const*const z = pcls.getZall();
    double const*const u = pcls.getUall();
    double const*const v = pcls.getVall();
    double const*const w = pcls.getWall();
    double const*const q = pcls.getQall();

    const int nop = pcls.getNOP();
    #pragma omp master
    { timeTasks_begin_task(TimeTasks::MOMENT_ACCUMULATION); }
    Moments10& speciesMoments10 = fetch_moments10Array(0);
    arr4_double moments = speciesMoments10.fetch_arr();
    //
    // moments.setmode(ompmode::ompfor);
    //moments.setall(0.);
    double *moments1d = &moments[0][0][0][0];
    int moments1dsize = moments.get_size();
    #pragma omp for // because shared
    for(int i=0; i<moments1dsize; i++) moments1d[i]=0;
    
    // prevent threads from writing to the same location
    for(int cxmod2=0; cxmod2<2; cxmod2++)
    for(int cymod2=0; cymod2<2; cymod2++)
    // each mesh cell is handled by its own thread
    #pragma omp for collapse(2)
    for(int cx=cxmod2;cx<nxc;cx+=2)
    for(int cy=cymod2;cy<nyc;cy+=2)
    for(int cz=0;cz<nzc;cz++)
    {
     //dprint(cz);
     // index of interface to right of cell
     const int ix = cx + 1;
     const int iy = cy + 1;
     const int iz = cz + 1;
     {
      // reference the 8 nodes to which we will
      // write moment data for particles in this mesh cell.
      //
      arr1_double_fetch momentsArray[8];
      arr2_double_fetch moments00 = moments[ix][iy];
      arr2_double_fetch moments01 = moments[ix][cy];
      arr2_double_fetch moments10 = moments[cx][iy];
      arr2_double_fetch moments11 = moments[cx][cy];
      momentsArray[0] = moments00[iz]; // moments000 
      momentsArray[1] = moments00[cz]; // moments001 
      momentsArray[2] = moments01[iz]; // moments010 
      momentsArray[3] = moments01[cz]; // moments011 
      momentsArray[4] = moments10[iz]; // moments100 
      momentsArray[5] = moments10[cz]; // moments101 
      momentsArray[6] = moments11[iz]; // moments110 
      momentsArray[7] = moments11[cz]; // moments111 

      const int numpcls_in_cell = pcls.get_numpcls_in_bucket(cx,cy,cz);
      const int bucket_offset = pcls.get_bucket_offset(cx,cy,cz);
      const int bucket_end = bucket_offset+numpcls_in_cell;

      bool vectorized=false;
      if(!vectorized)
      {
        // accumulators for moments per each of 8 threads
        double momentsAcc[8][10];
        memset(momentsAcc,0,sizeof(double)*8*10);
        for(int i=bucket_offset; i<bucket_end; i++)
        {
          add_moments_for_pcl(momentsAcc, i,
            x, y, z, u, v, w, q,
            xstart, ystart, zstart,
            inv_dx, inv_dy, inv_dz,
            cx, cy, cz);
        }
        for(int c=0; c<8; c++)
        for(int m=0; m<10; m++)
        {
          momentsArray[c][m] += momentsAcc[c][m];
        }
      }
      if(vectorized)
      {
        double velmoments[10][8];
        double weights[8][8];
        double momentsAccVec[8][10][8];
        memset(momentsAccVec,0,sizeof(double)*8*10*8);
        #pragma simd
        for(int i=bucket_offset; i<bucket_end; i++)
        {
          add_moments_for_pcl_vec(momentsAccVec, velmoments, weights,
            i, i%8,
            x, y, z, u, v, w, q,
            xstart, ystart, zstart,
            inv_dx, inv_dy, inv_dz,
            cx, cy, cz);
        }
        for(int c=0; c<8; c++)
        for(int m=0; m<10; m++)
        for(int i=0; i<8; i++)
        {
          momentsArray[c][m] += momentsAccVec[c][m][i];
        }
      }
     }
    }
    #pragma omp master
    { timeTasks_end_task(TimeTasks::MOMENT_ACCUMULATION); }

    // reduction
    #pragma omp master
    { timeTasks_begin_task(TimeTasks::MOMENT_REDUCTION); }
    {
      #pragma omp for collapse(2)
      for(int i=0;i<nxn;i++){
      for(int j=0;j<nyn;j++){
      for(int k=0;k<nzn;k++)
      {
        rhons[is][i][j][k] = invVOL*moments[i][j][k][0];
        Jxs  [is][i][j][k] = invVOL*moments[i][j][k][1];
        Jys  [is][i][j][k] = invVOL*moments[i][j][k][2];
        Jzs  [is][i][j][k] = invVOL*moments[i][j][k][3];
        pXXsn[is][i][j][k] = invVOL*moments[i][j][k][4];
        pXYsn[is][i][j][k] = invVOL*moments[i][j][k][5];
        pXZsn[is][i][j][k] = invVOL*moments[i][j][k][6];
        pYYsn[is][i][j][k] = invVOL*moments[i][j][k][7];
        pYZsn[is][i][j][k] = invVOL*moments[i][j][k][8];
        pZZsn[is][i][j][k] = invVOL*moments[i][j][k][9];
      }}}
    }
    #pragma omp master
    { timeTasks_end_task(TimeTasks::MOMENT_REDUCTION); }
    // uncomment this and remove the loop below
    // when we change to use asynchronous communication.
    // communicateGhostP2G(is);
  }
  }
  for (int i = 0; i < ns; i++)
  {
    communicateGhostP2G(i);
  }
}

void EMfields3D::sumMoments_vectorized_AoS(const Particles3Dcomm* part)
{
  const Grid *grid = &get_grid();

  const double inv_dx = grid->get_invdx();
  const double inv_dy = grid->get_invdy();
  const double inv_dz = grid->get_invdz();
  const int nxn = grid->getNXN();
  const int nyn = grid->getNYN();
  const int nzn = grid->getNZN();
  const double xstart = grid->getXstart();
  const double ystart = grid->getYstart();
  const double zstart = grid->getZstart();
  #pragma omp parallel
  {
  for (int species_idx = 0; species_idx < ns; species_idx++)
  {
    const Particles3Dcomm& pcls = part[species_idx];
    assert_eq(pcls.get_particleType(), ParticleType::AoS);
    const int is = pcls.get_species_num();
    assert_eq(species_idx,is);

    const int nop = pcls.getNOP();
    #pragma omp master
    { timeTasks_begin_task(TimeTasks::MOMENT_ACCUMULATION); }
    Moments10& speciesMoments10 = fetch_moments10Array(0);
    arr4_double moments = speciesMoments10.fetch_arr();
    //
    // moments.setmode(ompmode::ompfor);
    //moments.setall(0.);
    double *moments1d = &moments[0][0][0][0];
    int moments1dsize = moments.get_size();
    #pragma omp for // because shared
    for(int i=0; i<moments1dsize; i++) moments1d[i]=0;
    
    // prevent threads from writing to the same location
    for(int cxmod2=0; cxmod2<2; cxmod2++)
    for(int cymod2=0; cymod2<2; cymod2++)
    // each mesh cell is handled by its own thread
    #pragma omp for collapse(2)
    for(int cx=cxmod2;cx<nxc;cx+=2)
    for(int cy=cymod2;cy<nyc;cy+=2)
    for(int cz=0;cz<nzc;cz++)
    {
     //dprint(cz);
     // index of interface to right of cell
     const int ix = cx + 1;
     const int iy = cy + 1;
     const int iz = cz + 1;
     {
      // reference the 8 nodes to which we will
      // write moment data for particles in this mesh cell.
      //
      arr1_double_fetch momentsArray[8];
      arr2_double_fetch moments00 = moments[ix][iy];
      arr2_double_fetch moments01 = moments[ix][cy];
      arr2_double_fetch moments10 = moments[cx][iy];
      arr2_double_fetch moments11 = moments[cx][cy];
      momentsArray[0] = moments00[iz]; // moments000 
      momentsArray[1] = moments00[cz]; // moments001 
      momentsArray[2] = moments01[iz]; // moments010 
      momentsArray[3] = moments01[cz]; // moments011 
      momentsArray[4] = moments10[iz]; // moments100 
      momentsArray[5] = moments10[cz]; // moments101 
      momentsArray[6] = moments11[iz]; // moments110 
      momentsArray[7] = moments11[cz]; // moments111 

      // accumulator for moments per each of 8 threads
      double momentsAcc[8][10][8];
      const int numpcls_in_cell = pcls.get_numpcls_in_bucket(cx,cy,cz);
      const int bucket_offset = pcls.get_bucket_offset(cx,cy,cz);
      const int bucket_end = bucket_offset+numpcls_in_cell;

      // data is not stride-1, so we do *not* use
      // #pragma simd
      {
        // accumulators for moments per each of 8 threads
        double momentsAcc[8][10];
        memset(momentsAcc,0,sizeof(double)*8*10);
        for(int pidx=bucket_offset; pidx<bucket_end; pidx++)
        {
          const SpeciesParticle* pcl = &pcls.get_pcl(pidx);
          // This depends on the fact that the memory
          // occupied by a particle coincides with
          // the alignment interval (64 bytes)
          ALIGNED(pcl);
          double velmoments[10];
          double weights[8];
          // compute the quadratic moments of velocity
          //
          const double ui=pcl->get_u();
          const double vi=pcl->get_v();
          const double wi=pcl->get_w();
          const double uui=ui*ui;
          const double uvi=ui*vi;
          const double uwi=ui*wi;
          const double vvi=vi*vi;
          const double vwi=vi*wi;
          const double wwi=wi*wi;
          //double velmoments[10];
          velmoments[0] = 1.;
          velmoments[1] = ui;
          velmoments[2] = vi;
          velmoments[3] = wi;
          velmoments[4] = uui;
          velmoments[5] = uvi;
          velmoments[6] = uwi;
          velmoments[7] = vvi;
          velmoments[8] = vwi;
          velmoments[9] = wwi;
        
          // compute the weights to distribute the moments
          //
          //double weights[8];
          const double abs_xpos = pcl->get_x();
          const double abs_ypos = pcl->get_y();
          const double abs_zpos = pcl->get_z();
          const double rel_xpos = abs_xpos - xstart;
          const double rel_ypos = abs_ypos - ystart;
          const double rel_zpos = abs_zpos - zstart;
          const double cxm1_pos = rel_xpos * inv_dx;
          const double cym1_pos = rel_ypos * inv_dy;
          const double czm1_pos = rel_zpos * inv_dz;
          //if(true)
          //{
          //  const int cx_inf = int(floor(cxm1_pos));
          //  const int cy_inf = int(floor(cym1_pos));
          //  const int cz_inf = int(floor(czm1_pos));
          //  assert_eq(cx-1,cx_inf);
          //  assert_eq(cy-1,cy_inf);
          //  assert_eq(cz-1,cz_inf);
          //}
          // fraction of the distance from the right of the cell
          const double w1x = cx - cxm1_pos;
          const double w1y = cy - cym1_pos;
          const double w1z = cz - czm1_pos;
          // fraction of distance from the left
          const double w0x = 1-w1x;
          const double w0y = 1-w1y;
          const double w0z = 1-w1z;
          // we are calculating a charge moment.
          const double qi=pcl->get_q();
          const double weight0 = qi*w0x;
          const double weight1 = qi*w1x;
          const double weight00 = weight0*w0y;
          const double weight01 = weight0*w1y;
          const double weight10 = weight1*w0y;
          const double weight11 = weight1*w1y;
          weights[0] = weight00*w0z; // weight000
          weights[1] = weight00*w1z; // weight001
          weights[2] = weight01*w0z; // weight010
          weights[3] = weight01*w1z; // weight011
          weights[4] = weight10*w0z; // weight100
          weights[5] = weight10*w1z; // weight101
          weights[6] = weight11*w0z; // weight110
          weights[7] = weight11*w1z; // weight111
        
          // add moments for this particle
          {
            // which is the superior order for the following loop?
            for(int c=0; c<8; c++)
            for(int m=0; m<10; m++)
            {
              momentsAcc[c][m] += velmoments[m]*weights[c];
            }
          }
        }
        for(int c=0; c<8; c++)
        for(int m=0; m<10; m++)
        {
          momentsArray[c][m] += momentsAcc[c][m];
        }
      }
     }
    }
    #pragma omp master
    { timeTasks_end_task(TimeTasks::MOMENT_ACCUMULATION); }

    // reduction
    #pragma omp master
    { timeTasks_begin_task(TimeTasks::MOMENT_REDUCTION); }
    {
      #pragma omp for collapse(2)
      for(int i=0;i<nxn;i++){
      for(int j=0;j<nyn;j++){
      for(int k=0;k<nzn;k++)
      {
        rhons[is][i][j][k] = invVOL*moments[i][j][k][0];
        Jxs  [is][i][j][k] = invVOL*moments[i][j][k][1];
        Jys  [is][i][j][k] = invVOL*moments[i][j][k][2];
        Jzs  [is][i][j][k] = invVOL*moments[i][j][k][3];
        pXXsn[is][i][j][k] = invVOL*moments[i][j][k][4];
        pXYsn[is][i][j][k] = invVOL*moments[i][j][k][5];
        pXZsn[is][i][j][k] = invVOL*moments[i][j][k][6];
        pYYsn[is][i][j][k] = invVOL*moments[i][j][k][7];
        pYZsn[is][i][j][k] = invVOL*moments[i][j][k][8];
        pZZsn[is][i][j][k] = invVOL*moments[i][j][k][9];
      }}}
    }
    #pragma omp master
    { timeTasks_end_task(TimeTasks::MOMENT_REDUCTION); }
    // uncomment this and remove the loop below
    // when we change to use asynchronous communication.
    // communicateGhostP2G(is);
  }
  }
  for (int i = 0; i < ns; i++)
  {
    communicateGhostP2G(i);
  }
}

//* Compute the product of mass matrix with vector "V = (Vx, Vy, Vz)"
void EMfields3D::mass_matrix_times_vector(double* MEx, double* MEy, double* MEz, const_arr3_double vectX, const_arr3_double vectY, const_arr3_double vectZ, int i, int j, int k)
{
    double resX = 0.0; double resY = 0.0; double resZ = 0.0;

    //* Center (g = 0)
    resX  = vectX[i][j][k]*Mxx[0][i][j][k] + vectY[i][j][k]*Myx[0][i][j][k] + vectZ[i][j][k]*Mzx[0][i][j][k];
    resY  = vectX[i][j][k]*Mxy[0][i][j][k] + vectY[i][j][k]*Myy[0][i][j][k] + vectZ[i][j][k]*Mzy[0][i][j][k];
    resZ  = vectX[i][j][k]*Mxz[0][i][j][k] + vectY[i][j][k]*Myz[0][i][j][k] + vectZ[i][j][k]*Mzz[0][i][j][k];

    //* Neighbours
    for (int g = 1; g < NE_MASS; g++) 
    {
        int i1 = i + NeNo.getX(g);
        int j1 = j + NeNo.getY(g);
        int k1 = k + NeNo.getZ(g);

        resX += vectX[i1][j1][k1]*Mxx[g][i][j][k] + vectY[i1][j1][k1]*Myx[g][i][j][k] + vectZ[i1][j1][k1]*Mzx[g][i][j][k];
        resY += vectX[i1][j1][k1]*Mxy[g][i][j][k] + vectY[i1][j1][k1]*Myy[g][i][j][k] + vectZ[i1][j1][k1]*Mzy[g][i][j][k];
        resZ += vectX[i1][j1][k1]*Mxz[g][i][j][k] + vectY[i1][j1][k1]*Myz[g][i][j][k] + vectZ[i1][j1][k1]*Mzz[g][i][j][k];

        // if (g == 0)
        //     continue;
        
        int i2 = i - NeNo.getX(g);
        int j2 = j - NeNo.getY(g);
        int k2 = k - NeNo.getZ(g);

        resX += vectX[i2][j2][k2]*Mxx[g][i2][j2][k2] + vectY[i2][j2][k2]*Myx[g][i2][j2][k2] + vectZ[i2][j2][k2]*Mzx[g][i2][j2][k2];
        resY += vectX[i2][j2][k2]*Mxy[g][i2][j2][k2] + vectY[i2][j2][k2]*Myy[g][i2][j2][k2] + vectZ[i2][j2][k2]*Mzy[g][i2][j2][k2];
        resZ += vectX[i2][j2][k2]*Mxz[g][i2][j2][k2] + vectY[i2][j2][k2]*Myz[g][i2][j2][k2] + vectZ[i2][j2][k2]*Mzz[g][i2][j2][k2];
    }

    *MEx = resX;
    *MEy = resY;
    *MEz = resZ;
}

//! Communicate ghost data for IMM moments
void EMfields3D::communicateGhostP2G(int ns)
{
    //* interpolate adding common nodes among processors
    timeTasks_set_communicating();

    const VirtualTopology3D *vct = &get_vct();

    double ***moment0 = convert_to_arr3(rhons[ns]);
    double ***moment1 = convert_to_arr3(Jxs  [ns]);
    double ***moment2 = convert_to_arr3(Jys  [ns]);
    double ***moment3 = convert_to_arr3(Jzs  [ns]);
    double ***moment4 = convert_to_arr3(pXXsn[ns]);
    double ***moment5 = convert_to_arr3(pXYsn[ns]);
    double ***moment6 = convert_to_arr3(pXZsn[ns]);
    double ***moment7 = convert_to_arr3(pYYsn[ns]);
    double ***moment8 = convert_to_arr3(pYZsn[ns]);
    double ***moment9 = convert_to_arr3(pZZsn[ns]);
    // add the values for the shared nodes

    //* NonBlocking Halo Exchange for Interpolation
    communicateInterp(nxn, nyn, nzn, moment0, vct, this);
    communicateInterp(nxn, nyn, nzn, moment1, vct, this);
    communicateInterp(nxn, nyn, nzn, moment2, vct, this);
    communicateInterp(nxn, nyn, nzn, moment3, vct, this);
    communicateInterp(nxn, nyn, nzn, moment4, vct, this);
    communicateInterp(nxn, nyn, nzn, moment5, vct, this);
    communicateInterp(nxn, nyn, nzn, moment6, vct, this);
    communicateInterp(nxn, nyn, nzn, moment7, vct, this);
    communicateInterp(nxn, nyn, nzn, moment8, vct, this);
    communicateInterp(nxn, nyn, nzn, moment9, vct, this);
    
    //* Calculate correct densities on the boundaries
    adjustNonPeriodicDensities(ns);

    //* Populate the ghost nodes - Nonblocking Halo Exchange
    communicateNode_P(nxn, nyn, nzn, moment0, vct, this);
    communicateNode_P(nxn, nyn, nzn, moment1, vct, this);
    communicateNode_P(nxn, nyn, nzn, moment2, vct, this);
    communicateNode_P(nxn, nyn, nzn, moment3, vct, this);
    communicateNode_P(nxn, nyn, nzn, moment4, vct, this);
    communicateNode_P(nxn, nyn, nzn, moment5, vct, this);
    communicateNode_P(nxn, nyn, nzn, moment6, vct, this);
    communicateNode_P(nxn, nyn, nzn, moment7, vct, this);
    communicateNode_P(nxn, nyn, nzn, moment8, vct, this);
    communicateNode_P(nxn, nyn, nzn, moment9, vct, this);
}

//! Communicate ghost data for ECSIM/RelSIM (computation) moments

void EMfields3D::communicateGhostP2G_ecsim(int is)
{
    const VirtualTopology3D *vct = &get_vct();
    int rank = vct->getCartesian_rank();

    //* Convert ECSIM/RelSIM moments from type array4_double to *** for communication
    // double ***moment_rhons = convert_to_arr3(rhons[is]);
    // double ***moment_Jxhs  = convert_to_arr3(Jxhs[is]);
    // double ***moment_Jyhs  = convert_to_arr3(Jyhs[is]);
    // double ***moment_Jzhs  = convert_to_arr3(Jzhs[is]);

    // interpolate adding common nodes among processors
    communicateInterp(nxn, nyn, nzn, Jxh, vct, this);
    communicateInterp(nxn, nyn, nzn, Jyh, vct, this);
    communicateInterp(nxn, nyn, nzn, Jzh, vct, this);

    //* NonBlocking Halo Exchange for Interpolation
    // communicateInterp(nxn, nyn, nzn, moment_rhons, vct, this);
    // communicateInterp(nxn, nyn, nzn, moment_Jxhs,  vct, this);
    // communicateInterp(nxn, nyn, nzn, moment_Jyhs,  vct, this);
    // communicateInterp(nxn, nyn, nzn, moment_Jzhs,  vct, this);

    communicateInterp_old(nxn, nyn, nzn, is, Jxhs,  0, 0, 0, 0, 0, 0, vct, this);
    communicateInterp_old(nxn, nyn, nzn, is, Jyhs,  0, 0, 0, 0, 0, 0, vct, this);
    communicateInterp_old(nxn, nyn, nzn, is, Jzhs,  0, 0, 0, 0, 0, 0, vct, this);
    communicateInterp_old(nxn, nyn, nzn, is, rhons, 0, 0, 0, 0, 0, 0, vct, this);

    //* Populate the ghost nodes - Nonblocking Halo Exchange
    communicateNode_P(nxn, nyn, nzn, Jxh, vct, this);
    communicateNode_P(nxn, nyn, nzn, Jyh, vct, this);
    communicateNode_P(nxn, nyn, nzn, Jzh, vct, this);

    // communicateNode_P(nxn, nyn, nzn, moment_Jxhs,  vct, this);
    // communicateNode_P(nxn, nyn, nzn, moment_Jyhs,  vct, this);
    // communicateNode_P(nxn, nyn, nzn, moment_Jzhs,  vct, this);
    // communicateNode_P(nxn, nyn, nzn, moment_rhons, vct, this);

    communicateNode_P_old(nxn, nyn, nzn, is, Jxhs,  vct, this);
    communicateNode_P_old(nxn, nyn, nzn, is, Jyhs,  vct, this);
    communicateNode_P_old(nxn, nyn, nzn, is, Jzhs,  vct, this);
    communicateNode_P_old(nxn, nyn, nzn, is, rhons, vct, this);
}

void EMfields3D::communicateGhostP2G_mass_matrix()
{
    const VirtualTopology3D * vct = &get_vct();
    int rank = vct->getCartesian_rank();

    for (int m = 0; m < NE_MASS; m++)
    {
        //! This gives wrong results
        // double ***moment_Mxx = convert_to_arr3(Mxx[m]);
        // double ***moment_Mxy = convert_to_arr3(Mxy[m]);
        // double ***moment_Mxz = convert_to_arr3(Mxz[m]);
        // double ***moment_Myx = convert_to_arr3(Myx[m]);
        // double ***moment_Myy = convert_to_arr3(Myy[m]);
        // double ***moment_Myz = convert_to_arr3(Myz[m]);
        // double ***moment_Mzx = convert_to_arr3(Mzx[m]);
        // double ***moment_Mzy = convert_to_arr3(Mzy[m]);
        // double ***moment_Mzz = convert_to_arr3(Mzz[m]);

        // communicateInterp(nxn, nyn, nzn, moment_Mxx, vct, this);
        // communicateInterp(nxn, nyn, nzn, moment_Mxy, vct, this);
        // communicateInterp(nxn, nyn, nzn, moment_Mxz, vct, this);
        // communicateInterp(nxn, nyn, nzn, moment_Myx, vct, this);
        // communicateInterp(nxn, nyn, nzn, moment_Myy, vct, this);
        // communicateInterp(nxn, nyn, nzn, moment_Myz, vct, this);
        // communicateInterp(nxn, nyn, nzn, moment_Mzx, vct, this);
        // communicateInterp(nxn, nyn, nzn, moment_Mzy, vct, this);
        // communicateInterp(nxn, nyn, nzn, moment_Mzz, vct, this);

        // communicateNode_P(nxn, nyn, nzn, moment_Mxx, vct, this);
        // communicateNode_P(nxn, nyn, nzn, moment_Mxy, vct, this);
        // communicateNode_P(nxn, nyn, nzn, moment_Mxz, vct, this);
        // communicateNode_P(nxn, nyn, nzn, moment_Myx, vct, this);
        // communicateNode_P(nxn, nyn, nzn, moment_Myy, vct, this);
        // communicateNode_P(nxn, nyn, nzn, moment_Myz, vct, this);
        // communicateNode_P(nxn, nyn, nzn, moment_Mzx, vct, this);
        // communicateNode_P(nxn, nyn, nzn, moment_Mzy, vct, this);
        // communicateNode_P(nxn, nyn, nzn, moment_Mzz, vct, this);

        communicateInterp_old(nxn, nyn, nzn, m, Mxx, 0, 0, 0, 0, 0, 0, vct, this);
        communicateInterp_old(nxn, nyn, nzn, m, Mxy, 0, 0, 0, 0, 0, 0, vct, this);
        communicateInterp_old(nxn, nyn, nzn, m, Mxz, 0, 0, 0, 0, 0, 0, vct, this);
        communicateInterp_old(nxn, nyn, nzn, m, Myx, 0, 0, 0, 0, 0, 0, vct, this);
        communicateInterp_old(nxn, nyn, nzn, m, Myy, 0, 0, 0, 0, 0, 0, vct, this);
        communicateInterp_old(nxn, nyn, nzn, m, Myz, 0, 0, 0, 0, 0, 0, vct, this);
        communicateInterp_old(nxn, nyn, nzn, m, Mzx, 0, 0, 0, 0, 0, 0, vct, this);
        communicateInterp_old(nxn, nyn, nzn, m, Mzy, 0, 0, 0, 0, 0, 0, vct, this);
        communicateInterp_old(nxn, nyn, nzn, m, Mzz, 0, 0, 0, 0, 0, 0, vct, this);

        communicateNode_P_old(nxn, nyn, nzn, m, Mxx, vct, this);
        communicateNode_P_old(nxn, nyn, nzn, m, Mxy, vct, this);
        communicateNode_P_old(nxn, nyn, nzn, m, Mxz, vct, this);
        communicateNode_P_old(nxn, nyn, nzn, m, Myx, vct, this);
        communicateNode_P_old(nxn, nyn, nzn, m, Myy, vct, this);
        communicateNode_P_old(nxn, nyn, nzn, m, Myz, vct, this);
        communicateNode_P_old(nxn, nyn, nzn, m, Mzx, vct, this);
        communicateNode_P_old(nxn, nyn, nzn, m, Mzy, vct, this);
        communicateNode_P_old(nxn, nyn, nzn, m, Mzz, vct, this);
    }
}

//! Communicate ghost data for ECSIM/RelSIM (output only) moments
void EMfields3D::communicateGhostP2G_supplementary_moments(int is) 
{
    const VirtualTopology3D *vct = &get_vct();
    int rank = vct->getCartesian_rank();

    communicateInterp_old(nxn, nyn, nzn, is, rhons, 0, 0, 0, 0, 0, 0, vct, this);
    
    communicateInterp_old(nxn, nyn, nzn, is, Jxs,  0, 0, 0, 0, 0, 0, vct, this);
    communicateInterp_old(nxn, nyn, nzn, is, Jys,  0, 0, 0, 0, 0, 0, vct, this);
    communicateInterp_old(nxn, nyn, nzn, is, Jzs,  0, 0, 0, 0, 0, 0, vct, this);

    communicateInterp_old(nxn, nyn, nzn, is, E_flux_xs,  0, 0, 0, 0, 0, 0, vct, this);
    communicateInterp_old(nxn, nyn, nzn, is, E_flux_ys,  0, 0, 0, 0, 0, 0, vct, this);
    communicateInterp_old(nxn, nyn, nzn, is, E_flux_zs,  0, 0, 0, 0, 0, 0, vct, this);

    if (SaveHeatFluxTensor) 
    {
        communicateInterp_old(nxn, nyn, nzn, is, Qxxxs, 0, 0, 0, 0, 0, 0, vct, this);
        communicateInterp_old(nxn, nyn, nzn, is, Qxxys, 0, 0, 0, 0, 0, 0, vct, this);
        communicateInterp_old(nxn, nyn, nzn, is, Qxyys, 0, 0, 0, 0, 0, 0, vct, this);
        communicateInterp_old(nxn, nyn, nzn, is, Qxzzs, 0, 0, 0, 0, 0, 0, vct, this);
        communicateInterp_old(nxn, nyn, nzn, is, Qyyys, 0, 0, 0, 0, 0, 0, vct, this);
        communicateInterp_old(nxn, nyn, nzn, is, Qyzzs, 0, 0, 0, 0, 0, 0, vct, this);
        communicateInterp_old(nxn, nyn, nzn, is, Qzzzs, 0, 0, 0, 0, 0, 0, vct, this);
        communicateInterp_old(nxn, nyn, nzn, is, Qxyzs, 0, 0, 0, 0, 0, 0, vct, this);
        communicateInterp_old(nxn, nyn, nzn, is, Qxxzs, 0, 0, 0, 0, 0, 0, vct, this);
        communicateInterp_old(nxn, nyn, nzn, is, Qyyzs, 0, 0, 0, 0, 0, 0, vct, this);
    }

    communicateInterp_old(nxn, nyn, nzn, is, pXXsn, 0, 0, 0, 0, 0, 0, vct, this);
    communicateInterp_old(nxn, nyn, nzn, is, pXYsn, 0, 0, 0, 0, 0, 0, vct, this);
    communicateInterp_old(nxn, nyn, nzn, is, pXZsn, 0, 0, 0, 0, 0, 0, vct, this);
    communicateInterp_old(nxn, nyn, nzn, is, pYYsn, 0, 0, 0, 0, 0, 0, vct, this);
    communicateInterp_old(nxn, nyn, nzn, is, pYZsn, 0, 0, 0, 0, 0, 0, vct, this);
    communicateInterp_old(nxn, nyn, nzn, is, pZZsn, 0, 0, 0, 0, 0, 0, vct, this);

    communicateNode_P_old(nxn, nyn, nzn, is, rhons, vct, this);

    communicateNode_P_old(nxn, nyn, nzn, is, Jxs, vct, this);
    communicateNode_P_old(nxn, nyn, nzn, is, Jys, vct, this);
    communicateNode_P_old(nxn, nyn, nzn, is, Jzs, vct, this);

    communicateNode_P_old(nxn, nyn, nzn, is, E_flux_xs, vct, this);
    communicateNode_P_old(nxn, nyn, nzn, is, E_flux_ys, vct, this);
    communicateNode_P_old(nxn, nyn, nzn, is, E_flux_zs, vct, this);

    if (SaveHeatFluxTensor)
    {
        communicateNode_P_old(nxn, nyn, nzn, is, Qxxxs, vct, this);
        communicateNode_P_old(nxn, nyn, nzn, is, Qxxys, vct, this);
        communicateNode_P_old(nxn, nyn, nzn, is, Qxyys, vct, this);
        communicateNode_P_old(nxn, nyn, nzn, is, Qxzzs, vct, this);
        communicateNode_P_old(nxn, nyn, nzn, is, Qyyys, vct, this);
        communicateNode_P_old(nxn, nyn, nzn, is, Qyzzs, vct, this);
        communicateNode_P_old(nxn, nyn, nzn, is, Qzzzs, vct, this);
        communicateNode_P_old(nxn, nyn, nzn, is, Qxyzs, vct, this);
        communicateNode_P_old(nxn, nyn, nzn, is, Qxxzs, vct, this);
        communicateNode_P_old(nxn, nyn, nzn, is, Qyyzs, vct, this);
    }

    communicateNode_P_old(nxn, nyn, nzn, is, pXXsn, vct, this);
    communicateNode_P_old(nxn, nyn, nzn, is, pXYsn, vct, this);
    communicateNode_P_old(nxn, nyn, nzn, is, pXZsn, vct, this);
    communicateNode_P_old(nxn, nyn, nzn, is, pYYsn, vct, this);
    communicateNode_P_old(nxn, nyn, nzn, is, pYZsn, vct, this);
    communicateNode_P_old(nxn, nyn, nzn, is, pZZsn, vct, this);
}

//! ===================================== Compute Fields ===================================== !//

//? Convert a 3D field to a 1D array (not considering guard cells)
void solver2phys(arr3_double vectPhys, double *vectSolver, int nx, int ny, int nz) 
{
    for (int i = 1; i < nx - 1; i++)
        for (int j = 1; j < ny - 1; j++)
            for (int k = 1; k < nz - 1; k++)
                vectPhys[i][j][k] = *vectSolver++;
}

//? Convert three 3D fields to a 1D array (not considering guard cells)
void solver2phys(arr3_double vectPhys1, arr3_double vectPhys2, arr3_double vectPhys3, double *vectSolver, int nx, int ny, int nz) 
{
    for (int i = 1; i < nx - 1; i++)
        for (int j = 1; j < ny - 1; j++)
            for (int k = 1; k < nz - 1; k++) 
            {
                vectPhys1[i][j][k] = *vectSolver++;
                vectPhys2[i][j][k] = *vectSolver++;
                vectPhys3[i][j][k] = *vectSolver++;
            }
}

//? Convert a 1D vector to 3D field (not considering guard cells)
void phys2solver(double *vectSolver, const arr3_double vectPhys, int nx, int ny, int nz) 
{
    for (int i = 1; i < nx - 1; i++)
        for (int j = 1; j < ny - 1; j++)
            for (int k = 1; k < nz - 1; k++)
                *vectSolver++ = vectPhys.get(i,j,k);
}

//? Convert a 1D vector to three 3D fields (not considering guard cells)
void phys2solver(double *vectSolver, const arr3_double vectPhys1, const arr3_double vectPhys2, const arr3_double vectPhys3, int nx, int ny, int nz) 
{
    for (int i = 1; i < nx - 1; i++)
        for (int j = 1; j < ny - 1; j++)
            for (int k = 1; k < nz - 1; k++) 
            {
                *vectSolver++ = vectPhys1.get(i,j,k);
                *vectSolver++ = vectPhys2.get(i,j,k);
                *vectSolver++ = vectPhys3.get(i,j,k);
            }
}

//? Calculate electric field using GMRes
void EMfields3D::calculateE()
{
    #ifdef __PROFILE_FIELDS__
    LeXInt::timer time_ms, time_gmres, time_com, time_total;

    time_total.start();
    #endif

    const Collective *col = &get_col();
    const VirtualTopology3D * vct = &get_vct();
    const Grid *grid = &get_grid();

    if (vct->getCartesian_rank() == 0)
        cout << "*** Electric field computation ***" << endl;

    //? X,Y,Z components for E
    double *xkrylov = new double[3 * (nxn - 2) * (nyn - 2) * (nzn - 2)];
    double *bkrylov = new double[3 * (nxn - 2) * (nyn - 2) * (nzn - 2)];

    //? Initialise all params with zeros 
    eqValue(0.0, xkrylov, 3 * (nxn - 2) * (nyn - 2) * (nzn - 2));
    eqValue(0.0, bkrylov, 3 * (nxn - 2) * (nyn - 2) * (nzn - 2));

    #ifdef __PROFILE_FIELDS__
    time_ms.start();
    #endif

    //* Prepare the source 
    MaxwellSource(bkrylov);

    #ifdef __PROFILE_FIELDS__
    time_ms.stop();
    #endif

    //* Move to Krylov space from physical space
    phys2solver(xkrylov, Ex, Ey, Ez, nxn, nyn, nzn);

    #ifdef __PROFILE_FIELDS__
    time_gmres.start();
    #endif
    
    //? Solve using GMRes
    GMRES(&Field::MaxwellImage, xkrylov, 3 * (nxn - 2) * (nyn - 2) * (nzn - 2), bkrylov, 20, 50, GMREStol, this);

    #ifdef __PROFILE_FIELDS__
    time_gmres.stop();
    #endif

    //* Move from Krylov space to physical space
    solver2phys(Exth, Eyth, Ezth, xkrylov, nxn, nyn, nzn);

    #ifdef __PROFILE_FIELDS__
    time_com.start();
    #endif

    //? Communicate E theta so the interpolation can have good values
    communicateNodeBC(nxn, nyn, nzn, Exth, col->bcEx[0], col->bcEx[1], col->bcEx[2], col->bcEx[3], col->bcEx[4], col->bcEx[5], vct, this);
    communicateNodeBC(nxn, nyn, nzn, Eyth, col->bcEy[0], col->bcEy[1], col->bcEy[2], col->bcEy[3], col->bcEy[4], col->bcEy[5], vct, this);
    communicateNodeBC(nxn, nyn, nzn, Ezth, col->bcEz[0], col->bcEz[1], col->bcEz[2], col->bcEz[3], col->bcEz[4], col->bcEz[5], vct, this);

    #ifdef __PROFILE_FIELDS__
    time_com.stop();
    #endif

    //* E(x,y,z) = -(1.0 - th)/th * E(x,y,z) + 1.0/th * Eth(x,y,z): scale the electric field values
    addscale(1.0/th, -(1.0 - th)/th, Ex, Exth, nxn, nyn, nzn);
    addscale(1.0/th, -(1.0 - th)/th, Ey, Eyth, nxn, nyn, nzn);
    addscale(1.0/th, -(1.0 - th)/th, Ez, Ezth, nxn, nyn, nzn);

    #ifdef __PROFILE_FIELDS__
    time_com.start();
    #endif

    //? Communicate E
    communicateNodeBC(nxn, nyn, nzn, Ex, col->bcEx[0],col->bcEx[1],col->bcEx[2],col->bcEx[3],col->bcEx[4],col->bcEx[5], vct, this);
    communicateNodeBC(nxn, nyn, nzn, Ey, col->bcEy[0],col->bcEy[1],col->bcEy[2],col->bcEy[3],col->bcEy[4],col->bcEy[5], vct, this);
    communicateNodeBC(nxn, nyn, nzn, Ez, col->bcEz[0],col->bcEz[1],col->bcEz[2],col->bcEz[3],col->bcEz[4],col->bcEz[5], vct, this);

    #ifdef __PROFILE_FIELDS__
    time_com.stop();
    #endif

    //? OpenBC Inflow: this needs to be integrate to Halo Exchange BC
    //TODO: Is this implemented? Ask Andong/Stefano
    // OpenBoundaryInflowE(Exth, Eyth, Ezth, nxn, nyn, nzn);
    // OpenBoundaryInflowB(Ex, Ey, Ez, nxn, nyn, nzn);

    //* Deallocate temporary arrays
    delete[]xkrylov;
    delete[]bkrylov;

    #ifdef __PROFILE_FIELDS__
    time_total.stop();

    if(MPIdata::get_rank() == 0)
    {
        cout << endl << "   FIELD SOLVER (calculateE())" << endl; 
        cout << "       Maxwell Source              : " << time_ms.total()    << " s, fraction of time taken in calculateE(): " << time_ms.total()/time_total.total() << endl;
        cout << "       GMRes                       : " << time_gmres.total() << " s, fraction of time taken in calculateE(): " << time_gmres.total()/time_total.total() << endl;
        cout << "       Communicate                 : " << time_com.total()   << " s, fraction of time taken in calculateE(): " << time_com.total()/time_total.total() << endl;
        cout << "       calculateE()                : " << time_total.total() << " s" << endl << endl;
    }
    #endif  
}

//? LHS of the Maxwell solver
void EMfields3D::MaxwellSource(double *bkrylov)
{
    const Collective *col = &get_col();
    const VirtualTopology3D * vct = &get_vct();
    const Grid *grid = &get_grid();

    //* Valid for second-order formulation
    communicateCenterBC(nxc, nyc, nzc, Bxc, col->bcBx[0],col->bcBx[1],col->bcBx[2],col->bcBx[3],col->bcBx[4],col->bcBx[5], vct, this);
    communicateCenterBC(nxc, nyc, nzc, Byc, col->bcBy[0],col->bcBy[1],col->bcBy[2],col->bcBy[3],col->bcBy[4],col->bcBy[5], vct, this);
    communicateCenterBC(nxc, nyc, nzc, Bzc, col->bcBz[0],col->bcBz[1],col->bcBz[2],col->bcBz[3],col->bcBz[4],col->bcBz[5], vct, this);

    //* Compute curl of magnetic field (defined at cell centres) on nodes
    grid->curlC2N(temp2X, temp2Y, temp2Z, Bxc, Byc, Bzc);

    //? External magnetic field
    // if (col->getAddExternalCurlB()) 
    // {
    //     //* Dipole SOURCE version using J_ext
    //     if (vct->getCartesian_rank() == 0)
    //         cout << "*** Add contribution to the Curl of B_ext to Maxwell Source ***" << endl;
    //
    //     communicateCenterBC(nxc, nyc, nzc, Bxc_ext, col->bcBx[0],col->bcBx[1],col->bcBx[2],col->bcBx[3],col->bcBx[4],col->bcBx[5], vct, this);
    //     communicateCenterBC(nxc, nyc, nzc, Byc_ext, col->bcBy[0],col->bcBy[1],col->bcBy[2],col->bcBy[3],col->bcBy[4],col->bcBy[5], vct, this);
    //     communicateCenterBC(nxc, nyc, nzc, Bzc_ext, col->bcBz[0],col->bcBz[1],col->bcBz[2],col->bcBz[3],col->bcBz[4],col->bcBz[5], vct, this);
    //    
    //     grid->curlC2N(Jx_ext, Jy_ext, Jz_ext, Bxc_ext, Byc_ext, Bzc_ext);
    //    
    //     addscale(1.0, temp2X, Jx_ext, nxn, nyn, nzn);
    //     addscale(1.0, temp2Y, Jy_ext, nxn, nyn, nzn);
    //     addscale(1.0, temp2Z, Jz_ext, nxn, nyn, nzn);
    // }

    //? --------------------------------------------------------- ?//

    //* Energy-conserving smoothing (BC nodes are taken care of in the smoothing process)
    energy_conserve_smooth(Jxh, Jyh, Jzh, nxn, nyn, nzn);

    for (int i = 0; i < nxn; i++)
        for (int j = 0; j < nyn; j++) 
            for (int k = 0; k < nzn; k++)
            {
                //? If zeroCurrent is implemented, the follwoing 3 lines need to be changed to J_tot = Jh.(i,j,k) + zeroCurrent*J_ext.get(i, j, k)
                double Jx_tot = Jxh.get(i, j, k);
                double Jy_tot = Jyh.get(i, j, k);
                double Jz_tot = Jzh.get(i, j, k);
                
                temp3X.fetch(i, j, k) = Jxh.get(i, j, k)*invVOL;
                temp3Y.fetch(i, j, k) = Jyh.get(i, j, k)*invVOL;
                temp3Z.fetch(i, j, k) = Jzh.get(i, j, k)*invVOL;

                tempX.fetch(i, j, k) = th*dt*(c*temp2X.get(i, j, k) - FourPI*Jx_tot*invVOL);
                tempY.fetch(i, j, k) = th*dt*(c*temp2Y.get(i, j, k) - FourPI*Jy_tot*invVOL);
                tempZ.fetch(i, j, k) = th*dt*(c*temp2Z.get(i, j, k) - FourPI*Jz_tot*invVOL);
            }

    //* temp(x,y,z) = temp(x,y,z) + 1.0*E(x,y,z)
    addscale(1.0, tempX, Ex, nxn, nyn, nzn);
    addscale(1.0, tempY, Ey, nxn, nyn, nzn);
    addscale(1.0, tempZ, Ez, nxn, nyn, nzn);

    // //? Add contribution of external electric field to the change in B
    // if (col->getAddExternalCurlE()) 
    // {
    //     grid->lapN2N(temp2X, Ex_ext, this);
    //     grid->lapN2N(temp2Y, Ey_ext, this);
    //     grid->lapN2N(temp2Z, Ez_ext, this);
    //
    //     addscale(c*th*dt*c*th*dt, tempX, temp2X, nxn, nyn, nzn);
    //     addscale(c*th*dt*c*th*dt, tempY, temp2Y, nxn, nyn, nzn);
    //     addscale(c*th*dt*c*th*dt, tempZ, temp2Z, nxn, nyn, nzn);
    // }

    //* Physical space --> Krylov space
    phys2solver(bkrylov, tempX, tempY, tempZ, nxn, nyn, nzn);
}

//? RHS of the Maxwell solver
//
// In the field solver, there is one layer of ghost cells. The nodes on the ghost cells define two outer layers of nodes: the
// outermost nodes are clearly in the interior of the neighboring subdomain and can naturally be referred to as "ghost nodes",
// but the second-outermost layer is on the boundary between subdomains and thus does not clearly belong to any one process.
// Refer to these shared nodes as "boundary nodes".
//
// To compute the laplacian, we first compute the gradient at the center of each cell by differencing the values at
// the corners of the cell. We then compute the Laplacian (i.e. the divergence of the gradient) at each node by
// differencing the cell-center values in the cells sharing the node. 
//
// The laplacian is required to be defined on all boundary and interior nodes.  
// 
// In the krylov solver, we make no attempt to use or to update the (outer) ghost nodes, and we assume (presumably
// correctly) that the boundary nodes are updated identically by all processes that share them. sTherefore, we must
// communicate gradient values in the ghost cells. The subsequent computation of the divergence requires that
// this boundary communication first complete.
//
// An alternative way would be to communicate outer ghost node values after each update of Eth. In this case, there would
// be no need for the 10=3*3+1 boundary communications in the body of MaxwellImage() entailed in the calls to lapN2N plus the
// call needed prior to the call to gradC2N. Of course, we would then need to communicate the 3 components of the
// electric field for the outer ghost nodes prior to each call to MaxwellImage().  This second alternative would thus reduce
// communication by over a factor of 3. Essentially, we would replace the cost of communicating cell-centered differences
// for ghost cell values with the cost of directly computing them.
//
// Also, while this second method does not increase the potential to avoid exposing latency, it can make it easier to do so.
//
// Another change that I would propose: define:
//   array4_double physical_vector(3,nxn,nyn,nzn);
//   arr3_double vectX = physical_vector[0];
//   arr3_double vectY = physical_vector[1];
//   arr3_double vectZ = physical_vector[2];
//   vector = &physical_vector[0][0][0][0];
//
// It is currently the case that boundary nodes are duplicated in "vector" and therefore receive a weight
// that is twice, four times, or eight times as much as other nodes in the Krylov inner product. The definitions
// above would imply that ghost nodes also appear in the inner product. To avoid this issue, we could simply zero
// ghost nodes before returning to the Krylov solver. With the definitions above, phys2solver() would simply zero
// the ghost nodes and solver2phys() would populate them via communication. Note that it would also be possible, if
// desired, to give duplicated nodes equal weight by rescaling their values in these two methods.
void EMfields3D::MaxwellImage(double *im, double* vector)
{
    //* double *im      : Output
    //* double* vector  : Input

    const Collective *col = &get_col();
    const VirtualTopology3D *vct = &get_vct();
    const Grid *grid = &get_grid();

    eqValue(0.0, im, 3 * (nxn - 2) * (nyn - 2) * (nzn - 2));

    //? Move from Krylov space to physical space
    solver2phys(tempX, tempY, tempZ, vector, nxn, nyn, nzn);

    communicateNodeBC_old(nxn, nyn, nzn, tempX, col->bcEx[0],col->bcEx[1],col->bcEx[2],col->bcEx[3],col->bcEx[4],col->bcEx[5], vct, this);
	communicateNodeBC_old(nxn, nyn, nzn, tempY, col->bcEy[0],col->bcEy[1],col->bcEy[2],col->bcEy[3],col->bcEy[4],col->bcEy[5], vct, this);
	communicateNodeBC_old(nxn, nyn, nzn, tempZ, col->bcEz[0],col->bcEz[1],col->bcEz[2],col->bcEz[3],col->bcEz[4],col->bcEz[5], vct, this);

    //? curl(curl(E)) is computed using finite differences
    grid->curlN2C(tempXC, tempYC, tempZC, tempX, tempY, tempZ);
    
    communicateCenterBC(nxc, nyc, nzc, tempXC, 1, 1, 1, 1, 1, 1, vct, this);
    communicateCenterBC(nxc, nyc, nzc, tempYC, 1, 1, 1, 1, 1, 1, vct, this);
    communicateCenterBC(nxc, nyc, nzc, tempZC, 1, 1, 1, 1, 1, 1, vct, this);

    grid->curlC2N(imageX, imageY, imageZ, tempXC, tempYC, tempZC);

    //* Multiply by factor
    double factor = c*th*dt*c*th*dt;
    for (int i=1;i<nxn-1;i++)
        for (int j=1;j<nyn-1;j++)
            for (int k=1;k<nzn-1;k++) 
            {
                imageX[i][j][k] = tempX[i][j][k] + factor * imageX[i][j][k];
                imageY[i][j][k] = tempY[i][j][k] + factor * imageY[i][j][k];
                imageZ[i][j][k] = tempZ[i][j][k] + factor * imageZ[i][j][k];
            }

    //* Energy-conserving smoothing (BC nodes are taken care of in the smoothing process)
    energy_conserve_smooth(tempX, tempY, tempZ, nxn, nyn, nzn);

    #pragma omp parallel for collapse(3) schedule(static)
    for (int i=1; i<nxn-1; i++) 
        for (int j=1; j<nyn-1; j++) 
            for (int k=1; k<nzn-1; k++) 
            {
                double MEx, MEy, MEz;
                
                mass_matrix_times_vector(&MEx, &MEy, &MEz, tempX, tempY, tempZ, i, j, k);
                
                temp2X[i][j][k] = dt*th*FourPI*MEx;
                temp2Y[i][j][k] = dt*th*FourPI*MEy;
                temp2Z[i][j][k] = dt*th*FourPI*MEz;
            }

    //* Energy-conserving smoothing (BC nodes are taken care of in the smoothing process)
    energy_conserve_smooth(temp2X, temp2Y, temp2Z, nxn, nyn, nzn);

    for (int i=1; i<nxn-1; i++)
        for (int j=1; j<nyn-1; j++)
            for (int k=1; k<nzn-1; k++) 
            {
                temp2X[i][j][k] *= invVOL;
                temp2Y[i][j][k] *= invVOL;
                temp2Z[i][j][k] *= invVOL;
            }
    
    for (int i=1;i<nxn-1;i++)
        for (int j=1;j<nyn-1;j++)
            for (int k=1;k<nzn-1;k++) 
            {
                imageX[i][j][k] = temp2X[i][j][k] + imageX[i][j][k];
                imageY[i][j][k] = temp2Y[i][j][k] + imageY[i][j][k];
                imageZ[i][j][k] = temp2Z[i][j][k] + imageZ[i][j][k];
            }

    //? Move from physical space to Krylov space
    phys2solver(im, imageX, imageY, imageZ, nxn, nyn, nzn);
}

//? Update the values of magnetic field at the nodes at time n+1
void EMfields3D::C2NB() 
{
    const Collective *col = &get_col();
    const VirtualTopology3D *vct = &get_vct();
    const Grid *grid = &get_grid();

    grid->interpC2N(Bxn, Bxc);
    grid->interpC2N(Byn, Byc);
    grid->interpC2N(Bzn, Bzc);
    
    communicateNodeBC(nxn, nyn, nzn, Bxn, col->bcBx[0],col->bcBx[1],col->bcBx[2],col->bcBx[3],col->bcBx[4],col->bcBx[5], vct, this);
    communicateNodeBC(nxn, nyn, nzn, Byn, col->bcBy[0],col->bcBy[1],col->bcBy[2],col->bcBy[3],col->bcBy[4],col->bcBy[5], vct, this);
    communicateNodeBC(nxn, nyn, nzn, Bzn, col->bcBz[0],col->bcBz[1],col->bcBz[2],col->bcBz[3],col->bcBz[4],col->bcBz[5], vct, this);
}

//? Populate the field data used to push particles
void EMfields3D::set_fieldForPcls()
{
    #pragma omp parallel for collapse(3)
    for(int i = 0; i < nxn; i++)
        for(int j = 0; j < nyn; j++)
            for(int k = 0; k < nzn; k++)
            {
                fieldForPcls[i][j][k][0] = (pfloat) Bxn[i][j][k];
                fieldForPcls[i][j][k][1] = (pfloat) Byn[i][j][k];
                fieldForPcls[i][j][k][2] = (pfloat) Bzn[i][j][k];

                //TODO: When external fields are implemented, B_ext has to be implemented
                // fieldForPcls[i][j][k][0] = (pfloat) (Bxn[i][j][k] + Bx_ext[i][j][k]);
                // fieldForPcls[i][j][k][1] = (pfloat) (Byn[i][j][k] + By_ext[i][j][k]);
                // fieldForPcls[i][j][k][2] = (pfloat) (Bzn[i][j][k] + Bz_ext[i][j][k]);
                
                fieldForPcls[i][j][k][0+DFIELD_3or4] = (pfloat) Exth[i][j][k];
                fieldForPcls[i][j][k][1+DFIELD_3or4] = (pfloat) Eyth[i][j][k];
                fieldForPcls[i][j][k][2+DFIELD_3or4] = (pfloat) Ezth[i][j][k];
            }
}

//? Calculate magnetic field (defined on nodes) using precomputed E(n + theta), the magnetic field is evaluated from Faraday's law 
void EMfields3D::calculateB()
{
    const Collective *col = &get_col();
    const VirtualTopology3D *vct = &get_vct();
    const Grid *grid = &get_grid();

    if (vct->getCartesian_rank() == 0)
        cout << "*** Magnetic field computation ***" << endl;

    //? Compute curl of E_theta
    grid->curlN2C(tempXC, tempYC, tempZC, Exth, Eyth, Ezth);

    //? Compute curl of E_ext
    // if (col->getAddExternalCurlE()) 
    //     grid->curlN2C(tempXC2, tempYC2, tempZC2, Ex_ext, Ey_ext, Ez_ext);

    //* Energy-conserving smoothing (BC nodes are taken care of in the smoothing process)
    energy_conserve_smooth(Exth, Eyth, Ezth, nxn, nyn, nzn);

    //? Update magnetic field: Second order formulation
    addscale(-c * dt, 1, Bxc, tempXC, nxc, nyc, nzc);
    addscale(-c * dt, 1, Byc, tempYC, nxc, nyc, nzc);
    addscale(-c * dt, 1, Bzc, tempZC, nxc, nyc, nzc);
    
    // if (col->getAddExternalCurlE()) 
    // {
    //     addscale(-c * dt, 1, Bxc, tempXC2, nxc, nyc, nzc);
    //     addscale(-c * dt, 1, Byc, tempYC2, nxc, nyc, nzc);
    //     addscale(-c * dt, 1, Bzc, tempZC2, nxc, nyc, nzc);
    // }

    //? Communicate ghost cells -- centres for magnetic field
    communicateCenterBC(nxc, nyc, nzc, Bxc, col->bcBx[0],col->bcBx[1],col->bcBx[2],col->bcBx[3],col->bcBx[4],col->bcBx[5], vct, this);
    communicateCenterBC(nxc, nyc, nzc, Byc, col->bcBy[0],col->bcBy[1],col->bcBy[2],col->bcBy[3],col->bcBy[4],col->bcBy[5], vct, this);
    communicateCenterBC(nxc, nyc, nzc, Bzc, col->bcBz[0],col->bcBz[1],col->bcBz[2],col->bcBz[3],col->bcBz[4],col->bcBz[5], vct, this);

    //? Boundary conditions for magnetic field
    // fixBC_B(vct, col);
}


//! ===================================== Smoothing ===================================== !//

void EMfields3D::energy_conserve_smooth_direction(double*** data, int nx, int ny, int nz, int dir)
{
    const Collective *col = &get_col();
    const VirtualTopology3D *vct = &get_vct();

    int bc[6];
    if (dir == 0)      for (int i=0; i<6; i++) bc[i] = col->bcEx[i];    //* BC along X
    else if (dir == 1) for (int i=0; i<6; i++) bc[i] = col->bcEy[i];    //* BC along Y
    else if (dir == 2) for (int i=0; i<6; i++) bc[i] = col->bcEz[i];    //* BC along Z

    //? Initialise temporary arrays with zeros
    double ***temp = newArr3(double, nx, ny, nz);

    //! Using new communication routines results in energy growth
    communicateNodeBC_old(nx, ny, nz, data, bc[0], bc[1], bc[2], bc[3], bc[4], bc[5], vct, this);

    int k = 0;

    for (int icount = 0; icount < num_smoothings; icount++)
    {
        for (int i = 1; i < nx - 1; i++)
            for (int j = 1; j < ny - 1; j++)
                for (int k = 1; k < nz - 1; k++)
                    temp[i][j][k] = 0.015625 * (8.0*data[i][j][k]
                                              + 4.0 * (data[i-1][j][k] + data[i+1][j][k] + data[i][j-1][k] + data[i][j+1][k] + data[i][j][k-1] + data[i][j][k+1])     //* Faces
                                              + 2.0 * (data[i-1][j-1][k] + data[i+1][j-1][k] + data[i-1][j+1][k] + data[i+1][j+1][k]                                  //* Edges
                                                    +  data[i-1][j][k-1] + data[i+1][j][k-1] + data[i][j-1][k-1] + data[i][j+1][k-1]                                  //* Edges
                                                    +  data[i-1][j][k+1] + data[i+1][j][k+1] + data[i][j-1][k+1] + data[i][j+1][k+1])                                 //* Edges
                                              + 1.0 * (data[i-1][j-1][k-1] + data[i+1][j-1][k-1] + data[i-1][j+1][k-1] + data[i+1][j+1][k-1]                          //* Corners
                                                    +  data[i-1][j-1][k+1] + data[i+1][j-1][k+1] + data[i-1][j+1][k+1] + data[i+1][j+1][k+1]));                       //* Corners

        for (int i = 1; i < nx - 1; i++)
            for (int j = 1; j < ny - 1; j++)
                for (int k = 1; k < nz - 1; k++)
                    data[i][j][k] = temp[i][j][k];

        //! Using new communication routines results in energy growth
        communicateNodeBC_old(nx, ny, nz, data, bc[0], bc[1], bc[2], bc[3], bc[4], bc[5], vct, this);
    }

    delArr3(temp, nxn, nyn);
}

void EMfields3D::energy_conserve_smooth(arr3_double data_X, arr3_double data_Y, arr3_double data_Z, int nx, int ny, int nz)
{
    const Collective *col = &get_col();

    if (col->getSmoothCycle() > 0 && (col->getCurrentCycle() % col->getSmoothCycle() == 0) && Smooth == true)
    {
        //* Directional: First, smooth along X, then Y, and finally along Z (2 neighbours)
        energy_conserve_smooth_direction(data_X, nx, ny, nz, 0);
        energy_conserve_smooth_direction(data_Y, nx, ny, nz, 1);
        energy_conserve_smooth_direction(data_Z, nx, ny, nz, 2);
    }
}


//! ===================================== Initial Field Distributions ===================================== !//

//? B boundary for GEM (cell centres) - this assumes non-periodic boundaries along Y ?//
void EMfields3D::fixBcGEM()
{
    const VirtualTopology3D *vct = &get_vct();
    const Grid *grid = &get_grid();

    if (vct->getYright_neighbor() == MPI_PROC_NULL) 
    {
        for (int i = 0; i < nxc; i++)
            for (int k = 0; k < nzc; k++) 
            {
                Bxc[i][nyc - 1][k] = B0x * tanh((grid->getYC(i, nyc - 1, k) - Ly / 2) / delta);
                Bxc[i][nyc - 2][k] = Bxc[i][nyc - 1][k];
                Bxc[i][nyc - 3][k] = Bxc[i][nyc - 1][k];
                Byc[i][nyc - 1][k] = B0y;
                Bzc[i][nyc - 1][k] = B0z;
                Bzc[i][nyc - 2][k] = B0z;
                Bzc[i][nyc - 3][k] = B0z;
            }
    }
    
    if (vct->getYleft_neighbor() == MPI_PROC_NULL) 
    {
        for (int i = 0; i < nxc; i++)
            for (int k = 0; k < nzc; k++) 
            {
                Bxc[i][0][k] = B0x * tanh((grid->getYC(i, 0, k) - Ly / 2) / delta);
                Bxc[i][1][k] = Bxc[i][0][k];
                Bxc[i][2][k] = Bxc[i][0][k];
                Byc[i][0][k] = B0y;
                Bzc[i][0][k] = B0z;
                Bzc[i][1][k] = B0z;
                Bzc[i][2][k] = B0z;
            }
    }
}

//? B boundary for GEM (nodes) - this assumes non-periodic boundaries along Y ?//
void EMfields3D::fixBnGEM()
{
    const VirtualTopology3D *vct = &get_vct();
    const Grid *grid = &get_grid();

    if (vct->getYright_neighbor() == MPI_PROC_NULL) 
    {
        for (int i = 0; i < nxn; i++)
            for (int k = 0; k < nzn; k++) 
            {
                Bxn[i][nyn - 1][k] = B0x * tanh((grid->getYC(i, nyc - 1, k) - Ly / 2) / delta);
                Bxn[i][nyn - 2][k] = Bxn[i][nyn - 1][k];
                Bxn[i][nyn - 3][k] = Bxn[i][nyn - 1][k];
                Byn[i][nyn - 1][k] = B0y;
                Bzn[i][nyn - 1][k] = B0z;
                Bzn[i][nyn - 2][k] = B0z;
                Bzn[i][nyn - 3][k] = B0z;
            }
    }

    if (vct->getYleft_neighbor() == MPI_PROC_NULL) 
    {
        for (int i = 0; i < nxn; i++)
            for (int k = 0; k < nzn; k++) 
            {
                Bxn[i][0][k] = B0x * tanh((grid->getYC(i, 0, k) - Ly / 2) / delta);
                Bxn[i][1][k] = Bxn[i][0][k];
                Bxn[i][2][k] = Bxn[i][0][k];
                Byn[i][0][k] = B0y;
                Bzn[i][0][k] = B0z;
                Bzn[i][1][k] = B0z;
                Bzn[i][2][k] = B0z;
            }
    }
}

//? B boundary for forcefree ?//
void EMfields3D::fixBforcefree()
{
    const VirtualTopology3D *vct = &get_vct();
    const Grid *grid = &get_grid();

    if (vct->getYright_neighbor() == MPI_PROC_NULL) 
    {
        for (int i = 0; i < nxc; i++)
            for (int k = 0; k < nzc; k++) 
            {
                Bxc[i][nyc - 1][k] = B0x * tanh((grid->getYC(i, nyc - 1, k) - Ly / 2) / delta);
                Byc[i][nyc - 1][k] = B0y;
                Bzc[i][nyc - 1][k] = B0z / cosh((grid->getYC(i, nyc - 1, k) - Ly / 2) / delta);;
                Bzc[i][nyc - 2][k] = B0z / cosh((grid->getYC(i, nyc - 2, k) - Ly / 2) / delta);;
                Bzc[i][nyc - 3][k] = B0z / cosh((grid->getYC(i, nyc - 3, k) - Ly / 2) / delta);
            }
    }

    if (vct->getYleft_neighbor() == MPI_PROC_NULL) 
    {
        for (int i = 0; i < nxc; i++)
            for (int k = 0; k < nzc; k++) 
            {
                Bxc[i][0][k] = B0x * tanh((grid->getYC(i, 0, k) - Ly / 2) / delta);
                Byc[i][0][k] = B0y;
                Bzc[i][0][k] = B0z / cosh((grid->getYC(i, 0, k) - Ly / 2) / delta);
                Bzc[i][1][k] = B0z / cosh((grid->getYC(i, 1, k) - Ly / 2) / delta);
                Bzc[i][2][k] = B0z / cosh((grid->getYC(i, 2, k) - Ly / 2) / delta);
            }
    }
}

// This method assumes mirror boundary conditions;
// we therefore need to double the density on the boundary
// nodes to incorporate the mirror particles from the mirror
// cell just outside the domain.
//
/*! adjust densities on boundaries that are not periodic */
void EMfields3D::adjustNonPeriodicDensities(int is)
{
  const VirtualTopology3D *vct = &get_vct();
  if (vct->getXleft_neighbor_P() == MPI_PROC_NULL) {
    for (int i = 1; i < nyn - 1; i++)
    for (int k = 1; k < nzn - 1; k++)
    {
        rhons[is][1][i][k] *= 2;
        Jxs  [is][1][i][k] *= 2;
        Jys  [is][1][i][k] *= 2;
        Jzs  [is][1][i][k] *= 2;
        pXXsn[is][1][i][k] *= 2;
        pXYsn[is][1][i][k] *= 2;
        pXZsn[is][1][i][k] *= 2;
        pYYsn[is][1][i][k] *= 2;
        pYZsn[is][1][i][k] *= 2;
        pZZsn[is][1][i][k] *= 2;
    }
  }
  if (vct->getYleft_neighbor_P() == MPI_PROC_NULL) {
    for (int i = 1; i < nxn - 1; i++)
    for (int k = 1; k < nzn - 1; k++)
    {
        rhons[is][i][1][k] *= 2;
        Jxs  [is][i][1][k] *= 2;
        Jys  [is][i][1][k] *= 2;
        Jzs  [is][i][1][k] *= 2;
        pXXsn[is][i][1][k] *= 2;
        pXYsn[is][i][1][k] *= 2;
        pXZsn[is][i][1][k] *= 2;
        pYYsn[is][i][1][k] *= 2;
        pYZsn[is][i][1][k] *= 2;
        pZZsn[is][i][1][k] *= 2;
    }
  }
  if (vct->getZleft_neighbor_P() == MPI_PROC_NULL) {
    for (int i = 1; i < nxn - 1; i++)
    for (int j = 1; j < nyn - 1; j++)
    {
        rhons[is][i][j][1] *= 2;
        Jxs  [is][i][j][1] *= 2;
        Jys  [is][i][j][1] *= 2;
        Jzs  [is][i][j][1] *= 2;
        pXXsn[is][i][j][1] *= 2;
        pXYsn[is][i][j][1] *= 2;
        pXZsn[is][i][j][1] *= 2;
        pYYsn[is][i][j][1] *= 2;
        pYZsn[is][i][j][1] *= 2;
        pZZsn[is][i][j][1] *= 2;
    }
  }
  if (vct->getXright_neighbor_P() == MPI_PROC_NULL) {
    for (int i = 1; i < nyn - 1; i++)
    for (int k = 1; k < nzn - 1; k++)
    {
        rhons[is][nxn - 2][i][k] *= 2;
        Jxs  [is][nxn - 2][i][k] *= 2;
        Jys  [is][nxn - 2][i][k] *= 2;
        Jzs  [is][nxn - 2][i][k] *= 2;
        pXXsn[is][nxn - 2][i][k] *= 2;
        pXYsn[is][nxn - 2][i][k] *= 2;
        pXZsn[is][nxn - 2][i][k] *= 2;
        pYYsn[is][nxn - 2][i][k] *= 2;
        pYZsn[is][nxn - 2][i][k] *= 2;
        pZZsn[is][nxn - 2][i][k] *= 2;
    }
  }
  if (vct->getYright_neighbor_P() == MPI_PROC_NULL) {
    for (int i = 1; i < nxn - 1; i++)
    for (int k = 1; k < nzn - 1; k++)
    {
        rhons[is][i][nyn - 2][k] *= 2;
        Jxs  [is][i][nyn - 2][k] *= 2;
        Jys  [is][i][nyn - 2][k] *= 2;
        Jzs  [is][i][nyn - 2][k] *= 2;
        pXXsn[is][i][nyn - 2][k] *= 2;
        pXYsn[is][i][nyn - 2][k] *= 2;
        pXZsn[is][i][nyn - 2][k] *= 2;
        pYYsn[is][i][nyn - 2][k] *= 2;
        pYZsn[is][i][nyn - 2][k] *= 2;
        pZZsn[is][i][nyn - 2][k] *= 2;
    }
  }
  if (vct->getZright_neighbor_P() == MPI_PROC_NULL) {
    for (int i = 1; i < nxn - 1; i++)
    for (int j = 1; j < nyn - 1; j++)
    {
        rhons[is][i][j][nzn - 2] *= 2;
        Jxs  [is][i][j][nzn - 2] *= 2;
        Jys  [is][i][j][nzn - 2] *= 2;
        Jzs  [is][i][j][nzn - 2] *= 2;
        pXXsn[is][i][j][nzn - 2] *= 2;
        pXYsn[is][i][j][nzn - 2] *= 2;
        pXZsn[is][i][j][nzn - 2] *= 2;
        pYYsn[is][i][j][nzn - 2] *= 2;
        pYZsn[is][i][j][nzn - 2] *= 2;
        pZZsn[is][i][j][nzn - 2] *= 2;
    }
  }
}

void EMfields3D::ConstantChargeOpenBCv2()
{
  const VirtualTopology3D *vct = &get_vct();
  const Grid *grid = &get_grid();

  double ff;

  int nx = grid->getNXN();
  int ny = grid->getNYN();
  int nz = grid->getNZN();

  for (int is = 0; is < ns; is++) {

	ff = qom[is]/fabs(qom[is]);

    if(vct->getXleft_neighbor()==MPI_PROC_NULL && bcEMfaceXleft ==2) {
      for (int j=0; j < ny;j++)
        for (int k=0; k < nz;k++){
          rhons[is][0][j][k] = rhons[is][4][j][k];
          rhons[is][1][j][k] = rhons[is][4][j][k];
          rhons[is][2][j][k] = rhons[is][4][j][k];
          rhons[is][3][j][k] = rhons[is][4][j][k];
        }
    }

    if(vct->getXright_neighbor()==MPI_PROC_NULL && bcEMfaceXright ==2) {
      for (int j=0; j < ny;j++)
        for (int k=0; k < nz;k++){
          rhons[is][nx-4][j][k] = rhons[is][nx-5][j][k];
          rhons[is][nx-3][j][k] = rhons[is][nx-5][j][k];
          rhons[is][nx-2][j][k] = rhons[is][nx-5][j][k];
          rhons[is][nx-1][j][k] = rhons[is][nx-5][j][k];
        }
    }

    if(vct->getYleft_neighbor()==MPI_PROC_NULL && bcEMfaceYleft ==2)  {
      for (int i=0; i < nx;i++)
        for (int k=0; k < nz;k++){
          rhons[is][i][0][k] = rhons[is][i][4][k];
          rhons[is][i][1][k] = rhons[is][i][4][k];
          rhons[is][i][2][k] = rhons[is][i][4][k];
          rhons[is][i][3][k] = rhons[is][i][4][k];
        }
    }

    if(vct->getYright_neighbor()==MPI_PROC_NULL && bcEMfaceYright ==2)  {
      for (int i=0; i < nx;i++)
        for (int k=0; k < nz;k++){
          rhons[is][i][ny-4][k] = rhons[is][i][ny-5][k];
          rhons[is][i][ny-3][k] = rhons[is][i][ny-5][k];
          rhons[is][i][ny-2][k] = rhons[is][i][ny-5][k];
          rhons[is][i][ny-1][k] = rhons[is][i][ny-5][k];
        }
    }

    if(vct->getZleft_neighbor()==MPI_PROC_NULL && bcEMfaceZleft ==2)  {
      for (int i=0; i < nx;i++)
        for (int j=0; j < ny;j++){
          rhons[is][i][j][0] = rhons[is][i][j][4];
          rhons[is][i][j][1] = rhons[is][i][j][4];
          rhons[is][i][j][2] = rhons[is][i][j][4];
          rhons[is][i][j][3] = rhons[is][i][j][4];
        }
    }


    if(vct->getZright_neighbor()==MPI_PROC_NULL && bcEMfaceZright ==2)  {
      for (int i=0; i < nx;i++)
        for (int j=0; j < ny;j++){
          rhons[is][i][j][nz-4] = rhons[is][i][j][nz-5];
          rhons[is][i][j][nz-3] = rhons[is][i][j][nz-5];
          rhons[is][i][j][nz-2] = rhons[is][i][j][nz-5];
          rhons[is][i][j][nz-1] = rhons[is][i][j][nz-5];
        }
    }
  }

}

void EMfields3D::ConstantChargeOpenBC()
{
  const VirtualTopology3D *vct = &get_vct();
  const Grid *grid = &get_grid();

  double ff;

  int nx = grid->getNXN();
  int ny = grid->getNYN();
  int nz = grid->getNZN();

  for (int is = 0; is < ns; is++) {

    ff = qom[is]/fabs(qom[is]);

    if(vct->getXleft_neighbor()==MPI_PROC_NULL && (bcEMfaceXleft ==2)) {
      for (int j=0; j < ny;j++)
        for (int k=0; k < nz;k++){
          rhons[is][0][j][k] = ff * rhoINIT[is] / FourPI;
          rhons[is][1][j][k] = ff * rhoINIT[is] / FourPI;
          rhons[is][2][j][k] = ff * rhoINIT[is] / FourPI;
          rhons[is][3][j][k] = ff * rhoINIT[is] / FourPI;
        }
    }

    if(vct->getXright_neighbor()==MPI_PROC_NULL && (bcEMfaceXright ==2)) {
      for (int j=0; j < ny;j++)
        for (int k=0; k < nz;k++){
          rhons[is][nx-4][j][k] = ff * rhoINIT[is] / FourPI;
          rhons[is][nx-3][j][k] = ff * rhoINIT[is] / FourPI;
          rhons[is][nx-2][j][k] = ff * rhoINIT[is] / FourPI;
          rhons[is][nx-1][j][k] = ff * rhoINIT[is] / FourPI;
        }
    }

    if(vct->getYleft_neighbor()==MPI_PROC_NULL && (bcEMfaceYleft ==2))  {
      for (int i=0; i < nx;i++)
        for (int k=0; k < nz;k++){
          rhons[is][i][0][k] = ff * rhoINIT[is] / FourPI;
          rhons[is][i][1][k] = ff * rhoINIT[is] / FourPI;
          rhons[is][i][2][k] = ff * rhoINIT[is] / FourPI;
          rhons[is][i][3][k] = ff * rhoINIT[is] / FourPI;
        }
    }

    if(vct->getYright_neighbor()==MPI_PROC_NULL && (bcEMfaceYright ==2))  {
      for (int i=0; i < nx;i++)
        for (int k=0; k < nz;k++){
          rhons[is][i][ny-4][k] = ff * rhoINIT[is] / FourPI;
          rhons[is][i][ny-3][k] = ff * rhoINIT[is] / FourPI;
          rhons[is][i][ny-2][k] = ff * rhoINIT[is] / FourPI;
          rhons[is][i][ny-1][k] = ff * rhoINIT[is] / FourPI;
        }
    }

    if(vct->getZleft_neighbor()==MPI_PROC_NULL && (bcEMfaceZleft ==2))  {
      for (int i=0; i < nx;i++)
        for (int j=0; j < ny;j++){
          rhons[is][i][j][0] = ff * rhoINIT[is] / FourPI;
          rhons[is][i][j][1] = ff * rhoINIT[is] / FourPI;
          rhons[is][i][j][2] = ff * rhoINIT[is] / FourPI;
          rhons[is][i][j][3] = ff * rhoINIT[is] / FourPI;
        }
    }


    if(vct->getZright_neighbor()==MPI_PROC_NULL && (bcEMfaceZright ==2))  {
      for (int i=0; i < nx;i++)
        for (int j=0; j < ny;j++){
          rhons[is][i][j][nz-4] = ff * rhoINIT[is] / FourPI;
          rhons[is][i][j][nz-3] = ff * rhoINIT[is] / FourPI;
          rhons[is][i][j][nz-2] = ff * rhoINIT[is] / FourPI;
          rhons[is][i][j][nz-1] = ff * rhoINIT[is] / FourPI;
        }
    }
  }

}

void EMfields3D::ConstantChargePlanet(double R, double x_center, double y_center, double z_center)
{
  const Grid *grid = &get_grid();

  double xd;
  double yd;
  double zd;
  double ff;

  for (int is = 0; is < ns; is++) {

	ff = qom[is]/fabs(qom[is]);

    for (int i = 1; i < nxn; i++) {
      for (int j = 1; j < nyn; j++) {
        for (int k = 1; k < nzn; k++) {

          xd = grid->getXN(i,j,k) - x_center;
          yd = grid->getYN(i,j,k) - y_center;
          zd = grid->getZN(i,j,k) - z_center;

          if ((xd*xd+yd*yd+zd*zd) <= R*R) {
            rhons[is][i][j][k] = ff * rhoINIT[is] / FourPI;
          }

        }
      }
    }
  }

}

void EMfields3D::ConstantChargePlanet2DPlaneXZ(double R,  double x_center,double z_center)
{
  const Grid *grid = &get_grid();
  //if (get_vct().getCartesian_rank() == 0)
      //cout << "*** Constant Charge 2D Planet ***" << endl;

  assert_eq(nyn,4);
  double xd;
  double zd;

  for (int is = 0; is < ns; is++) {
    const double sign_q = qom[is]/(fabs(qom[is]));
    for (int i = 1; i < nxn; i++)
        for (int k = 1; k < nzn; k++) {

          xd = grid->getXN(i,1,k) - x_center;
          zd = grid->getZN(i,1,k) - z_center;

          if ((xd*xd+zd*zd) <= R*R) {
            rhons[is][i][1][k] = sign_q * rhoINIT[is] / FourPI;
            rhons[is][i][2][k] = sign_q * rhoINIT[is] / FourPI;
          }

	}
  }

}

/*! initialize EM field with transverse electric waves 1D and rotate anticlockwise (theta degrees) */
void EMfields3D::initEM_rotate(double B, double theta)
{
  const Grid *grid = &get_grid();

  // initialize E and rhos on nodes
  for (int i = 0; i < nxn; i++)
    for (int j = 0; j < nyn; j++) {
      Ex[i][j][0] = 0.0;
      Ey[i][j][0] = 0.0;
      Ez[i][j][0] = 0.0;
      Bxn[i][j][0] = B * cos(theta * M_PI / 180);
      Byn[i][j][0] = B * sin(theta * M_PI / 180);
      Bzn[i][j][0] = 0.0;
      rhons[0][i][j][0] = 0.07957747154595; // electrons: species is now first index
      rhons[1][i][j][0] = 0.07957747154595; // protons: species is now first index
    }
  // initialize B on centers
  grid->interpN2C(Bxc, Bxn);
  grid->interpN2C(Byc, Byn);
  grid->interpN2C(Bzc, Bzn);


  for (int is = 0; is < ns; is++)
    grid->interpN2C(rhocs, is, rhons);
}

/*!Add a periodic perturbation in rho exp i(kx - \omega t); deltaBoB is the ratio (Delta B / B0) * */
void EMfields3D::AddPerturbationRho(double deltaBoB, double kx, double ky, double Bx_mod, double By_mod, double Bz_mod, double ne_mod, double ne_phase, double ni_mod, double ni_phase, double B0, Grid * grid) 
{
  double alpha;
  alpha = deltaBoB * B0 / sqrt(Bx_mod * Bx_mod + By_mod * By_mod + Bz_mod * Bz_mod);

  ne_mod *= alpha;
  ni_mod *= alpha;
  // cout<<" ne="<<ne_mod<<" ni="<<ni_mod<<" alpha="<<alpha<<endl;
  for (int i = 0; i < nxn; i++)
    for (int j = 0; j < nyn; j++) {
      rhons[0][i][j][0] += ne_mod * cos(kx * grid->getXN(i, j, 0) + ky * grid->getYN(i, j, 0) + ne_phase);
      rhons[1][i][j][0] += ni_mod * cos(kx * grid->getXN(i, j, 0) + ky * grid->getYN(i, j, 0) + ni_phase);
    }

  for (int is = 0; is < ns; is++)
    grid->interpN2C(rhocs, is, rhons);
}

/*!Add a periodic perturbation exp i(kx - \omega t); deltaBoB is the ratio (Delta B / B0) * */
void EMfields3D::AddPerturbation(double deltaBoB, double kx, double ky, double Ex_mod, double Ex_phase, double Ey_mod, double Ey_phase, double Ez_mod, double Ez_phase, 
                                                                        double Bx_mod, double Bx_phase, double By_mod, double By_phase, double Bz_mod, double Bz_phase, double B0, Grid * grid)
{
    double alpha = deltaBoB * B0 / sqrt(Bx_mod * Bx_mod + By_mod * By_mod + Bz_mod * Bz_mod);

    Ex_mod *= alpha;
    Ey_mod *= alpha;
    Ez_mod *= alpha;
    Bx_mod *= alpha;
    By_mod *= alpha;
    Bz_mod *= alpha;

    for (int i = 0; i < nxn; i++)
        for (int j = 0; j < nyn; j++) 
        {
            Ex[i][j][0] += Ex_mod * cos(kx * grid->getXN(i, j, 0) + ky * grid->getYN(i, j, 0) + Ex_phase);
            Ey[i][j][0] += Ey_mod * cos(kx * grid->getXN(i, j, 0) + ky * grid->getYN(i, j, 0) + Ey_phase);
            Ez[i][j][0] += Ez_mod * cos(kx * grid->getXN(i, j, 0) + ky * grid->getYN(i, j, 0) + Ez_phase);
            Bxn[i][j][0] += Bx_mod * cos(kx * grid->getXN(i, j, 0) + ky * grid->getYN(i, j, 0) + Bx_phase);
            Byn[i][j][0] += By_mod * cos(kx * grid->getXN(i, j, 0) + ky * grid->getYN(i, j, 0) + By_phase);
            Bzn[i][j][0] += Bz_mod * cos(kx * grid->getXN(i, j, 0) + ky * grid->getYN(i, j, 0) + Bz_phase);
        }

    // initialize B on centers
    grid->interpN2C(Bxc, Bxn);
    grid->interpN2C(Byc, Byn);
    grid->interpN2C(Bzc, Bzn);
}


//! ===================================== Helper Functions (Fields) ===================================== !//

//? Compute divergence of electric field
void EMfields3D::divergence_E(double ma) 
{
    const VirtualTopology3D * vct = &get_vct();

    scale(residual_divergence, divE_average, -1.0/FourPI, ns, nxc, nyc, nzc);
    addscale(1.0, residual_divergence, rhoc_avg, ns, nxc, nyc, nzc);

    for (int is = 0; is < ns; is++)
        for (int i = 0; i < nxc; i++)
            for (int j = 0; j < nyc; j++)
                for (int k = 0; k < nzc; k++) 
                    residual_divergence.fetch(is, i, j, k) = residual_divergence.get(is, i, j, k)/(rhocs_avg.get(is, i, j, k) - 1e-10) * ma;

    for (int is = 0; is < ns; is++)
        communicateCenterBC(nxc, nyc, nzc, residual_divergence[is], 2, 2, 2, 2, 2, 2, vct, this);
}

//? Compute divergence of magnetic field
void EMfields3D::divergence_B() 
{
    const Grid *grid = &get_grid();
    grid->divC2N(divB, Bxc, Byc, Bzc);
}

void EMfields3D::timeAveragedRho(double ma) 
{
    //* rho_average = (1-ma)*rho_average + ma*rho
    scale(rhoc_avg, (1-ma), nxc, nyc, nzc);
    addscale(ma, rhoc_avg, rhoc, nxc, nyc, nzc);
}

//! Write to restart files
//TODO: Write "rhoc_avg", "rhocs_avg" and "divE_average"

void EMfields3D::timeAveragedDivE(double ma) 
{
    // TODO: Boundary conditions - TBD later
    // EMfields3D::BC_E_Poisson(vct,  Ex, Ey, Ez);

    const Grid *grid = &get_grid();
    grid->divN2C(divE, Ex, Ey, Ez);

    scale(divE_average, (1.0-ma), nxc, nyc, nzc);
    addscale(ma, divE_average, divE, nxc, nyc, nzc);
}


//! ===================================== Helper Functions (Moments) ===================================== !//

//? Set all elements of mass matrix to 0
void EMfields3D::setZeroMassMatrix()
{
    for (int c = 0; c < NE_MASS; c++) 
        for (int i = 0; i < nxn; i++)
            for (int j = 0; j < nyn; j++)
                for (int k = 0; k < nzn; k++) 
                {
                    Mxx[c][i][j][k] = 0.0;
                    Mxy[c][i][j][k] = 0.0;
                    Mxz[c][i][j][k] = 0.0;
                    Myx[c][i][j][k] = 0.0;
                    Myy[c][i][j][k] = 0.0;
                    Myz[c][i][j][k] = 0.0;
                    Mzx[c][i][j][k] = 0.0;
                    Mzy[c][i][j][k] = 0.0;
                    Mzz[c][i][j][k] = 0.0;
                }
}

//? Set the derived moments to zero
void EMfields3D::setZeroDerivedMoments()
{
    for (int i = 0; i < nxn; i++)
        for (int j = 0; j < nyn; j++)
            for (int k = 0; k < nzn; k++)
            {
                Jx[i][j][k] = 0.0;        //* J along X
                Jy[i][j][k] = 0.0;        //* J along Y
                Jz[i][j][k] = 0.0;        //* J along Z
                Jxh[i][j][k] = 0.0;       //* J hat along X
                Jyh[i][j][k] = 0.0;       //* J hat along Y
                Jzh[i][j][k] = 0.0;       //* J hat along Z
                rhon[i][j][k] = 0.0;       //* J hat along Z
            }

    eqValue(0.0, rhoc, nxc, nyc, nzc);
    eqValue(0.0, rhocs, ns, nxc, nyc, nzc);
}

//? Set the primary moments to zero
void EMfields3D::setZeroPrimaryMoments() 
{
    for (int is = 0; is < ns; is++) 
        for (int i = 0; i < nxn; i++)
            for (int j = 0; j < nyn; j++)
                for (int k = 0; k < nzn; k++) 
                {
                    Jxs[is][i][j][k] = 0.0;       //* J along X, for each species, at nodes
                    Jys[is][i][j][k] = 0.0;       //* J along Y, for each species, at nodes
                    Jzs[is][i][j][k] = 0.0;       //* J along Z, for each species, at nodes
                    Jxhs[is][i][j][k] = 0.0;      //* J hat along X, for each species, at nodes
                    Jyhs[is][i][j][k] = 0.0;      //* J hat along Y, for each species, at nodes
                    Jzhs[is][i][j][k] = 0.0;      //* J hat along Z, for each species, at nodes
                    rhons[is][i][j][k] = 0.0;     //* Rho, for each species, at nodes
                }
}

//? Set all moments to zero
void EMfields3D::setZeroDensities() 
{
    setZeroRho(); 
    setZeroDerivedMoments();
    setZeroPrimaryMoments();
    setZeroMassMatrix();
}

//? Set densities (at nodes and cell centres) of all species to 0
void EMfields3D::setZeroRho()
{
    eqValue(0.0, rhons, ns, nxn, nyn, nzn);     //* Rho, for each species, at nodes
    eqValue(0.0, rhocs, ns, nxc, nyc, nzc);     //* Rho, for each species, at cell centres  
    // eqValue(0.0, Nns, ns, nxn, nyn, nzn);       //*
}

//* Sum charge and current (hat) density of different species (used in computeMoments())
void EMfields3D::sumOverSpecies()
{
    const Grid *grid = &get_grid();
    const VirtualTopology3D * vct = &get_vct();

    for (int is = 0; is < ns; is++)
        for (int i = 0; i < nxn; i++)
            for (int j = 0; j < nyn; j++)
                for (int k = 0; k < nzn; k++)
                {
                    rhon[i][j][k]  += rhons[is][i][j][k];
                    Jxh[i][j][k]   += Jxhs[is][i][j][k];
                    Jyh[i][j][k]   += Jyhs[is][i][j][k];
                    Jzh[i][j][k]   += Jzhs[is][i][j][k];
                }

    communicateNode_P(nxn, nyn, nzn, rhon, vct, this);
    grid->interpN2C(rhoc, rhon);
    communicateCenterBC(nxc, nyc, nzc, rhoc, 2, 2, 2, 2, 2, 2, vct, this);
}

//* Sum mass and charge density of different species (on nodes) *//
// void EMfields3D::sumOverSpeciesRho()
// {
//     for (int is = 0; is < ns; is++)
//         for (int i = 0; i < nxn; i++)
//             for (int j = 0; j < nyn; j++)
//                 for (int k = 0; k < nzn; k++)
//                     rhon[i][j][k] += rhons[is][i][j][k];
// }

// //* Sum current density for different species //
// void EMfields3D::sumOverSpeciesJ() 
// {
//     for (int is = 0; is < ns; is++)
//         for (int i = 0; i < nxn; i++)
//             for (int j = 0; j < nyn; j++)
//                 for (int k = 0; k < nzn; k++) 
//                 {
//                     Jx[i][j][k] += Jxs[is][i][j][k];
//                     Jy[i][j][k] += Jys[is][i][j][k];
//                     Jz[i][j][k] += Jzs[is][i][j][k];
//                 }
// }

//* Sum charge and current density of different species (used in SupplementaryMoments())
void EMfields3D::sumOverSpecies_supplementary() 
{
    for (int is = 0; is < ns; is++)
        for (int i = 0; i < nxn; i++)
            for (int j = 0; j < nyn; j++)
                for (int k = 0; k < nzn; k++) 
                {
                    rhon[i][j][k] += rhons[is][i][j][k];
                    Jx[i][j][k]   += Jxs[is][i][j][k];
                    Jy[i][j][k]   += Jys[is][i][j][k];
                    Jz[i][j][k]   += Jzs[is][i][j][k];
                }
}

void EMfields3D::interpolateCenterSpecies(int is) 
{
    const Grid *grid = &get_grid();
    const VirtualTopology3D * vct = &get_vct();

    grid->interpN2C(rhocs_avg, is, rhons);
    communicateCenterBC(nxc, nyc, nzc, rhocs_avg[is], 2, 2, 2, 2, 2, 2, vct, this);
}

// void EMfields3D::setZeroCurrent()
// {
//     const VirtualTopology3D *vct = &get_vct();

//     for (int k=1; k<nzn-1; k++) 
//         for (int j=1; j<nyn-1; j++) 
//             for (int i=1; i<nxn-1; i++) 
//             {
//                 Jx_ext.fetch(i, j, k) = 0.0;
//                 Jx_ext.fetch(i, j, k) = 0.0;
//                 Jx_ext.fetch(i, j, k) = 0.0;

//                 for (int s=0; s<ns; s++) 
//                 {
//                     Jx_ext.fetch(i, j, k) -= Jxs.get(s, i, j, k);
//                     Jy_ext.fetch(i, j, k) -= Jys.get(s, i, j, k);
//                     Jz_ext.fetch(i, j, k) -= Jzs.get(s, i, j, k);
//                 }
//             }

//     communicateInterp(nxn, nyn, nzn, Jx_ext, vct, this);
//     communicateInterp(nxn, nyn, nzn, Jy_ext, vct, this);
//     communicateInterp(nxn, nyn, nzn, Jz_ext, vct, this);
  
//     communicateNode_P(nxn, nyn, nzn, Jx_ext, vct, this);
//     communicateNode_P(nxn, nyn, nzn, Jy_ext, vct, this);
//     communicateNode_P(nxn, nyn, nzn, Jz_ext, vct, this);
// }

//! ===================================== Initial Field Distributions ===================================== !//

//! Initial field distributions (Non Relativistic) !//

//* Default electric and magnetic field configurations
void EMfields3D::init()
{
    const Collective *col = &get_col();
    const VirtualTopology3D *vct = &get_vct();
    const Grid *grid = &get_grid();

    if (restart_status == 0)
    {
        if (vct->getCartesian_rank() == 0) 
            cout << "Default field initialisation; initial magnetic field components (Bx, By, Bz) = " << "(" << B0x << ", " << B0y << ", " << B0z << ")" << endl;

        //! Initial setup (NOT RESTART)

        for (int i = 0; i < nxn; i++)
            for (int j = 0; j < nyn; j++)
                for (int k = 0; k < nzn; k++)
                {
                    //* Initialise E on nodes (if external E exixts, this needs to be initalised here)
                    Ex[i][j][k] = 0.0;
                    Ey[i][j][k] = 0.0;
                    Ez[i][j][k] = 0.0;

                    //* Initialise B on nodes
                    Bxn[i][j][k] = B0x;
                    Byn[i][j][k] = B0y;
                    Bzn[i][j][k] = B0z;

                    //* Initialize rho on nodes
                    for (int is = 0; is < ns; is++)
                        rhons[is][i][j][k] = rhoINIT[is] / FourPI;
                }

        //* Initialise B and rho on cell centers
        grid->interpN2C(Bxc, Bxn);
        grid->interpN2C(Byc, Byn);
        grid->interpN2C(Bzc, Bzn);

        //* Communicate ghost data on cell centres
        communicateCenterBC(nxc, nyc, nzc, Bxc, col->bcBx[0],col->bcBx[1],col->bcBx[2],col->bcBx[3],col->bcBx[4],col->bcBx[5], vct, this);
        communicateCenterBC(nxc, nyc, nzc, Byc, col->bcBy[0],col->bcBy[1],col->bcBy[2],col->bcBy[3],col->bcBy[4],col->bcBy[5], vct, this);
        communicateCenterBC(nxc, nyc, nzc, Bzc, col->bcBz[0],col->bcBz[1],col->bcBz[2],col->bcBz[3],col->bcBz[4],col->bcBz[5], vct, this);

        for (int is = 0; is < ns; is++)
            grid->interpN2C(rhocs, is, rhons);
    }
    else
    {   
        //! READ FROM RESTART
        #ifdef NO_HDF5
            eprintf("restart requires compiling with HDF5");
        #else
            
            col->read_field_restart(vct, grid, Bxn, Byn, Bzn, Bxc, Byc, Bzc, Ex, Ey, Ez, rhoc_avg, divE_average, ns);

            //* Communicate ghost data for rho on nodes
            // for (int is = 0; is < ns; is++) 
            // {
            //     double ***moment0 = convert_to_arr3(rhons[is]);
            //     communicateNode_P(nxn, nyn, nzn, moment0, vct, this);
            // }

            if (col->getCase()=="Dipole") 
                ConstantChargePlanet(col->getL_square(),col->getx_center(),col->gety_center(),col->getz_center());
            else if (col->getCase()=="Dipole2D") 
                ConstantChargePlanet2DPlaneXZ(col->getL_square(),col->getx_center(),col->getz_center());
            // I am not sure what this open BC does, but perhaps it is responsible for energy losses in the restart? Jan 2017, Slavik.
            else if ((col->getCase().find("TaylorGreen") != std::string::npos) && (col->getCase() != "NullPoints"))
                ConstantChargeOpenBC();

            //* Communicate ghost data for B at cell centres
            communicateCenterBC(nxc, nyc, nzc, Bxc, col->bcBx[0],col->bcBx[1],col->bcBx[2],col->bcBx[3],col->bcBx[4],col->bcBx[5], vct,this);
            communicateCenterBC(nxc, nyc, nzc, Byc, col->bcBy[0],col->bcBy[1],col->bcBy[2],col->bcBy[3],col->bcBy[4],col->bcBy[5], vct,this);
            communicateCenterBC(nxc, nyc, nzc, Bzc, col->bcBz[0],col->bcBz[1],col->bcBz[2],col->bcBz[3],col->bcBz[4],col->bcBz[5], vct,this);

            //* Communicate ghost data for rhoc_avg at cell centres
            communicateCenterBC(nxc, nyc, nzc, rhoc_avg, col->bcBx[0],col->bcBx[1],col->bcBx[2],col->bcBx[3],col->bcBx[4],col->bcBx[5], vct,this);

            // grid->interpC2N(Bxn, Bxc);
            // grid->interpC2N(Byn, Byc);
            // grid->interpC2N(Bzn, Bzc);

            //* Communicate ghost data for B on nodes
            communicateNodeBC_old(nxn, nyn, nzn, Bxn, col->bcBx[0],col->bcBx[1],col->bcBx[2],col->bcBx[3],col->bcBx[4],col->bcBx[5], vct, this);
            communicateNodeBC_old(nxn, nyn, nzn, Byn, col->bcBy[0],col->bcBy[1],col->bcBy[2],col->bcBy[3],col->bcBy[4],col->bcBy[5], vct, this);
            communicateNodeBC_old(nxn, nyn, nzn, Bzn, col->bcBz[0],col->bcBz[1],col->bcBz[2],col->bcBz[3],col->bcBz[4],col->bcBz[5], vct, this);

            //* Communicate ghost data for E on nodes
            communicateNodeBC_old(nxn, nyn, nzn, Ex, col->bcEx[0],col->bcEx[1],col->bcEx[2],col->bcEx[3],col->bcEx[4],col->bcEx[5], vct, this);
            communicateNodeBC_old(nxn, nyn, nzn, Ey, col->bcEy[0],col->bcEy[1],col->bcEy[2],col->bcEy[3],col->bcEy[4],col->bcEy[5], vct, this);
            communicateNodeBC_old(nxn, nyn, nzn, Ez, col->bcEz[0],col->bcEz[1],col->bcEz[2],col->bcEz[3],col->bcEz[4],col->bcEz[5], vct, this);

            //* Initialise rho at cell centers
            // for (int is = 0; is < ns; is++)
            //     grid->interpN2C(rhocs, is, rhons);

            if (vct->getCartesian_rank() == 0)
            {
                cout << "--------------------------------------------------------" << endl;
                cout << "SUCCESSFULLY READ FIELD DATA FROM HDF5 FILES FOR RESTART" << endl;
            }
        
        #endif // NO_HDF5
    }
}

#ifdef BATSRUS
void EMfields3D::initBATSRUS()
{
    const Collective *col = &get_col();
    const Grid *grid = &get_grid();
    cout << "------------------------------------------" << endl;
    cout << "         Initialize from BATSRUS          " << endl;
    cout << "------------------------------------------" << endl;

    // loop over species and cell centers: fill in charge density
    for (int is=0; is < ns; is++)
        for (int i=0; i < nxc; i++)
            for (int j=0; j < nyc; j++)
                for (int k=0; k < nzc; k++)
                {
                // WARNING getFluidRhoCenter contains "case" statment
                rhocs[is][i][j][k] = col->getFluidRhoCenter(i,j,k,is);
                }

    // loop over cell centers and fill in magnetic and electric fields
    for (int i=0; i < nxc; i++)
        for (int j=0; j < nyc; j++)
            for (int k=0; k < nzc; k++)
            {
                // WARNING getFluidRhoCenter contains "case" statment
                col->setFluidFieldsCenter(&Ex[i][j][k],&Ey[i][j][k],&Ez[i][j][k],
                    &Bxc[i][j][k],&Byc[i][j][k],&Bzc[i][j][k],i,j,k);
            }

    // interpolate from cell centers to nodes (corners of cells)
    for (int is=0 ; is<ns; is++)
        grid->interpC2N(rhons[is],rhocs[is]);
        
    grid->interpC2N(Bxn,Bxc);
    grid->interpC2N(Byn,Byc);
    grid->interpC2N(Bzn,Bzc);
}
#endif

/*! initiliaze EM for GEM challange */
void EMfields3D::initGEM()
{
    //! Initial setup (NOT RESTART)
    const VirtualTopology3D *vct = &get_vct();
    const Grid *grid = &get_grid();
    
    //* Perturbation localized in X
    double pertX = 0.4;
    double xpert, ypert, exp_pert;
    
    if (restart_status == 0) 
    {
    // initialize
    if (get_vct().getCartesian_rank() == 0) {
      cout << "------------------------------------------" << endl;
      cout << "Initialize GEM Challenge with Perturbation" << endl;
      cout << "------------------------------------------" << endl;
      cout << "B0x                              = " << B0x << endl;
      cout << "B0y                              = " << B0y << endl;
      cout << "B0z                              = " << B0z << endl;
      cout << "Delta (current sheet thickness) = " << delta << endl;
      for (int i = 0; i < ns; i++) {
        cout << "rho species " << i << " = " << rhoINIT[i];
        if (DriftSpecies[i])
          cout << " DRIFTING " << endl;
        else
          cout << " BACKGROUND " << endl;
      }
      cout << "-------------------------" << endl;
    }
    for (int i = 0; i < nxn; i++)
      for (int j = 0; j < nyn; j++)
        for (int k = 0; k < nzn; k++) {
          // initialize the density for species
          for (int is = 0; is < ns; is++) {
            if (DriftSpecies[is])
              rhons[is][i][j][k] = ((rhoINIT[is] / (cosh((grid->getYN(i, j, k) - Ly / 2) / delta) * cosh((grid->getYN(i, j, k) - Ly / 2) / delta)))) / FourPI;
            else
              rhons[is][i][j][k] = rhoINIT[is] / FourPI;
          }
          // electric field
          Ex[i][j][k] = 0.0;
          Ey[i][j][k] = 0.0;
          Ez[i][j][k] = 0.0;
          // Magnetic field
          Bxn[i][j][k] = B0x * tanh((grid->getYN(i, j, k) - Ly / 2) / delta);
          // add the initial GEM perturbation
          // Bxn[i][j][k] += (B0x/10.0)*(M_PI/Ly)*cos(2*M_PI*grid->getXN(i,j,k)/Lx)*sin(M_PI*(grid->getYN(i,j,k)- Ly/2)/Ly );
          Byn[i][j][k] = B0y;   // - (B0x/10.0)*(2*M_PI/Lx)*sin(2*M_PI*grid->getXN(i,j,k)/Lx)*cos(M_PI*(grid->getYN(i,j,k)- Ly/2)/Ly); 
          // add the initial X perturbation
          xpert = grid->getXN(i, j, k) - Lx / 2;
          ypert = grid->getYN(i, j, k) - Ly / 2;
          exp_pert = exp(-(xpert / delta) * (xpert / delta) - (ypert / delta) * (ypert / delta));
          Bxn[i][j][k] += (B0x * pertX) * exp_pert * (-cos(M_PI * xpert / 10.0 / delta) * cos(M_PI * ypert / 10.0 / delta) * 2.0 * ypert / delta - cos(M_PI * xpert / 10.0 / delta) * sin(M_PI * ypert / 10.0 / delta) * M_PI / 10.0);
          Byn[i][j][k] += (B0x * pertX) * exp_pert * (cos(M_PI * xpert / 10.0 / delta) * cos(M_PI * ypert / 10.0 / delta) * 2.0 * xpert / delta + sin(M_PI * xpert / 10.0 / delta) * cos(M_PI * ypert / 10.0 / delta) * M_PI / 10.0);
          // guide field
          Bzn[i][j][k] = B0z;
        }
    // initialize B on centers
    for (int i = 0; i < nxc; i++)
      for (int j = 0; j < nyc; j++)
        for (int k = 0; k < nzc; k++) {
          // Magnetic field
          Bxc[i][j][k] = B0x * tanh((grid->getYC(i, j, k) - Ly / 2) / delta);
          // add the initial GEM perturbation
          // Bxc[i][j][k] += (B0x/10.0)*(M_PI/Ly)*cos(2*M_PI*grid->getXC(i,j,k)/Lx)*sin(M_PI*(grid->getYC(i,j,k)- Ly/2)/Ly );
          Byc[i][j][k] = B0y;   // - (B0x/10.0)*(2*M_PI/Lx)*sin(2*M_PI*grid->getXC(i,j,k)/Lx)*cos(M_PI*(grid->getYC(i,j,k)- Ly/2)/Ly); 
          // add the initial X perturbation
          xpert = grid->getXC(i, j, k) - Lx / 2;
          ypert = grid->getYC(i, j, k) - Ly / 2;
          exp_pert = exp(-(xpert / delta) * (xpert / delta) - (ypert / delta) * (ypert / delta));
          Bxc[i][j][k] += (B0x * pertX) * exp_pert * (-cos(M_PI * xpert / 10.0 / delta) * cos(M_PI * ypert / 10.0 / delta) * 2.0 * ypert / delta - cos(M_PI * xpert / 10.0 / delta) * sin(M_PI * ypert / 10.0 / delta) * M_PI / 10.0);
          Byc[i][j][k] += (B0x * pertX) * exp_pert * (cos(M_PI * xpert / 10.0 / delta) * cos(M_PI * ypert / 10.0 / delta) * 2.0 * xpert / delta + sin(M_PI * xpert / 10.0 / delta) * cos(M_PI * ypert / 10.0 / delta) * M_PI / 10.0);
          // guide field
          Bzc[i][j][k] = B0z;

        }
    for (int is = 0; is < ns; is++)
      grid->interpN2C(rhocs, is, rhons);
  }
  else {
    init(); // use the fields from restart file
  }
}

void EMfields3D::initNullPoints()
{
	const VirtualTopology3D *vct = &get_vct();
	const Grid *grid = &get_grid();
	if (restart_status ==0){
		if (vct->getCartesian_rank() ==0){
			cout << "----------------------------------------" << endl;
			cout << "       Initialize 3D null point(s)" << endl;
			cout << "----------------------------------------" << endl;
			cout << "B0x                              = " << B0x << endl;
			cout << "B0y                              = " << B0y << endl;
			cout << "B0z                              = " << B0z << endl;
			for (int i=0; i < ns; i++){
				cout << "rho species " << i <<" = " << rhoINIT[i] << endl;
			}
			cout << "Smoothing Factor = " << Smooth << endl;
			cout << "-------------------------" << endl;
		}

        for (int i=0; i < nxn; i++)
		for (int j=0; j < nyn; j++)
		for (int k=0; k < nzn; k++){
		   // initialize the density for species
		   for (int is=0; is < ns; is++)
			   rhons[is][i][j][k] = rhoINIT[is]/FourPI;

			// electric field
			Ex[i][j][k] =  0.0;
			Ey[i][j][k] =  0.0;
			Ez[i][j][k] =  0.0;
			// Magnetic field
			Bxn[i][j][k] = -B0x*cos(2.*M_PI*grid->getXN(i,j,k)/Lx)*sin(2.*M_PI*grid->getYN(i,j,k)/Ly);
			Byn[i][j][k] = B0x*cos(2.*M_PI*grid->getYN(i,j,k)/Ly)*(-2.*sin(2.*M_PI*grid->getZN(i,j,k)/Lz) + sin(2.*M_PI*grid->getXN(i,j,k)/Lx));
			Bzn[i][j][k] = 2.*B0x*cos(2.*M_PI*grid->getZN(i,j,k)/Lz)*sin(2.*M_PI*grid->getYN(i,j,k)/Ly);
		}

		for (int i=0; i <nxc; i++)
		for (int j=0; j <nyc; j++)
		for (int k=0; k <nzc; k++) {
			Bxc[i][j][k] = -B0x*cos(2.*M_PI*grid->getXC(i,j,k)/Lx)*sin(2.*M_PI*grid->getYC(i,j,k)/Ly);
			Byc[i][j][k] = B0x*cos(2.*M_PI*grid->getYC(i,j,k)/Ly)*(-2.*sin(2.*M_PI*grid->getZC(i,j,k)/Lz) + sin(2.*M_PI*grid->getXC(i,j,k)/Lx));
			Bzc[i][j][k] = 2.*B0x*cos(2.*M_PI*grid->getZC(i,j,k)/Lz)*sin(2.*M_PI*grid->getYC(i,j,k)/Ly);
		}

	    // currents are used to calculate in the Maxwell's solver
	    // The ion current is equal to 0 (all current is on electrons)
	    for (int i=0; i < nxn; i++)
		for (int j=0; j < nyn; j++)
		for (int k=0; k < nzn; k++){
			Jxs[1][i][j][k] = 0.0; // ion species is species 1
			Jys[1][i][j][k] = 0.0; // ion species is species 1
			Jzs[1][i][j][k] = 0.0; // ion species is species 1
		}

	    // calculate the electron current from
        eqValue(0.0,tempXN,nxn,nyn,nzn);
        eqValue(0.0,tempYN,nxn,nyn,nzn);
        eqValue(0.0,tempZN,nxn,nyn,nzn);
        grid->curlC2N(tempXN,tempYN,tempZN,Bxc,Byc,Bzc); // here you calculate curl(B)
        // all current is on electrons, calculated from Ampere's law
        for (int i=0; i < nxn; i++)
        for (int j=0; j < nyn; j++)
        for (int k=0; k < nzn; k++){  // electrons are species 0
			Jxs[0][i][j][k] = c*tempXN[i][j][k]/FourPI; // ion species is species 1
			Jys[0][i][j][k] = c*tempYN[i][j][k]/FourPI; // ion species is species 1
			Jzs[0][i][j][k] = c*tempZN[i][j][k]/FourPI; // ion species is species 1
		}

		for (int is=0 ; is<ns; is++)
			grid->interpN2C(rhocs,is,rhons);
	} else {
		init();  // use the fields from restart file
	}
}

void EMfields3D::initTaylorGreen()
{
  const VirtualTopology3D *vct = &get_vct();
  const Grid *grid = &get_grid();
  if (restart_status ==0){
    if (vct->getCartesian_rank() ==0){
      cout << "----------------------------------------" << endl;
      cout << "       Initialize Taylor-Green flow     " << endl;
      cout << "----------------------------------------" << endl;
      cout << "B0                               = " << B0x << endl;
      cout << "u0                               = " << ue0 << endl;
      for (int i=0; i < ns; i++){
	cout << "rho species " << i <<" = " << rhoINIT[i] << endl;
      }
      cout << "Smoothing Factor = " << Smooth << endl;
      cout << "-------------------------" << endl;
    }

    for (int i=0; i < nxn; i++)
      for (int j=0; j < nyn; j++)
	for (int k=0; k < nzn; k++){
	  // initialize the density for species
	  for (int is=0; is < ns; is++) {
	    rhons[is][i][j][k] = rhoINIT[is]/FourPI;
             
	    // The flow will be initialized from currents
	    Jxs[is][i][j][k] = ue0 * rhons[is][i][j][k] * sin(2.*M_PI*grid->getXC(i,j,k)/Lx) * cos(2.*M_PI*grid->getYC(i,j,k)/Ly) * cos(2.*M_PI*grid->getZC(i,j,k)/Lz); 
	    Jys[is][i][j][k] = -ue0 * rhons[is][i][j][k] * cos(2.*M_PI*grid->getXC(i,j,k)/Lx) * sin(2.*M_PI*grid->getYC(i,j,k)/Ly) * cos(2.*M_PI*grid->getZC(i,j,k)/Lz);
	    Jzs[is][i][j][k] = 0.; // Z velocity is zero
	  }

	  // electric field
	  Ex[i][j][k] =  0.0;
	  Ey[i][j][k] =  0.0;
	  Ez[i][j][k] =  0.0;
	  // Magnetic field
	  Bxn[i][j][k] = B0x * cos(2.*M_PI*grid->getXN(i,j,k)/Lx) * sin(2.*M_PI*grid->getYN(i,j,k)/Ly) * sin(2.*M_PI*grid->getZN(i,j,k)/Lz);
	  Byn[i][j][k] = B0x * sin(2.*M_PI*grid->getXN(i,j,k)/Lx) * cos(2.*M_PI*grid->getYN(i,j,k)/Ly) * sin(2.*M_PI*grid->getZN(i,j,k)/Lz);
	  Bzn[i][j][k] = -2. * B0x  * sin(2.*M_PI*grid->getXN(i,j,k)/Lx) * sin(2.*M_PI*grid->getYN(i,j,k)/Ly) * cos(2.*M_PI*grid->getZN(i,j,k)/Lz);
	}

    for (int i=0; i <nxc; i++)
      for (int j=0; j <nyc; j++)
	for (int k=0; k <nzc; k++) {
	  Bxc[i][j][k] = B0x * cos(2.*M_PI*grid->getXC(i,j,k)/Lx) * sin(2.*M_PI*grid->getYC(i,j,k)/Ly) * sin(2.*M_PI*grid->getZC(i,j,k)/Lz);
	  Byc[i][j][k] = B0x * sin(2.*M_PI*grid->getXC(i,j,k)/Lx) * cos(2.*M_PI*grid->getYC(i,j,k)/Ly) * sin(2.*M_PI*grid->getZC(i,j,k)/Lz);
	  Bzc[i][j][k] = -2. * B0x * sin(2.*M_PI*grid->getXC(i,j,k)/Lx) * sin(2.*M_PI*grid->getYC(i,j,k)/Ly) * cos(2.*M_PI*grid->getZC(i,j,k)/Lz);
	}

    for (int is=0 ; is<ns; is++)
      grid->interpN2C(rhocs,is,rhons);
  } else {
    init();  // use the fields from restart file
  }
}

void EMfields3D::initOriginalGEM()
{
  const Grid *grid = &get_grid();
  // perturbation localized in X
  if (restart_status == 0) {
    // initialize
    if (get_vct().getCartesian_rank() == 0) {
      cout << "------------------------------------------" << endl;
      cout << "Initialize GEM Challenge with Pertubation" << endl;
      cout << "------------------------------------------" << endl;
      cout << "B0x                              = " << B0x << endl;
      cout << "B0y                              = " << B0y << endl;
      cout << "B0z                              = " << B0z << endl;
      cout << "Delta (current sheet thickness) = " << delta << endl;
      for (int i = 0; i < ns; i++) {
        cout << "rho species " << i << " = " << rhoINIT[i];
        if (DriftSpecies[i])
          cout << " DRIFTING " << endl;
        else
          cout << " BACKGROUND " << endl;
      }
      cout << "-------------------------" << endl;
    }
    for (int i = 0; i < nxn; i++)
      for (int j = 0; j < nyn; j++)
        for (int k = 0; k < nzn; k++) {
          // initialize the density for species
          for (int is = 0; is < ns; is++) {
            if (DriftSpecies[is])
              rhons[is][i][j][k] = ((rhoINIT[is] / (cosh((grid->getYN(i, j, k) - Ly / 2) / delta) * cosh((grid->getYN(i, j, k) - Ly / 2) / delta)))) / FourPI;
            else
              rhons[is][i][j][k] = rhoINIT[is] / FourPI;
          }
          // electric field
          Ex[i][j][k] = 0.0;
          Ey[i][j][k] = 0.0;
          Ez[i][j][k] = 0.0;
          // Magnetic field
          const double yM = grid->getYN(i, j, k) - .5 * Ly;
          Bxn[i][j][k] = B0x * tanh(yM / delta);
          // add the initial GEM perturbation
          const double xM = grid->getXN(i, j, k) - .5 * Lx;
          Bxn[i][j][k] -= (B0x / 10.0) * (M_PI / Ly) * cos(2 * M_PI * xM / Lx) * sin(M_PI * yM / Ly);
          Byn[i][j][k] = B0y + (B0x / 10.0) * (2 * M_PI / Lx) * sin(2 * M_PI * xM / Lx) * cos(M_PI * yM / Ly);
          Bzn[i][j][k] = B0z;
        }
    // initialize B on centers
    for (int i = 0; i < nxc; i++)
      for (int j = 0; j < nyc; j++)
        for (int k = 0; k < nzc; k++) {
          // Magnetic field
          const double yM = grid->getYC(i, j, k) - .5 * Ly;
          Bxc[i][j][k] = B0x * tanh(yM / delta);
          // add the initial GEM perturbation
          const double xM = grid->getXC(i, j, k) - .5 * Lx;
          Bxc[i][j][k] -= (B0x / 10.0) * (M_PI / Ly) * cos(2 * M_PI * xM / Lx) * sin(M_PI * yM / Ly);
          Byc[i][j][k] = B0y + (B0x / 10.0) * (2 * M_PI / Lx) * sin(2 * M_PI * xM / Lx) * cos(M_PI * yM / Ly);
          Bzc[i][j][k] = B0z;
        }
    for (int is = 0; is < ns; is++)
      grid->interpN2C(rhocs, is, rhons);
  }
  else {
    init(); // use the fields from restart file
  }
}

//* Initialise double Harris sheets for magnetic reconnection
void EMfields3D::init_double_Harris()
{
    const Collective *col = &get_col();
    const VirtualTopology3D *vct = &get_vct();
    const Grid *grid = &get_grid();

    //* Custom input parameters
    const double perturbation               = input_param[0];       //* Amplitude of initial perturbation
    const double delta                      = input_param[1];       //* Half-thickness of current sheet

    if (restart_status == 0)
    {
        if (vct->getCartesian_rank() ==0)
        {
            cout << "------------------------------------------" << endl;
            cout << "     Initialising double Harris sheet     " << endl;
            cout << "------------------------------------------" << endl;
            cout << "Initial magnetic field components (Bx, By, Bz) = " << "(" << B0x << ", " << B0y << ", " << B0z << ")" << endl;
            cout << "Initial perturbation                           = " << perturbation << endl;
            cout << "Half-thickness of current sheet                = " << delta << endl;
            cout << "------------------------------------------" << endl;
        }

        for (int i = 0; i < nxn; i++)
            for (int j = 0; j < nyn; j++)
	            for (int k = 0; k < nzn; k++)
                {
                    double global_x = grid->getXN(i, j, k) + grid->getDX();
                    double global_y = grid->getYN(i, j, k) + grid->getDY();

                    const double yB = global_y - 0.25*Ly;
                    const double yT = global_y - 0.75*Ly;
                    const double yBd = yB/delta;
                    const double yTd = yT/delta;

                    //* Initialise rho on nodes
                    for (int is = 0; is < ns; is++) 
                    {
                        if (DriftSpecies[is]) 
                        {
                            const double sech_yBd = 1. / cosh(yBd);
                            const double sech_yTd = 1. / cosh(yTd);
                            rhons[is][i][j][k] =  qom[is] / fabs(qom[is]) * rhoINIT[is] * sech_yBd * sech_yBd / FourPI;
                            rhons[is][i][j][k] += qom[is] / fabs(qom[is]) * rhoINIT[is] * sech_yTd * sech_yTd / FourPI;
                        }
                        else
                            rhons[is][i][j][k] = qom[is] / fabs(qom[is]) * rhoINIT[is] / FourPI;
                    }

                    //* Initialise E on nodes
                    Ex[i][j][k] =  0.0;
                    Ey[i][j][k] =  0.0;
                    Ez[i][j][k] =  0.0;
                    
                    //* Initialise B on nodes
                    Bxn[i][j][k] = B0x * (-1.0 + tanh(yBd) + tanh(-yTd));
                    Byn[i][j][k] = B0y;
                    Bzn[i][j][k] = B0z;                             //* Guide field
                    
                    //* Add first initial GEM perturbation
                    double xpert = global_x - Lx/4;
                    double ypert = global_y - Ly/4;

                    if (xpert < Lx/2 and ypert < Ly/2) 
                    {
                        Bxn[i][j][k] += (B0x * perturbation) * (M_PI/(0.5*Ly))   * cos(2*M_PI*xpert/(0.5*Lx)) * sin(M_PI*ypert/(0.5*Ly));
                        Byn[i][j][k] -= (B0x * perturbation) * (2*M_PI/(0.5*Lx)) * sin(2*M_PI*xpert/(0.5*Lx)) * cos(M_PI*ypert/(0.5*Ly));
                    }

                    //* Add second initial GEM perturbation
                    xpert = global_x - 3*Lx/4;
                    ypert = global_y - 3*Ly/4;

                    if (xpert > Lx/2 and ypert > Ly/2) 
                    {
                        Bxn[i][j][k] += (B0x * perturbation) * (M_PI/(0.5*Ly))   * cos(2*M_PI*xpert/(0.5*Lx)) * sin(M_PI*ypert/(0.5*Ly));
                        Byn[i][j][k] -= (B0x * perturbation) * (2*M_PI/(0.5*Lx)) * sin(2*M_PI*xpert/(0.5*Lx)) * cos(M_PI*ypert/(0.5*Ly));
                    }

                    //* Add first initial X perturbation
                    xpert = global_x - Lx/4;
                    ypert = global_y - Ly/4;
                    double exp_pert = exp(-(xpert / delta) * (xpert / delta) - (ypert / delta) * (ypert / delta));

                    Bxn[i][j][k] += (B0x * perturbation) * exp_pert * (-cos(M_PI * xpert / 10.0 / delta) * cos(M_PI * ypert / 10.0 / delta) * 2.0 * ypert / delta - cos(M_PI * xpert / 10.0 / delta) * sin(M_PI * ypert / 10.0 / delta) * M_PI / 10.0);
                    Byn[i][j][k] += (B0x * perturbation) * exp_pert * ( cos(M_PI * xpert / 10.0 / delta) * cos(M_PI * ypert / 10.0 / delta) * 2.0 * xpert / delta + sin(M_PI * xpert / 10.0 / delta) * cos(M_PI * ypert / 10.0 / delta) * M_PI / 10.0);

                    //* Add second initial X perturbation
                    xpert = global_x - 3*Lx/4;
                    ypert = global_y - 3*Ly/4;
                    exp_pert = exp(-(xpert / delta) * (xpert / delta) - (ypert / delta) * (ypert / delta));

                    Bxn[i][j][k] += (-B0x * perturbation) * exp_pert * (-cos(M_PI * xpert / 10.0 / delta) * cos(M_PI * ypert / 10.0 / delta) * 2.0 * ypert / delta - cos(M_PI * xpert / 10.0 / delta) * sin(M_PI * ypert / 10.0 / delta) * M_PI / 10.0);
                    Byn[i][j][k] += (-B0x * perturbation) * exp_pert * ( cos(M_PI * xpert / 10.0 / delta) * cos(M_PI * ypert / 10.0 / delta) * 2.0 * xpert / delta + sin(M_PI * xpert / 10.0 / delta) * cos(M_PI * ypert / 10.0 / delta) * M_PI / 10.0);
                }

        //* Communicate ghost data on nodes
        communicateNodeBC_old(nxn, nyn, nzn, Bxn, col->bcBx[0],col->bcBx[1],col->bcBx[2],col->bcBx[3],col->bcBx[4],col->bcBx[5], vct, this);
        communicateNodeBC_old(nxn, nyn, nzn, Byn, col->bcBy[0],col->bcBy[1],col->bcBy[2],col->bcBy[3],col->bcBy[4],col->bcBy[5], vct, this);
        communicateNodeBC_old(nxn, nyn, nzn, Bzn, col->bcBz[0],col->bcBz[1],col->bcBz[2],col->bcBz[3],col->bcBz[4],col->bcBz[5], vct, this);
        
        //* Initialise B on cell centres
        grid->interpN2C(Bxc, Bxn);
        grid->interpN2C(Byc, Byn);
        grid->interpN2C(Bzc, Bzn);
        
        //* Communicate ghost data on cell centres
        communicateCenterBC(nxc, nyc, nzc, Bxc, col->bcBx[0],col->bcBx[1],col->bcBx[2],col->bcBx[3],col->bcBx[4],col->bcBx[5], vct, this);
        communicateCenterBC(nxc, nyc, nzc, Byc, col->bcBy[0],col->bcBy[1],col->bcBy[2],col->bcBy[3],col->bcBy[4],col->bcBy[5], vct, this);
        communicateCenterBC(nxc, nyc, nzc, Bzc, col->bcBz[0],col->bcBz[1],col->bcBz[2],col->bcBz[3],col->bcBz[4],col->bcBz[5], vct, this);
    
        //* Initialise rho on cell centres
        for (int is = 0; is < ns; is++)
            grid->interpN2C(rhocs, is, rhons);
    } 
    else
        init();  //! READ FROM RESTART
}

void EMfields3D::init_double_Harris_hump()
{
    const Collective *col = &get_col();
    const VirtualTopology3D *vct = &get_vct();
    const Grid *grid = &get_grid();

    //* Custom input parameters
    const double perturbation               = input_param[0];       //* Amplitude of initial perturbation (localised in X)
    const double delta                      = input_param[1];       //* Half-thickness of current sheet

    const double delta_x = 8.0 * delta;
    const double delta_y = 4.0 * delta;
    
    if (restart_status == 0) 
    {
        if (vct->getCartesian_rank() ==0) 
        {
            cout << "-------------------------------------------" << endl;
            cout << " Initialising double Harris sheet with hump" << endl;
            cout << "-------------------------------------------" << endl;
            cout << "Initial magnetic field components (Bx, By, Bz) = " << "(" << B0x << ", " << B0y << ", " << B0z << ")" << endl;
            cout << "Initial perturbation                           = " << perturbation << endl;
            cout << "Half-thickness of current sheet                = " << delta << endl;
            cout << "-------------------------------------------" << endl;
        }

        //* Initialise E, B, and rho on nodes
        for (int i = 0; i < nxn; i++)
            for (int j = 0; j < nyn; j++)
                for (int k = 0; k < nzn; k++) 
                {
                    const double xM = grid->getXN(i, j, k) - 0.5  * Lx;
                    const double yB = grid->getYN(i, j, k) - 0.25 * Ly;
                    const double yT = grid->getYN(i, j, k) - 0.75 * Ly;
                    const double yBd = yB / delta;
                    const double yTd = yT / delta;

                    //* Initialise rho on nodes
                    for (int is = 0; is < ns; is++) 
                    {
                        if (DriftSpecies[is]) 
                        {
                            const double sech_yBd = 1. / cosh(yBd);
                            const double sech_yTd = 1. / cosh(yTd);
                            rhons[is][i][j][k] =  qom[is] / fabs(qom[is]) * rhoINIT[is] * sech_yBd * sech_yBd / FourPI;
                            rhons[is][i][j][k] += qom[is] / fabs(qom[is]) * rhoINIT[is] * sech_yTd * sech_yTd / FourPI;
                        }
                        else
                            rhons[is][i][j][k] = qom[is] / fabs(qom[is]) * rhoINIT[is] / FourPI;
                    }

                    //* Initialise E on nodes
                    Ex[i][j][k] = 0.0;
                    Ey[i][j][k] = 0.0;
                    Ez[i][j][k] = 0.0;

                    //* Initialise B on nodes
                    Bxn[i][j][k] = B0x * (-1.0 + tanh(yBd) - tanh(yTd));
                    Bxn[i][j][k] += 0.0;                                            // add the initial GEM perturbation

                    const double xMdx = xM / delta_x;
                    const double yBdy = yB / delta_y;
                    const double yTdy = yT / delta_y;
                    const double humpB = exp(-xMdx * xMdx - yBdy * yBdy);
                    
                    Byn[i][j][k] = B0y;
                    Bxn[i][j][k] -= (B0x * perturbation) * humpB * (2.0 * yBdy);    // add the initial X perturbation
                    Byn[i][j][k] += (B0x * perturbation) * humpB * (2.0 * xMdx);    // add the initial X perturbation
                    
                    const double humpT = exp(-xMdx * xMdx - yTdy * yTdy);
                    
                    Bxn[i][j][k] += (B0x * perturbation) * humpT * (2.0 * yTdy);    // add the second initial X perturbation
                    Byn[i][j][k] -= (B0x * perturbation) * humpT * (2.0 * xMdx);    // add the second initial X perturbation

                    //* Guide field
                    Bzn[i][j][k] = B0z;
                }

        //* Communicate ghost data on nodes
        communicateNodeBC(nxn, nyn, nzn, Bxn, col->bcBx[0],col->bcBx[1],col->bcBx[2],col->bcBx[3],col->bcBx[4],col->bcBx[5], vct, this);
        communicateNodeBC(nxn, nyn, nzn, Byn, col->bcBy[0],col->bcBy[1],col->bcBy[2],col->bcBy[3],col->bcBy[4],col->bcBy[5], vct, this);
        communicateNodeBC(nxn, nyn, nzn, Bzn, col->bcBz[0],col->bcBz[1],col->bcBz[2],col->bcBz[3],col->bcBz[4],col->bcBz[5], vct, this);

        //* Initialise B on cell centres
        for (int i = 0; i < nxc; i++)
            for (int j = 0; j < nyc; j++)
                for (int k = 0; k < nzc; k++) 
                {
                    const double xM = grid->getXN(i, j, k) - 0.5  * Lx;
                    const double yB = grid->getYN(i, j, k) - 0.25 * Ly;
                    const double yT = grid->getYN(i, j, k) - 0.75 * Ly;
                    const double yBd = yB / delta;
                    const double yTd = yT / delta;
                    
                    Bxc[i][j][k] = B0x * (-1.0 + tanh(yBd) - tanh(yTd));
                    Bxc[i][j][k] += 0.0;                                            // add the initial GEM perturbation
                    
                    const double xMdx = xM / delta_x;
                    const double yBdy = yB / delta_y;
                    const double yTdy = yT / delta_y;
                    const double humpB = exp(-xMdx * xMdx - yBdy * yBdy);

                    Byc[i][j][k] = B0y;
                    Bxc[i][j][k] -= (B0x * perturbation) * humpB * (2.0 * yBdy);    // add the initial X perturbation
                    Byc[i][j][k] += (B0x * perturbation) * humpB * (2.0 * xMdx);    // add the initial X perturbation
                    
                    const double humpT = exp(-xMdx * xMdx - yTdy * yTdy);

                    Bxc[i][j][k] += (B0x * perturbation) * humpT * (2.0 * yTdy);    // add the second initial X perturbation
                    Byc[i][j][k] -= (B0x * perturbation) * humpT * (2.0 * xMdx);    // add the second initial X perturbation
                    
                    //* Guide field
                    Bzc[i][j][k] = B0z;
                }

        //* Communicate ghost data on cell centres
        communicateCenterBC(nxc, nyc, nzc, Bxc, col->bcBx[0],col->bcBx[1],col->bcBx[2],col->bcBx[3],col->bcBx[4],col->bcBx[5], vct, this);
        communicateCenterBC(nxc, nyc, nzc, Byc, col->bcBy[0],col->bcBy[1],col->bcBy[2],col->bcBy[3],col->bcBy[4],col->bcBy[5], vct, this);
        communicateCenterBC(nxc, nyc, nzc, Bzc, col->bcBz[0],col->bcBz[1],col->bcBz[2],col->bcBz[3],col->bcBz[4],col->bcBz[5], vct, this);
        
        //* Initialise rho on cell centres
        for (int is = 0; is < ns; is++)
            grid->interpN2C(rhocs, is, rhons);
    }
    else
        init();  //! READ FROM RESTART
}

//* initialize GEM challenge with no Perturbation with dipole-like tail topology
void EMfields3D::initGEMDipoleLikeTailNoPert()
{
  const Grid *grid = &get_grid();
  // parameters controling the field topology
  // e.g., x1=Lx/5,x2=Lx/4 give 'separated' fields, x1=Lx/4,x2=Lx/3 give 'reconnected' topology

  double x1 = Lx / 6.0;         // minimal position of the gaussian peak 
  double x2 = Lx / 4.0;         // maximal position of the gaussian peak (the one closer to the center)
  double sigma = Lx / 15;       // base sigma of the gaussian - later it changes with the grid
  double stretch_curve = 2.0;   // stretch the sin^2 function over the x dimension - also can regulate the number of 'knots/reconnecitons points' if less than 1
  double skew_parameter = 0.50; // skew of the shape of the gaussian
  double pi = 3.1415927;
  double r1, r2, delta_x1x2;

  if (restart_status == 0) {

    // initialize
    if (get_vct().getCartesian_rank() == 0) {
      cout << "----------------------------------------------" << endl;
      cout << "Initialize GEM Challenge without Perturbation" << endl;
      cout << "----------------------------------------------" << endl;
      cout << "B0x                              = " << B0x << endl;
      cout << "B0y                              = " << B0y << endl;
      cout << "B0z                              = " << B0z << endl;
      cout << "Delta (current sheet thickness) = " << delta << endl;
      for (int i = 0; i < ns; i++) {
        cout << "rho species " << i << " = " << rhoINIT[i];
        if (DriftSpecies[i])
          cout << " DRIFTING " << endl;
        else
          cout << " BACKGROUND " << endl;
      }
      cout << "-------------------------" << endl;
    }

    for (int i = 0; i < nxn; i++)
      for (int j = 0; j < nyn; j++)
        for (int k = 0; k < nzn; k++) {
          // initialize the density for species
          for (int is = 0; is < ns; is++) {
            if (DriftSpecies[is])
              rhons[is][i][j][k] = ((rhoINIT[is] / (cosh((grid->getYN(i, j, k) - Ly / 2) / delta) * cosh((grid->getYN(i, j, k) - Ly / 2) / delta)))) / FourPI;
            else
              rhons[is][i][j][k] = rhoINIT[is] / FourPI;
          }
          // electric field
          Ex[i][j][k] = 0.0;
          Ey[i][j][k] = 0.0;
          Ez[i][j][k] = 0.0;
          // Magnetic field

          delta_x1x2 = x1 - x2 * (sin(((grid->getXN(i, j, k) - Lx / 2) / Lx * 180.0 / stretch_curve) * (0.25 * FourPI) / 180.0)) * (sin(((grid->getXN(i, j, k) - Lx / 2) / Lx * 180.0 / stretch_curve) * (0.25 * FourPI) / 180.0));

          r1 = (grid->getYN(i, j, k) - (x1 + delta_x1x2)) * (1.0 - skew_parameter * (sin(((grid->getXN(i, j, k) - Lx / 2) / Lx * 180.0) * (0.25 * FourPI) / 180.0)) * (sin(((grid->getXN(i, j, k) - Lx / 2) / Lx * 180.0) * (0.25 * FourPI) / 180.0)));
          r2 = (grid->getYN(i, j, k) - ((Lx - x1) - delta_x1x2)) * (1.0 - skew_parameter * (sin(((grid->getXN(i, j, k) - Lx / 2) / Lx * 180.0) * (0.25 * FourPI) / 180.0)) * (sin(((grid->getXN(i, j, k) - Lx / 2) / Lx * 180.0) * (0.25 * FourPI) / 180.0)));

          // tail-like field topology
          Bxn[i][j][k] = B0x * 0.5 * (-exp(-((r1) * (r1)) / (sigma * sigma)) + exp(-((r2) * (r2)) / (sigma * sigma)));

          Byn[i][j][k] = B0y;
          // guide field
          Bzn[i][j][k] = B0z;
        }
    // initialize B on centers
    for (int i = 0; i < nxc; i++)
      for (int j = 0; j < nyc; j++)
        for (int k = 0; k < nzc; k++) {
          // Magnetic field

          delta_x1x2 = x1 - x2 * (sin(((grid->getXC(i, j, k) - Lx / 2) / Lx * 180.0 / stretch_curve) * (0.25 * FourPI) / 180.0)) * (sin(((grid->getXC(i, j, k) - Lx / 2) / Lx * 180.0 / stretch_curve) * (0.25 * FourPI) / 180.0));

          r1 = (grid->getYC(i, j, k) - (x1 + delta_x1x2)) * (1.0 - skew_parameter * (sin(((grid->getXC(i, j, k) - Lx / 2) / Lx * 180.0) * (0.25 * FourPI) / 180.0)) * (sin(((grid->getXC(i, j, k) - Lx / 2) / Lx * 180.0) * (0.25 * FourPI) / 180.0)));
          r2 = (grid->getYC(i, j, k) - ((Lx - x1) - delta_x1x2)) * (1.0 - skew_parameter * (sin(((grid->getXC(i, j, k) - Lx / 2) / Lx * 180.0) * (0.25 * FourPI) / 180.0)) * (sin(((grid->getXC(i, j, k) - Lx / 2) / Lx * 180.0) * (0.25 * FourPI) / 180.0)));

          // tail-like field topology
          Bxn[i][j][k] = B0x * 0.5 * (-exp(-((r1) * (r1)) / (sigma * sigma)) + exp(-((r2) * (r2)) / (sigma * sigma)));

          Byc[i][j][k] = B0y;
          // guide field
          Bzc[i][j][k] = B0z;

        }
    for (int is = 0; is < ns; is++)
      grid->interpN2C(rhocs, is, rhons);
  }
  else {
    init(); // use the fields from restart file
  }

}

//* initialize GEM challenge with no Perturbation
void EMfields3D::initGEMnoPert()
{
  const Collective *col = &get_col();
  const VirtualTopology3D *vct = &get_vct();
  const Grid *grid = &get_grid();
  if (restart_status == 0) {

    // initialize
    if (get_vct().getCartesian_rank() == 0) {
      cout << "----------------------------------------------" << endl;
      cout << "Initialize GEM Challenge without Perturbation" << endl;
      cout << "----------------------------------------------" << endl;
      cout << "B0x                              = " << B0x << endl;
      cout << "B0y                              = " << B0y << endl;
      cout << "B0z                              = " << B0z << endl;
      cout << "Delta (current sheet thickness) = " << delta << endl;
      for (int i = 0; i < ns; i++) {
        cout << "rho species " << i << " = " << rhoINIT[i];
        if (DriftSpecies[i])
          cout << " DRIFTING " << endl;
        else
          cout << " BACKGROUND " << endl;
      }
      cout << "-------------------------" << endl;
    }
    for (int i = 0; i < nxn; i++)
      for (int j = 0; j < nyn; j++)
        for (int k = 0; k < nzn; k++) {
          // initialize the density for species
          for (int is = 0; is < ns; is++) {
            if (DriftSpecies[is])
              rhons[is][i][j][k] = ((rhoINIT[is] / (cosh((grid->getYN(i, j, k) - Ly / 2) / delta) * cosh((grid->getYN(i, j, k) - Ly / 2) / delta)))) / FourPI;
            else
              rhons[is][i][j][k] = rhoINIT[is] / FourPI;
          }
          // electric field
          Ex[i][j][k] = 0.0;
          Ey[i][j][k] = 0.0;
          Ez[i][j][k] = 0.0;
          // Magnetic field
          Bxn[i][j][k] = B0x * tanh((grid->getYN(i, j, k) - Ly / 2) / delta);
          Byn[i][j][k] = B0y;
          // guide field
          Bzn[i][j][k] = B0z;
        }
    // initialize B on centers
    for (int i = 0; i < nxc; i++)
      for (int j = 0; j < nyc; j++)
        for (int k = 0; k < nzc; k++) {
          // Magnetic field
          Bxc[i][j][k] = B0x * tanh((grid->getYC(i, j, k) - Ly / 2) / delta);
          Byc[i][j][k] = B0y;
          // guide field
          Bzc[i][j][k] = B0z;

        }
    for (int is = 0; is < ns; is++)
      grid->interpN2C(rhocs, is, rhons);
  }
  else {
    init(); // use the fields from restart file
  }
}

//* Random field generation
void EMfields3D::initRandomField()
{
  const VirtualTopology3D *vct = &get_vct();
  const Grid *grid = &get_grid();
  double **modes_seed = newArr2(double, 7, 7);
  if (restart_status ==0){
    // initialize
    if (get_vct().getCartesian_rank() ==0){
      cout << "------------------------------------------" << endl;
      cout << "Initialize GEM Challenge with Pertubation" << endl;
      cout << "------------------------------------------" << endl;
      cout << "B0x                              = " << B0x << endl;
      cout << "B0y                              = " << B0y << endl;
      cout << "B0z                              = " << B0z << endl;
      cout << "Delta (current sheet thickness) = " << delta << endl;
      for (int i=0; i < ns; i++){
	cout << "rho species " << i <<" = " << rhoINIT[i];
	if (DriftSpecies[i])
	  cout << " DRIFTING " << endl;
	else
	  cout << " BACKGROUND " << endl;
      }
      cout << "-------------------------" << endl;
    }
    double kx;
    double ky;
        
    /*       stringstream num_proc;
	     num_proc << vct->getCartesian_rank() ;
	     string cqsat = SaveDirName + "/RandomNumbers" + num_proc.str() + ".txt";
        ofstream my_file(cqsat.c_str(), fstream::binary);
	for (int m=-3; m < 4; m++)
            for (int n=-3; n < 4; n++){
            modes_seed[m+3][n+3] = rand() / (double) RAND_MAX;
            my_file <<"modes_seed["<< m+3<<"][" << "\t" << n+3 << "] = " << modes_seed[m+3][n+3] << endl;
            }
              my_file.close();
    */
    modes_seed[0][0] = 0.532767;
    modes_seed[0][1] = 0.218959;
    modes_seed[0][2] = 0.0470446;
    modes_seed[0][3] = 0.678865;
    modes_seed[0][4] = 0.679296;
    modes_seed[0][5] = 0.934693;
    modes_seed[0][6] = 0.383502;
    modes_seed[1][0] = 0.519416;
    modes_seed[1][1] = 0.830965;
    modes_seed[1][2] = 0.0345721;
    modes_seed[1][3] = 0.0534616;
    modes_seed[1][4] = 0.5297;
    modes_seed[1][5] = 0.671149;
    modes_seed[1][6] = 0.00769819;
    modes_seed[2][0] = 0.383416;
    modes_seed[2][1] = 0.0668422;
    modes_seed[2][2] = 0.417486;
    modes_seed[2][3] = 0.686773;
    modes_seed[2][4] = 0.588977;
    modes_seed[2][5] = 0.930436;
    modes_seed[2][6] = 0.846167;
    modes_seed[3][0] = 0.526929;
    modes_seed[3][1] = 0.0919649;
    modes_seed[3][2] = 0.653919;
    modes_seed[3][3] = 0.415999;
    modes_seed[3][4] = 0.701191;
    modes_seed[3][5] = 0.910321;
    modes_seed[3][6] = 0.762198;
    modes_seed[4][0] = 0.262453;
    modes_seed[4][1] = 0.0474645;
    modes_seed[4][2] = 0.736082;
    modes_seed[4][3] = 0.328234;
    modes_seed[4][4] = 0.632639;
    modes_seed[4][5] = 0.75641;
    modes_seed[4][6] = 0.991037;
    modes_seed[5][0] = 0.365339;
    modes_seed[5][1] = 0.247039;
    modes_seed[5][2] = 0.98255;
    modes_seed[5][3] = 0.72266;
    modes_seed[5][4] = 0.753356;
    modes_seed[5][5] = 0.651519;
    modes_seed[5][6] = 0.0726859;
    modes_seed[6][0] = 0.631635;
    modes_seed[6][1] = 0.884707;
    modes_seed[6][2] = 0.27271;
    modes_seed[6][3] = 0.436411;
    modes_seed[6][4] = 0.766495;
    modes_seed[6][5] = 0.477732;
    modes_seed[6][6] = 0.237774;

    for (int i=0; i < nxn; i++)
      for (int j=0; j < nyn; j++)
	for (int k=0; k < nzn; k++){
	  // initialize the density for species
	  for (int is=0; is < ns; is++){
	    rhons[is][i][j][k] = rhoINIT[is]/FourPI;
	  }
	  // electric field
	  Ex[i][j][k] =  0.0;
	  Ey[i][j][k] =  0.0;
	  Ez[i][j][k] =  0.0;
	  // Magnetic field
	  Bxn[i][j][k] =  0.0;
	  Byn[i][j][k] =  0.0;
	  Bzn[i][j][k] =  B0z;
	  for (int m=-3; m < 4; m++)
	    for (int n=-3; n < 4; n++){

	      kx=2.0*M_PI*m/Lx;
	      ky=2.0*M_PI*n/Ly;
	      Bxn[i][j][k] += -B0x*ky*cos(grid->getXN(i,j,k)*kx+grid->getYN(i,j,k)*ky+2.0*M_PI*modes_seed[m+3][n+3]);
	      Byn[i][j][k] += B0x*kx*cos(grid->getXN(i,j,k)*kx+grid->getYN(i,j,k)*ky+2.0*M_PI*modes_seed[m+3][n+3]);
	      // Bzn[i][j][k] += B0x*cos(grid->getXN(i,j,k)*kx+grid->getYN(i,j,k)*ky+2.0*M_PI*modes_seed[m+3][n+3]);
	    }
	}
	  // communicate ghost
	  communicateNodeBC(nxn, nyn, nzn, Bxn, 1, 1, 2, 2, 1, 1, vct, this);
	  communicateNodeBC(nxn, nyn, nzn, Byn, 1, 1, 1, 1, 1, 1, vct, this);
	  communicateNodeBC(nxn, nyn, nzn, Bzn, 1, 1, 2, 2, 1, 1, vct, this);

	  // initialize B on centers
	  grid->interpN2C(Bxc, Bxn);
	  grid->interpN2C(Byc, Byn);
	  grid->interpN2C(Bzc, Bzn);
	  // communicate ghost
	  communicateCenterBC(nxc, nyc, nzc, Bxc, 2, 2, 2, 2, 2, 2, vct,this);
	  communicateCenterBC(nxc, nyc, nzc, Byc, 1, 1, 1, 1, 1, 1, vct,this);
	  communicateCenterBC(nxc, nyc, nzc, Bzc, 2, 2, 2, 2, 2, 2, vct,this);
	  for (int is=0 ; is<ns; is++)
            grid->interpN2C(rhocs,is,rhons);
	} else {
    init(); // use the fields from restart file
    }
  delArr2(modes_seed, 7);
}

//* Initialise force free fields (JxB=0)
void EMfields3D::initForceFree()
{
  const VirtualTopology3D *vct = &get_vct();
  const Grid *grid = &get_grid();
  if (restart_status == 0) {

    // initialize
    if (get_vct().getCartesian_rank() == 0) {
      cout << "----------------------------------------" << endl;
      cout << "Initialize Force Free with Perturbation" << endl;
      cout << "----------------------------------------" << endl;
      cout << "B0x                              = " << B0x << endl;
      cout << "B0y                              = " << B0y << endl;
      cout << "B0z                              = " << B0z << endl;
      cout << "Delta (current sheet thickness) = " << delta << endl;
      for (int i = 0; i < ns; i++) {
        cout << "rho species " << i << " = " << rhoINIT[i];
      }
      cout << "Smoothing Factor = " << Smooth << endl;
      cout << "-------------------------" << endl;
    }
    for (int i = 0; i < nxn; i++)
      for (int j = 0; j < nyn; j++)
        for (int k = 0; k < nzn; k++) {
          // initialize the density for species
          for (int is = 0; is < ns; is++) {
            rhons[is][i][j][k] = rhoINIT[is] / FourPI;
          }
          // electric field
          Ex[i][j][k] = 0.0;
          Ey[i][j][k] = 0.0;
          Ez[i][j][k] = 0.0;
          // Magnetic field
          Bxn[i][j][k] = B0x * tanh((grid->getYN(i, j, k) - Ly / 2) / delta);
          // add the initial GEM perturbation
          Bxn[i][j][k] += (B0x / 10.0) * (M_PI / Ly) * cos(2 * M_PI * grid->getXN(i, j, k) / Lx) * sin(M_PI * (grid->getYN(i, j, k) - Ly / 2) / Ly);
          Byn[i][j][k] = B0y - (B0x / 10.0) * (2 * M_PI / Lx) * sin(2 * M_PI * grid->getXN(i, j, k) / Lx) * cos(M_PI * (grid->getYN(i, j, k) - Ly / 2) / Ly);
          // guide field
          Bzn[i][j][k] = B0z / cosh((grid->getYN(i, j, k) - Ly / 2) / delta);
        }
    for (int i = 0; i < nxc; i++)
      for (int j = 0; j < nyc; j++)
        for (int k = 0; k < nzc; k++) {
          Bxc[i][j][k] = B0x * tanh((grid->getYC(i, j, k) - Ly / 2) / delta);
          // add the perturbation
          Bxc[i][j][k] += (B0x / 10.0) * (M_PI / Ly) * cos(2 * M_PI * grid->getXC(i, j, k) / Lx) * sin(M_PI * (grid->getYC(i, j, k) - Ly / 2) / Ly);
          Byc[i][j][k] = B0y - (B0x / 10.0) * (2 * M_PI / Lx) * sin(2 * M_PI * grid->getXC(i, j, k) / Lx) * cos(M_PI * (grid->getYC(i, j, k) - Ly / 2) / Ly);
          // guide field
          Bzc[i][j][k] = B0z / cosh((grid->getYC(i, j, k) - Ly / 2) / delta);
        }

    for (int is = 0; is < ns; is++)
      grid->interpN2C(rhocs, is, rhons);
  }
  else {
    init(); // use the fields from restart file
  }
}

//* Initialize the EM field with constants values or from restart
void EMfields3D::initBEAM(double x_center, double y_center, double z_center, double radius)
{
  const Grid *grid = &get_grid();

  double distance;
  // initialize E and rhos on nodes
  if (restart_status == 0) {
    for (int i = 0; i < nxn; i++)
      for (int j = 0; j < nyn; j++)
        for (int k = 0; k < nzn; k++) {
          Ex[i][j][k] = 0.0;
          Ey[i][j][k] = 0.0;
          Ez[i][j][k] = 0.0;
          Bxn[i][j][k] = 0.0;
          Byn[i][j][k] = 0.0;
          Bzn[i][j][k] = 0.0;
          distance = (grid->getXN(i, j, k) - x_center) * (grid->getXN(i, j, k) - x_center) / (radius * radius) + (grid->getYN(i, j, k) - y_center) * (grid->getYN(i, j, k) - y_center) / (radius * radius) + (grid->getZN(i, j, k) - z_center) * (grid->getZN(i, j, k) - z_center) / (4 * radius * radius);
          // plasma
          rhons[0][i][j][k] = rhoINIT[0] / FourPI;  // initialize with constant density
          // electrons
          rhons[1][i][j][k] = rhoINIT[1] / FourPI;
          // beam
          if (distance < 1.0)
            rhons[2][i][j][k] = rhoINIT[2] / FourPI;
          else
            rhons[2][i][j][k] = 0.0;
        }
    // initialize B on centers
    for (int i = 0; i < nxc; i++)
      for (int j = 0; j < nyc; j++)
        for (int k = 0; k < nzc; k++) {
          // Magnetic field
          Bxc[i][j][k] = 0.0;
          Byc[i][j][k] = 0.0;
          Bzc[i][j][k] = 0.0;


        }
    for (int is = 0; is < ns; is++)
      grid->interpN2C(rhocs, is, rhons);
  }
  else {
    init(); // use the fields from restart file
  }
}

//* Initialise a combination of magnetic dipoles
void EMfields3D::initDipole()
{
  const Collective *col = &get_col();
  const VirtualTopology3D *vct = &get_vct();
  const Grid *grid = &get_grid();

  // initialize
  if (vct->getCartesian_rank() ==0){
      cout << "------------------------------------------" << endl;
      cout << "Initialise a Magnetic Dipole " << endl;
      cout << "------------------------------------------" << endl;
      cout << "B0x                              = " << B0x << endl;
      cout << "B0y                              = " << B0y << endl;
      cout << "B0z                              = " << B0z << endl;
      cout << "B1x   (external dipole field) - X  = " << B1x << endl;
      cout << "B1y                              = " << B1y << endl;
      cout << "B1z                              = " << B1z << endl;
      cout << "L_square - no magnetic field inside a sphere with radius L_square  = " << L_square << endl;
      cout << "Center dipole - X                = " << x_center << endl;
      cout << "Center dipole - Y                = " << y_center << endl;
      cout << "Center dipole - Z                = " << z_center << endl;
      cout << "Solar Wind drift velocity        = " << ue0 << endl;
  }


  double distance;
  double x_displ, y_displ, z_displ, fac1;

  double ebc[3];
  cross_product(ue0,ve0,we0,B0x,B0y,B0z,ebc);
  scale(ebc,-1.0,3);

  for (int i=0; i < nxn; i++){
    for (int j=0; j < nyn; j++){
      for (int k=0; k < nzn; k++){
        for (int is=0; is < ns; is++){
          rhons[is][i][j][k] = rhoINIT[is]/FourPI;
        }
        Ex[i][j][k] = ebc[0];
        Ey[i][j][k] = ebc[1];
        Ez[i][j][k] = ebc[2];

        double blp[3];
        // radius of the planet
        double a=L_square;

        double xc=x_center;
        double yc=y_center;
        double zc=z_center;

        double x = grid->getXN(i,j,k);
        double y = grid->getYN(i,j,k);
        double z = grid->getZN(i,j,k);

        double r2 = ((x-xc)*(x-xc)) + ((y-yc)*(y-yc)) + ((z-zc)*(z-zc));

        // Compute dipolar field B_ext

        if (r2 > a*a) {
            x_displ = x - xc;
            y_displ = y - yc;
            z_displ = z - zc;
            fac1 =  -B1z*a*a*a/pow(r2,2.5);
	    Bx_ext[i][j][k] = 3*x_displ*z_displ*fac1;
	    By_ext[i][j][k] = 3*y_displ*z_displ*fac1;
	    Bz_ext[i][j][k] = (2*z_displ*z_displ -x_displ*x_displ -y_displ*y_displ)*fac1;
        }
        else { // no field inside the planet
            Bx_ext[i][j][k]  = 0.0;
            By_ext[i][j][k]  = 0.0;
            Bz_ext[i][j][k]  = 0.0;
        }
        Bxn[i][j][k] = B0x;// + Bx_ext[i][j][k]
        Byn[i][j][k] = B0y;// + By_ext[i][j][k]
        Bzn[i][j][k] = B0z;// + Bz_ext[i][j][k]

      }
    }
  }

	grid->interpN2C(Bxc,Bxn);
	grid->interpN2C(Byc,Byn);
	grid->interpN2C(Bzc,Bzn);dprintf("1 Bzc[1][15][0]=%f",Bzc[1][15][0]);

	communicateCenterBC_P(nxc,nyc,nzc,Bxc,col->bcBx[0],col->bcBx[1],col->bcBx[2],col->bcBx[3],col->bcBx[4],col->bcBx[5],vct, this);
	communicateCenterBC_P(nxc,nyc,nzc,Byc,col->bcBy[0],col->bcBy[1],col->bcBy[2],col->bcBy[3],col->bcBy[4],col->bcBy[5],vct, this);
	communicateCenterBC_P(nxc,nyc,nzc,Bzc,col->bcBz[0],col->bcBz[1],col->bcBz[2],col->bcBz[3],col->bcBz[4],col->bcBz[5],vct, this);

	for (int is=0 ; is<ns; is++)
		grid->interpN2C(rhocs,is,rhons);

	if (restart_status != 0) { // EM initialization from RESTART
		init();  // use the fields from restart file
	}

}

//* Initialise a 2D magnetic dipoles according to paper L.K.S Two-way coupling of a global Hall
void EMfields3D::initDipole2D()
{
  const Collective *col = &get_col();
  const VirtualTopology3D *vct = &get_vct();
  const Grid *grid = &get_grid();

  // initialize
  if (vct->getCartesian_rank() ==0){
      cout << "------------------------------------------" << endl;
      cout << "Initialise a 2D Magnetic Dipole on XY Plane" << endl;
      cout << "------------------------------------------" << endl;
      cout << "B0x                              = " << B0x << endl;
      cout << "B0y                              = " << B0y << endl;
      cout << "B0z                              = " << B0z << endl;
      cout << "B1x   (external dipole field)    = " << B1x << endl;
      cout << "B1y                              = " << B1y << endl;
      cout << "B1z                              = " << B1z << endl;
      cout << "L_square - no magnetic field inside a sphere with radius L_square  = " << L_square << endl;
      cout << "Center dipole - X                = " << x_center << endl;
      cout << "Center dipole - Y                = " << y_center << endl;
      cout << "Center dipole - Z                = " << z_center << endl;
      cout << "Solar Wind drift velocity        = " << ue0 << endl;
      cout << "2D Smoothing Factor              = " << Smooth << endl;
      cout << "Smooth Iteration                 = " << smooth_cycle << endl;
  }


  double distance;
  double x_displ, z_displ, fac1;

  double ebc[3];
  cross_product(ue0,ve0,we0,B0x,B0y,B0z,ebc);
  scale(ebc,-1.0,3);

  for (int i=0; i < nxn; i++){
    for (int j=0; j < nyn; j++){
      for (int k=0; k < nzn; k++){
        for (int is=0; is < ns; is++){
          rhons[is][i][j][k] = rhoINIT[is]/FourPI;
        }
        Ex[i][j][k] = ebc[0];
        Ey[i][j][k] = ebc[1];
        Ez[i][j][k] = ebc[2];

        double blp[3];
        double a=L_square;

        double xc=x_center;
        double zc=z_center;

        double x = grid->getXN(i,j,k);
        double z = grid->getZN(i,j,k);

        double r2 = ((x-xc)*(x-xc)) + ((z-zc)*(z-zc));

        // Compute dipolar field B_ext

        if (r2 > a*a) {
            x_displ = x - xc;
            z_displ = z - zc;

            fac1 =  -B1z*a*a/(r2*r2);//fac1 = D/4?

			Bx_ext[i][j][k] = 2*x_displ*z_displ*fac1;
			By_ext[i][j][k] = 0.0;
			Bz_ext[i][j][k] = (z_displ*z_displ - x_displ*x_displ)*fac1;

        }
        else { // no field inside the planet
            Bx_ext[i][j][k]  = 0.0;
            By_ext[i][j][k]  = 0.0;
            Bz_ext[i][j][k]  = 0.0;
        }

        Bxn[i][j][k] = B0x;// + Bx_ext[i][j][k]
        Byn[i][j][k] = B0y;// + By_ext[i][j][k]
        Bzn[i][j][k] = B0z;// + Bz_ext[i][j][k]

      }
    }
  }


	grid->interpN2C(Bxc,Bxn);
	grid->interpN2C(Byc,Byn);
	grid->interpN2C(Bzc,Bzn);

	communicateCenterBC_P(nxc,nyc,nzc,Bxc,col->bcBx[0],col->bcBx[1],col->bcBx[2],col->bcBx[3],col->bcBx[4],col->bcBx[5],vct, this);
	communicateCenterBC_P(nxc,nyc,nzc,Byc,col->bcBy[0],col->bcBy[1],col->bcBy[2],col->bcBy[3],col->bcBy[4],col->bcBy[5],vct, this);
	communicateCenterBC_P(nxc,nyc,nzc,Bzc,col->bcBz[0],col->bcBz[1],col->bcBz[2],col->bcBz[3],col->bcBz[4],col->bcBz[5],vct, this);

	for (int is=0 ; is<ns; is++)
		grid->interpN2C(rhocs,is,rhons);



	if (restart_status != 0) { // EM initialization from RESTART
		init();  // use the fields from restart file
	}
}

//* Initialise fields for shear velocity in fluid finite Larmor radius (FLR) equilibrium (Cerri et al. 2013)
//* The charge is set to 1/(4 pi) in order to satisfy the omega_pi = 1. The 2 species have same charge density to guarantee plasma neutrality
void EMfields3D::init_KHI_FLR()
{
    const Collective *col = &get_col();
    const VirtualTopology3D *vct = &get_vct();
    const Grid *grid = &get_grid();

    //* Custom input parameters
    const double velocity_shear         = input_param[0];       //* Initial velocity shear
    const double perturbation           = input_param[1];       //* Amplitude of initial perturbation
    const double gamma_electrons        = input_param[2];       //* Gamma for isothermal electrons (FLR corrections)
    const double gamma_ions_perp        = input_param[3];       //* Gamma (perpendicular) for ions (FLR corrections)
    const double gamma_ions_parallel    = input_param[4];       //* Gamma (parallel) for ions (FLR corrections)
    const double s3                     = input_param[5];       //* +/-1 (Here -1 : Ux(y) or 1 : Uy(x)) (FLR corrections)
    const double delta                  = input_param[6];       //* Thickness of shear layer (FLR corrections)

    //* Gauss' law
    array3_double divE0c(nxc, nyc, nzc);
    array3_double divE0n(nxn, nyn, nzn);

    double Vthi = col->getUth(1);                               //* Ion thermal velocity (supposed isotropic far from velocity shear layer)
    double qomi = col->getQOM(1);                               //* Ion charge to mass ratio
    double Vthe = col->getUth(0);                               //* Electron thermal velocity (supposed isotropic far from velocity shear layer)
    double qome = col->getQOM(0);                               //* Electron charge to mass ratio
    double TeTi = -qomi/qome * (Vthe/Vthi) * (Vthe/Vthi);       //* Electron to ion temperature ratio (computed from input file parameters)
    
    //* For FLR corrections
    double B0              = sqrt(B0x*B0x+B0y*B0y+B0z*B0z);     //* Magnetic field amplitude
    double beta            = 2.0*(Vthi/B0)*(Vthi/B0);           //* Ion plasma beta from input file parameters; NOTE: beta = beta_i
    const double Omega_ci  = B0;                                //* Cf. normalisation qom = 1 for ions
    double gammabar        = gamma_electrons/gamma_ions_perp - 1.0;
    double betaiperp0      = beta;
    double betae0          = TeTi*betaiperp0;
    double betae0bar       = betae0 / (1.0 + betae0 + betaiperp0);
    double betaiperp0bar   = betaiperp0 / (1.0 + betae0 + betaiperp0);
    double C0              = 0.5*s3*betaiperp0bar*velocity_shear/(Omega_ci*delta);
    double Cinf            = C0/(1.0 + gammabar*betae0bar);
    double power           = 1.0/gamma_ions_parallel;

    if (vct->getCartesian_rank() == 0)
    {
        cout << "---------------------------------------------------------" << endl;
        cout << "    Initialising velocity shear (with FLR correction)    " << endl;
        cout << "---------------------------------------------------------" << endl;
        cout << " Initial magnetic field components (Bx, By, Bz) = " << "(" << B0x << ", " << B0y << ", " << B0z << ")" << endl;
        cout << " Thickness of velocity shear (delta)            = " << delta            << endl;
        cout << " Velocity shear                                 = " << velocity_shear   << endl;
        cout << " Electron thermal velocity                      = " << Vthe             << endl;
        cout << " Ion thermal velocity                           = " << Vthi             << endl;
        cout << " Temperature ratio Te/Ti                        = " << TeTi             << endl;
        cout << " Ion plasma beta                                = " << beta             << endl << endl;
        cout << " No initial mean velocity perturbation: test effect SVP " << endl;
        cout << "---------------------------------------------------------" << endl << endl;
    }

    if (restart_status == 0)
    {
        for (int i = 0; i < nxn; i++)
            for (int j = 0; j < nyn; j++)
                for (int k = 0; k < nzn; k++) 
                {
                    //* For ion FLR corrections
                    double ay  = 1.0/pow((cosh((grid->getYN(i,j,k)-0.25*Ly)/delta)), 2.0) - 1.0/pow((cosh((grid->getYN(i,j,k)-0.75*Ly)/delta)), 2.0);
                    double finf = 1.0/(1.0 - Cinf*ay);

                    //* Initialise B on nodes
                    Bxn[i][j][k] = B0x;
                    Byn[i][j][k] = B0y*sqrt(finf);      //* FLR profile on magnetic field (if mag. field angle)
                    Bzn[i][j][k] = B0z*sqrt(finf);      //* FLR profile on magnetic field
                    
                    double udrift = velocity_shear * (tanh((grid->getYN(i,j,k)-0.25*Ly)/delta) - tanh((grid->getYN(i,j,k)-0.75*Ly)/delta)-1.0);   //* X velocity drift to calculate electric field (identical for electrons and ions)

                    //* Initialise E on nodes (ideal Ohm's law)
                    Ex[i][j][k] =  0.0;
                    Ey[i][j][k] =  udrift*Bzn[i][j][k];
                    Ez[i][j][k] =  0.0;
                }

        //* Divergence of E for correcting rho
        grid->divN2C(divE0c, Ex, Ey, Ez);
        scale(divE0c, 1.0/FourPI, nxc, nyc, nzc);
        grid->interpN2C(divE0c, divE0n);

        for (int i = 0; i < nxn; i++)
            for (int j = 0; j < nyn; j++)
                for (int k = 0; k < nzn; k++) 
                {
                    //* For ion FLR corrections
                    double ay  = 1.0/pow((cosh((grid->getYC(i,j,k)-0.25*Ly)/delta)),2.0) - 1.0/pow((cosh((grid->getYC(i,j,k)-0.75*Ly)/delta)), 2.0);
                    double finf = 1.0/(1.0 - Cinf*ay);
                    
                    //* Initialise rho on nodes
                    for (int is = 0; is < ns; is++) 
                    {
                        rhons[is][i][j][k] = rhoINIT[is]/FourPI;
                        rhons[is][i][j][k] = rhons[is][i][j][k]*pow(finf, power);       //* FLR corrections for density

                        if (qom[is] < 0.0)
                        {
                            //! Electrons
                            rhons[is][i][j][k] = rhons[is][i][j][k] - divE0n[i][j][k];  //* Gauss' law 
                        }
                    }
                }
        
        for (int i = 0; i < nxc; i++)
            for (int j = 0; j < nyc; j++)
                for (int k = 0; k < nzc; k++) 
                {
                    //* For ion FLR corrections
                    double ay  = 1.0/pow((cosh((grid->getYC(i,j,k)-0.25*Ly)/delta)),2.0) - 1.0/pow((cosh((grid->getYC(i,j,k)-0.75*Ly)/delta)), 2.0);
                    double finf = 1.0/(1.0 - Cinf*ay);
          
                    //* Initialise B at cell centres
                    Bxc[i][j][k] = B0x;
                    Byc[i][j][k] = B0y*sqrt(finf);      //* FLR profile on magnetic field (if mag. field angle)
                    Bzc[i][j][k] = B0z*sqrt(finf);      //* FLR profile on magnetic field
                }

        //* Initialise rho on cell centres
        for (int is = 0; is < ns; is++)
            grid->interpN2C(rhocs, is, rhons);
    } 
    else
        init();  //! READ FROM RESTART

}


//! Initial field distributions (Relativistic) !//

//? Quasi-1D ion-electron shock (Relativistic and Non relativistic)
void EMfields3D::initShock1D() 
{
    const Collective *col = &get_col();
    const VirtualTopology3D *vct = &get_vct();
    const Grid *grid = &get_grid();

    double v0  = col->getU0(1);
    double thb = col->getUth(1);

    if (restart_status == 0)
    {
        //! Initial setup (NOT RESTART)
        if (vct->getCartesian_rank() == 0) 
        {
            cout << "-------------------------------------------------------" << endl;
            cout << "Initialise quasi-1D double periodic ion-electron shock " << endl;
            cout << "-------------------------------------------------------" << endl;
            cout << "Background ion sigma                = " << (B0x*B0x+B0y*B0y+B0z*B0z)/sqrt(FourPI*rhoINIT[1]) << endl;
            if (col->getRelativistic())
            cout << "Background theta_i                  = " << thb << endl;
            cout << "Background bulk velocity            = " << v0 << endl;
            cout << "-------------------------------------------------------" << endl;
        }
  
        //* Initialise B at cell centres
        for (int i = 1; i < nxc-1; i++) 
            for (int j = 1; j < nyc-1; j++)
                for (int k = 1; k < nzc-1; k++) 
                {
                    Bxc[i][j][k] = B0x;
                    Byc[i][j][k] = B0y;
                    Bzc[i][j][k] = B0z;
                }

        //* Communicate ghost data at cell centres
        communicateCenterBC(nxc, nyc, nzc, Bxc, col->bcBx[0],col->bcBx[1],col->bcBx[2],col->bcBx[3],col->bcBx[4],col->bcBx[5], vct, this);
        communicateCenterBC(nxc, nyc, nzc, Byc, col->bcBy[0],col->bcBy[1],col->bcBy[2],col->bcBy[3],col->bcBy[4],col->bcBy[5], vct, this);
        communicateCenterBC(nxc, nyc, nzc, Bzc, col->bcBz[0],col->bcBz[1],col->bcBz[2],col->bcBz[3],col->bcBz[4],col->bcBz[5], vct, this);
      
        //* Initialise B at cell centres
        grid->interpC2N(Bxn,Bxc);
        grid->interpC2N(Byn,Byc);
        grid->interpC2N(Bzn,Bzc);

        //* Communicate ghost data on nodes
        communicateNodeBC(nxn, nyn, nzn, Bxn, col->bcBx[0],col->bcBx[1],col->bcBx[2],col->bcBx[3],col->bcBx[4],col->bcBx[5], vct, this);
        communicateNodeBC(nxn, nyn, nzn, Byn, col->bcBy[0],col->bcBy[1],col->bcBy[2],col->bcBy[3],col->bcBy[4],col->bcBy[5], vct, this);
        communicateNodeBC(nxn, nyn, nzn, Bzn, col->bcBz[0],col->bcBz[1],col->bcBz[2],col->bcBz[3],col->bcBz[4],col->bcBz[5], vct, this);
  
        //* Initialise E on nodes
        for (int i = 1; i < nxn-1; i++) 
            for (int j = 1; j < nyn-1; j++)
                for (int k = 1; k < nzn-1; k++) 
                {
                    double xN = grid->getXN(i, j, k);
                    double fac = (xN>Lx/2.0 && xN < Lx-grid->getDX()) ? -1.0 : 1.0;
                    Ex[i][j][k] = 0.0;
                    Ey[i][j][k] = fac*v0*B0z;
                    Ez[i][j][k] = -fac*v0*B0y;
                }

        //* Communicate ghost data on nodes
        communicateNodeBC(nxn, nyn, nzn, Ex, col->bcEx[0],col->bcEx[1],col->bcEx[2],col->bcEx[3],col->bcBx[4],col->bcEx[5], vct, this);
        communicateNodeBC(nxn, nyn, nzn, Ey, col->bcEy[0],col->bcEy[1],col->bcEy[2],col->bcEy[3],col->bcBy[4],col->bcEy[5], vct, this);
        communicateNodeBC(nxn, nyn, nzn, Ez, col->bcEz[0],col->bcEz[1],col->bcEz[2],col->bcEz[3],col->bcBz[4],col->bcEz[5], vct, this);
    }
    else
    {
        //! READ FROM RESTART
        init();
    }
}

//? Relativistic double Harris for pair plasma: Maxwellian background, drifting particles in the sheets
void EMfields3D::init_Relativistic_Double_Harris_pairs() 
{
    const Collective *col = &get_col();
    const VirtualTopology3D *vct = &get_vct();
    const Grid *grid = &get_grid();

    //* Custom input parameters for relativistic reconnection
    const double sigma                  = input_param[0];       //* Magnetisation parameter
    const double eta                    = input_param[1];       //* Ratio of current sheet density to upstream density (this is "alpha" in Fabio's paper; Eqs 52 and 53)
    const double delta_CS               = input_param[2];       //* Half-thickness of current sheet (free parameter)
    const double perturbation           = input_param[3];       //* Amplitude of initial perturbation
    const double guide_field_ratio      = input_param[4];       //* Ratio of guide field to in-plane magnetic field

    //* Background (BG) or upstream particles
    double thermal_spread_BG    = col->getUth(0);                           //* Thermal spread
    double rho_BG               = rhoINIT[0]/(4.0*M_PI);                    //* Density (rho_BG = n * mc^2)
    double B_BG                 = sqrt(sigma*4.0*M_PI*rho_BG*2.0);          //* sigma = B^2/(4*pi*rho_electron*rho_prositron)

    //* Current sheet (CS) particles
    double rho_CS              = eta*rho_BG;                                            //* Density (rho_CS = eta * n * mc^2)
    double drift_velocity      = B_BG/(2.0*4.0*M_PI*rho_CS*delta_CS/c);                 //* v = B*c/(8 * pi * rho_CS * delta_CS); Eq 52
    double lorentz_factor_CS   = 1.0/sqrt(1.0 - drift_velocity*drift_velocity);         //* Lorentz factor of the relativistic drifting particles
    double thermal_spread_CS   = B_BG*B_BG*lorentz_factor_CS/(16.0*M_PI*rho_CS);        //* Thermal spread (B^2 * Gamma/(16 * pi * eta * n * mc^2)); Eq 53
    
    if (restart_status == 0) 
    {
        if (vct->getCartesian_rank() == 0) 
        {
            cout << "-----------------------------------------------------------"   << endl;
            cout << "Relativistic double Harris sheet for pair plasma"              << endl;
            cout << "-----------------------------------------------------------"   << endl << endl; 

            cout << "Ratio of CS density to upstream density            = " << eta                      << endl;
            cout << "Perturbation amplitude                             = " << perturbation             << endl; 
            cout << "Ratio of guide magnetic field to background field  = " << guide_field_ratio        << endl << endl; 
            
            cout << "BACKGROUND/UPSTREAM:"                                                              << endl;
            cout << "   Magnetisation parameter                 = " << sigma                            << endl; 
            cout << "   Plasma beta                             = " << 2.0*rho_BG*thermal_spread_BG/(B_BG*B_BG/2.0/FourPI)  << endl;
            cout << "   Thermal spread                          = " << thermal_spread_BG                << endl << endl;
            
            cout << "CURRENT SHEET:"                                                                    << endl;
            cout << "   Thermal spread of drifiting particles   = " << thermal_spread_CS                << endl; 
            cout << "   Lorentz factor of drifiting particles   = " << lorentz_factor_CS                << endl; 
                    
            cout << "-----------------------------------------------------------"   << endl;
        }
  
        //* Params for setting up current sheet
        double x14=Lx/4.0;
        double x34=3.0*Lx/4.0;
        double y12=Ly/2.0;
        double y14=Ly/4.0;
        double y34=3.0*Ly/4.0;
        double ym=Ly;   // 4 times the perturbation height
        double xm=Lx;   // perturbation wavelength

        double xN, yN, yh, xh, cosyh, cosxh, sinyh, sinxh;
        double fBx, fBy;

        for (int i = 1; i < nxc-1; i++)
            for (int j = 1; j < nyc-1; j++)
                for (int k = 1; k < nzc-1; k++) 
                {
                    double xN = grid->getXC(i, j, k);
                    double yN = grid->getYC(i, j, k);
                    if (yN <= y12) 
                    {
                        yh = yN-y14;
                        xh = xN-x14;
                        fBx = -1.0;
                        fBy = 1.0;
                    }
                    else 
                    {
                        yh = yN-y34;
                        xh = xN-x34;
                        fBx = 1.0;
                        fBy = -1.0;
                    }

                    cosyh = cos(2.0*M_PI*yh/ym);
                    cosxh = cos(2.0*M_PI*xh/xm);
                    sinyh = sin(2.0*M_PI*yh/ym);
                    sinxh = sin(2.0*M_PI*xh/xm);
        
                    Bxc[i][j][k] = fBx * B_BG * tanh(yh/delta_CS);
                    
                    //* Add perturbation
                    Bxc[i][j][k] = Bxc[i][j][k] * (1.0 + perturbation*cosxh*cosyh*cosyh) + fBx*2.0*perturbation*cosxh*2.0*M_PI/ym*cosyh*sinyh 
                                                * (B_BG*delta_CS*LOG_COSH(y14/delta_CS)-B_BG*delta_CS*LOG_COSH(yh/delta_CS));
        
                    Byc[i][j][k] = fBy*2.0*perturbation*M_PI/xm*sinxh*cosyh*cosyh * (B_BG*delta_CS*LOG_COSH(y14/delta_CS)-delta_CS*B_BG*LOG_COSH(yh/delta_CS));
        
                    //* Guide field
                    Bzc[i][j][k] = B_BG*guide_field_ratio;
                }

        for (int i = 0; i < nxn; i++)
            for (int j = 0; j < nyn; j++)
                for (int k = 0; k < nzn; k++)
                {
                    //* Initialise E on nodes
                    Ex[i][j][k] = 0.0;
                    Ey[i][j][k] = 0.0;
                    Ez[i][j][k] = 0.0;
                }
        
        //* Communicate ghost data at cell centres
        communicateCenterBC(nxc, nyc, nzc, Bxc, col->bcBx[0],col->bcBx[1],col->bcBx[2],col->bcBx[3],col->bcBx[4],col->bcBx[5], vct, this);
        communicateCenterBC(nxc, nyc, nzc, Byc, col->bcBy[0],col->bcBy[1],col->bcBy[2],col->bcBy[3],col->bcBy[4],col->bcBy[5], vct, this);
        communicateCenterBC(nxc, nyc, nzc, Bzc, col->bcBz[0],col->bcBz[1],col->bcBz[2],col->bcBz[3],col->bcBz[4],col->bcBz[5], vct, this);

        //* Initialise B on nodes
        grid->interpC2N(Bxn, Bxc);
        grid->interpC2N(Byn, Byc);
        grid->interpC2N(Bzn, Bzc);
       
        //* Communicate ghost data on nodes
        communicateNodeBC(nxn, nyn, nzn, Bxn, col->bcBx[0],col->bcBx[1],col->bcBx[2],col->bcBx[3],col->bcBx[4],col->bcBx[5], vct, this);
        communicateNodeBC(nxn, nyn, nzn, Byn, col->bcBy[0],col->bcBy[1],col->bcBy[2],col->bcBy[3],col->bcBy[4],col->bcBy[5], vct, this);
        communicateNodeBC(nxn, nyn, nzn, Bzn, col->bcBz[0],col->bcBz[1],col->bcBz[2],col->bcBz[3],col->bcBz[4],col->bcBz[5], vct, this);
    }
    else
        init();  //! READ FROM RESTART
}

//? Relativistic double Harris for ion-electron plasma: Maxwellian background, drifting particles in the sheets
void EMfields3D::init_Relativistic_Double_Harris_ion_electron()
{
    const Collective *col = &get_col();
    const VirtualTopology3D *vct = &get_vct();
    const Grid *grid = &get_grid();

    //* Custom input parameters for relativistic reconnection
    const double sigma                  = input_param[0];       //* Magnetisation parameter
    const double eta                    = input_param[1];       //* Ratio of current sheet density to upstream density (this is "alpha" in Fabio's paper; Eqs 52 and 53)
    const double delta_CS               = input_param[2];       //* Half-thickness of current sheet (free parameter)
    const double perturbation           = input_param[3];       //* Amplitude of initial perturbation
    const double guide_field_ratio      = input_param[4];       //* Ratio of guide field to in-plane magnetic field

    //* Background (BG) or upstream particles
    double thermal_spread_BG_electrons  = col->getUth(0);                           //* Thermal spread of electrons
    double thermal_spread_BG_ions       = col->getUth(1);                           //* Thermal spread of ions
    double rho_BG                       = rhoINIT[0]/(4.0*M_PI);                    //* Density (rho_BG = n * mc^2)
    double B_BG                         = sqrt(sigma*4.0*M_PI*rho_BG);              //* sigma = B^2/(4*pi*rho_electrons)
    
    //* Current sheet (CS) particles
    double rho_CS                       = eta*rho_BG;                                            //* Density (rho_CS = eta * n * mc^2)
    double drift_velocity               = B_BG/(8.0*M_PI*rho_CS*delta_CS/c);                     //* v = B*c/(8 * pi * rho_CS * delta_CS); Eq 52
    double lorentz_factor_CS            = 1.0/sqrt(1.0 - drift_velocity*drift_velocity);         //* Lorentz factor of the relativistic drifting particles
    double thermal_spread_CS_ions       = B_BG*B_BG*lorentz_factor_CS/(16.0*M_PI*rho_CS);        //* Thermal spread of ions (B^2 * Gamma/(16 * pi * eta * n * mc^2)); Eq 53
    double thermal_spread_CS_electrons  = thermal_spread_CS_ions * fabs(col->getQOM(0));         //* Thermal spread of electrons (Ratio of thermal spread = mass ratio)
    
    if (restart_status == 0) 
    {
        if (vct->getCartesian_rank() == 0) 
        {
            cout << "-----------------------------------------------------------"   << endl;
            cout << "Relativistic double Harris sheet for ion-electron plasma"      << endl;
            cout << "-----------------------------------------------------------"   << endl << endl; 
            
            cout << "Ratio of CS density to upstream density            = " << eta                      << endl;
            cout << "Perturbation amplitude                             = " << perturbation             << endl; 
            cout << "Ratio of guide magnetic field to background field  = " << guide_field_ratio        << endl << endl; 
            
            cout << "BACKGROUND/UPSTREAM:"                                                              << endl;
            cout << "   Magnetisation parameter (ions)          = " << sigma                            << endl; 
            cout << "   Plasma beta                             = " << 2.0*rho_BG*thermal_spread_BG_ions/(B_BG*B_BG/2.0/FourPI)  << endl;
            cout << "   Thermal spread of ions                  = " << thermal_spread_BG_ions           << endl; 
            cout << "   Thermal spread of electrons             = " << thermal_spread_BG_electrons      << endl; 
            cout << "   Lorentz factor of electrons            ~= " << 3*thermal_spread_BG_electrons    << endl << endl;

            cout << "CURRENT SHEET:"                                                                    << endl;
            cout << "   Thermal spread of drifting ions         = " << thermal_spread_CS_ions           << endl; 
            cout << "   Thermal spread of drifting electrons    = " << thermal_spread_CS_electrons      << endl; 
            cout << "   Lorentz factor of drifting particles    = " << lorentz_factor_CS                << endl; 

            cout << "-----------------------------------------------------------"   << endl;
        }
  
        //* Params for setting up current sheet
        double x14=Lx/4.0;
        double x34=3.0*Lx/4.0;
        double y12=Ly/2.0;
        double y14=Ly/4.0;
        double y34=3.0*Ly/4.0;
        double ym=Ly;           // 4 times the perturbation height
        double xm=Lx;           // perturbation wavelength

        double xN, yN, yh, xh, cosyh, cosxh, sinyh, sinxh;
        double fBx, fBy;
        
        for (int i = 1; i < nxc-1; i++)
            for (int j = 1; j < nyc-1; j++)
                for (int k = 1; k < nzc-1; k++) 
                {
                    double xN = grid->getXC(i, j, k);
                    double yN = grid->getYC(i, j, k);
                    if (yN <= y12) 
                    {
                        yh = yN-y14;
                        xh = xN-x14;
                        fBx = -1.0;
                        fBy = 1.0;
                    }
                    else 
                    {
                        yh = yN-y34;
                        xh = xN-x34;
                        fBx = 1.0;
                        fBy = -1.0;
                    }
                    
                    cosyh = cos(2.0*M_PI*yh/ym);
                    cosxh = cos(2.0*M_PI*xh/xm);
                    sinyh = sin(2.0*M_PI*yh/ym);
                    sinxh = sin(2.0*M_PI*xh/xm);
        
                    Bxc[i][j][k] = fBx * B_BG * tanh(yh/delta_CS);

                    //* Add perturbation
                    Bxc[i][j][k] = Bxc[i][j][k] * (1.0+perturbation*cosxh*cosyh*cosyh) + fBx*2.0*perturbation*cosxh*2.0*M_PI/ym*cosyh*sinyh 
                                                * (B_BG*delta_CS*LOG_COSH(y14/delta_CS)-B_BG*delta_CS*LOG_COSH(yh/delta_CS));
        
                    Byc[i][j][k] = fBy*2.0*perturbation*M_PI/xm*sinxh*cosyh*cosyh * (B_BG*delta_CS*LOG_COSH(y14/delta_CS)-delta_CS*B_BG*LOG_COSH(yh/delta_CS));
        
                    //* Guide field
                    Bzc[i][j][k] = B_BG*guide_field_ratio;
                }

        for (int i = 0; i < nxn; i++)
            for (int j = 0; j < nyn; j++)
                for (int k = 0; k < nzn; k++)
                {
                    //* Initialise E on nodes
                    Ex[i][j][k] = 0.0;
                    Ey[i][j][k] = 0.0;
                    Ez[i][j][k] = 0.0;
                }
        
        //* Communicate ghost data at cell centres
        communicateCenterBC(nxc, nyc, nzc, Bxc, col->bcBx[0],col->bcBx[1],col->bcBx[2],col->bcBx[3],col->bcBx[4],col->bcBx[5], vct, this);
        communicateCenterBC(nxc, nyc, nzc, Byc, col->bcBy[0],col->bcBy[1],col->bcBy[2],col->bcBy[3],col->bcBy[4],col->bcBy[5], vct, this);
        communicateCenterBC(nxc, nyc, nzc, Bzc, col->bcBz[0],col->bcBz[1],col->bcBz[2],col->bcBz[3],col->bcBz[4],col->bcBz[5], vct, this);

        //* Initialise B on nodes
        grid->interpC2N(Bxn, Bxc);
        grid->interpC2N(Byn, Byc);
        grid->interpC2N(Bzn, Bzc);
       
        //* Communicate ghost data on nodes
        communicateNodeBC(nxn, nyn, nzn, Bxn, col->bcBx[0],col->bcBx[1],col->bcBx[2],col->bcBx[3],col->bcBx[4],col->bcBx[5], vct, this);
        communicateNodeBC(nxn, nyn, nzn, Byn, col->bcBy[0],col->bcBy[1],col->bcBy[2],col->bcBy[3],col->bcBy[4],col->bcBy[5], vct, this);
        communicateNodeBC(nxn, nyn, nzn, Bzn, col->bcBz[0],col->bcBz[1],col->bcBz[2],col->bcBz[3],col->bcBz[4],col->bcBz[5], vct, this);
    }
    else
        init();  //! READ FROM RESTART
}

//* =========================================================================================================== *//

/*! Calculate the susceptibility on the boundary leftX */
void EMfields3D::sustensorLeftX(double **susxx, double **susyx, double **suszx) 
{
  double beta, omcx, omcy, omcz, denom;
  for (int j = 0; j < nyn; j++)
    for (int k = 0; k < nzn; k++) {
      susxx[j][k] = 1.0;
      susyx[j][k] = 0.0;
      suszx[j][k] = 0.0;
    }
  for (int is = 0; is < ns; is++) {
    beta = .5 * qom[is] * dt / c;
    for (int j = 0; j < nyn; j++)
      for (int k = 0; k < nzn; k++) {
        omcx = beta * (Bxn[1][j][k]+Bx_ext[1][j][k]);
        omcy = beta * (Byn[1][j][k]+By_ext[1][j][k]);
        omcz = beta * (Bzn[1][j][k]+Bz_ext[1][j][k]);
        denom = FourPI / 2 * delt * dt / c * qom[is] * rhons[is][1][j][k] / (1.0 + omcx * omcx + omcy * omcy + omcz * omcz);
        susxx[j][k] += (  1.0 + omcx * omcx) * denom;
        susyx[j][k] += (-omcz + omcx * omcy) * denom;
        suszx[j][k] += ( omcy + omcx * omcz) * denom;
      }
  }

}

/*! Calculate the susceptibility on the boundary rightX */
void EMfields3D::sustensorRightX(double **susxx, double **susyx, double **suszx) 
{
  double beta, omcx, omcy, omcz, denom;
  for (int j = 0; j < nyn; j++)
    for (int k = 0; k < nzn; k++) {
      susxx[j][k] = 1.0;
      susyx[j][k] = 0.0;
      suszx[j][k] = 0.0;
    }
  for (int is = 0; is < ns; is++) {
    beta = .5 * qom[is] * dt / c;
    for (int j = 0; j < nyn; j++)
      for (int k = 0; k < nzn; k++) {
        omcx = beta * (Bxn[nxn - 2][j][k]+Bx_ext[nxn - 2][j][k]);
        omcy = beta * (Byn[nxn - 2][j][k]+By_ext[nxn - 2][j][k]);
        omcz = beta * (Bzn[nxn - 2][j][k]+Bz_ext[nxn - 2][j][k]);
        denom = FourPI / 2 * delt * dt / c * qom[is] * rhons[is][nxn - 2][j][k] / (1.0 + omcx * omcx + omcy * omcy + omcz * omcz);
        susxx[j][k] += (  1.0 + omcx * omcx) * denom;
        susyx[j][k] += (-omcz + omcx * omcy) * denom;
        suszx[j][k] += ( omcy + omcx * omcz) * denom;
      }
  }
}

/*! Calculate the susceptibility on the boundary left */
void EMfields3D::sustensorLeftY(double **susxy, double **susyy, double **suszy) 
{
  double beta, omcx, omcy, omcz, denom;
  for (int i = 0; i < nxn; i++)
    for (int k = 0; k < nzn; k++) {
      susxy[i][k] = 0.0;
      susyy[i][k] = 1.0;
      suszy[i][k] = 0.0;
    }
  for (int is = 0; is < ns; is++) {
    beta = .5 * qom[is] * dt / c;
    for (int i = 0; i < nxn; i++)
      for (int k = 0; k < nzn; k++) {
        omcx = beta * (Bxn[i][1][k]+Bx_ext[i][1][k]);
        omcy = beta * (Byn[i][1][k]+By_ext[i][1][k]);
        omcz = beta * (Bzn[i][1][k]+Bz_ext[i][1][k]);
        denom = FourPI / 2 * delt * dt / c * qom[is] * rhons[is][i][1][k] / (1.0 + omcx * omcx + omcy * omcy + omcz * omcz);
        susxy[i][k] += ( omcz + omcx * omcy) * denom;
        susyy[i][k] += (  1.0 + omcy * omcy) * denom;
        suszy[i][k] += (-omcx + omcy * omcz) * denom;
      }
  }

}

/*! Calculate the susceptibility on the boundary right */
void EMfields3D::sustensorRightY(double **susxy, double **susyy, double **suszy) 
{
  double beta, omcx, omcy, omcz, denom;
  for (int i = 0; i < nxn; i++)
    for (int k = 0; k < nzn; k++) {
      susxy[i][k] = 0.0;
      susyy[i][k] = 1.0;
      suszy[i][k] = 0.0;
    }
  for (int is = 0; is < ns; is++) {
    beta = .5 * qom[is] * dt / c;
    for (int i = 0; i < nxn; i++)
      for (int k = 0; k < nzn; k++) {
        omcx = beta * (Bxn[i][nyn - 2][k]+Bx_ext[i][nyn - 2][k]);
        omcy = beta * (Byn[i][nyn - 2][k]+By_ext[i][nyn - 2][k]);
        omcz = beta * (Bzn[i][nyn - 2][k]+Bz_ext[i][nyn - 2][k]);
        denom = FourPI / 2 * delt * dt / c * qom[is] * rhons[is][i][nyn - 2][k] / (1.0 + omcx * omcx + omcy * omcy + omcz * omcz);
        susxy[i][k] += ( omcz + omcx * omcy) * denom;
        susyy[i][k] += (  1.0 + omcy * omcy) * denom;
        suszy[i][k] += (-omcx + omcy * omcz) * denom;
      }
  }
}

/*! Calculate the susceptibility on the boundary left */
void EMfields3D::sustensorLeftZ(double **susxz, double **susyz, double **suszz) 
{
  double beta, omcx, omcy, omcz, denom;
  for (int i = 0; i < nxn; i++)
    for (int j = 0; j < nyn; j++) {
      susxz[i][j] = 0.0;
      susyz[i][j] = 0.0;
      suszz[i][j] = 1.0;
    }
  for (int is = 0; is < ns; is++) {
    beta = .5 * qom[is] * dt / c;
    for (int i = 0; i < nxn; i++)
      for (int j = 0; j < nyn; j++) {
        omcx = beta * (Bxn[i][j][1]+Bx_ext[i][j][1]);
        omcy = beta * (Byn[i][j][1]+By_ext[i][j][1]);
        omcz = beta * (Bzn[i][j][1]+Bz_ext[i][j][1]);
        denom = FourPI / 2 * delt * dt / c * qom[is] * rhons[is][i][j][1] / (1.0 + omcx * omcx + omcy * omcy + omcz * omcz);
        susxz[i][j] += (-omcy + omcx * omcz) * denom;
        susyz[i][j] += ( omcx + omcy * omcz) * denom;
        suszz[i][j] += (  1.0 + omcz * omcz) * denom;
      }
  }

}

/*! Calculate the susceptibility on the boundary right */
void EMfields3D::sustensorRightZ(double **susxz, double **susyz, double **suszz) 
{
  double beta, omcx, omcy, omcz, denom;
  for (int i = 0; i < nxn; i++)
    for (int j = 0; j < nyn; j++) {
      susxz[i][j] = 0.0;
      susyz[i][j] = 0.0;
      suszz[i][j] = 1.0;
    }
  for (int is = 0; is < ns; is++) {
    beta = .5 * qom[is] * dt / c;
    for (int i = 0; i < nxn; i++)
      for (int j = 0; j < nyn; j++) {
        omcx = beta * (Bxn[i][j][nzn - 2]+Bx_ext[i][j][nzn - 2]);
        omcy = beta * (Byn[i][j][nzn - 2]+By_ext[i][j][nzn - 2]);
        omcz = beta * (Bzn[i][j][nzn - 2]+Bz_ext[i][j][nzn - 2]);
        denom = FourPI / 2 * delt * dt / c * qom[is] * rhons[is][i][j][nyn - 2] / (1.0 + omcx * omcx + omcy * omcy + omcz * omcz);
        susxz[i][j] += (-omcy + omcx * omcz) * denom;
        susyz[i][j] += ( omcx + omcy * omcz) * denom;
        suszz[i][j] += (  1.0 + omcz * omcz) * denom;
      }
  }
}

/*! Perfect conductor boundary conditions: LEFT wall */
void EMfields3D::perfectConductorLeft(arr3_double imageX, arr3_double imageY, arr3_double imageZ, const_arr3_double vectorX, const_arr3_double vectorY, const_arr3_double vectorZ, int dir)
{
  double** susxy;
  double** susyy;
  double** suszy;
  double** susxx;
  double** susyx;
  double** suszx;
  double** susxz;
  double** susyz;
  double** suszz;
  switch(dir){
    case 0:  // boundary condition on X-DIRECTION 
      susxx = newArr2(double,nyn,nzn);
      susyx = newArr2(double,nyn,nzn);
      suszx = newArr2(double,nyn,nzn);
      sustensorLeftX(susxx, susyx, suszx);
      for (int i=1; i <  nyn-1;i++)
        for (int j=1; j <  nzn-1;j++){
          imageX[1][i][j] = vectorX.get(1,i,j) - (Ex[1][i][j] - susyx[i][j]*vectorY.get(1,i,j) - suszx[i][j]*vectorZ.get(1,i,j) - Jxh[1][i][j]*dt*th*FourPI)/susxx[i][j];
          imageY[1][i][j] = vectorY.get(1,i,j) - 0.0*vectorY.get(2,i,j);
          imageZ[1][i][j] = vectorZ.get(1,i,j) - 0.0*vectorZ.get(2,i,j);
        }
      delArr2(susxx,nxn);
      delArr2(susyx,nxn);
      delArr2(suszx,nxn);
      break;
    case 1: // boundary condition on Y-DIRECTION
      susxy = newArr2(double,nxn,nzn);
      susyy = newArr2(double,nxn,nzn);
      suszy = newArr2(double,nxn,nzn);
      sustensorLeftY(susxy, susyy, suszy);
      for (int i=1; i < nxn-1;i++)
        for (int j=1; j <  nzn-1;j++){
          imageX[i][1][j] = vectorX.get(i,1,j) - 0.0*vectorX.get(i,2,j);
          imageY[i][1][j] = vectorY.get(i,1,j) - (Ey[i][1][j] - susxy[i][j]*vectorX.get(i,1,j) - suszy[i][j]*vectorZ.get(i,1,j) - Jyh[i][1][j]*dt*th*FourPI)/susyy[i][j];
          imageZ[i][1][j] = vectorZ.get(i,1,j) - 0.0*vectorZ.get(i,2,j);
        }
      delArr2(susxy,nxn);
      delArr2(susyy,nxn);
      delArr2(suszy,nxn);
      break;
    case 2: // boundary condition on Z-DIRECTION
      susxz = newArr2(double,nxn,nyn);
      susyz = newArr2(double,nxn,nyn);
      suszz = newArr2(double,nxn,nyn);
      sustensorLeftZ(susxz, susyz, suszz);
      for (int i=1; i <  nxn-1;i++)
        for (int j=1; j <  nyn-1;j++){
          imageX[i][j][1] = vectorX.get(i,j,1);
          imageY[i][j][1] = vectorX.get(i,j,1);
          imageZ[i][j][1] = vectorZ.get(i,j,1) - (Ez[i][j][1] - susxz[i][j]*vectorX.get(i,j,1) - susyz[i][j]*vectorY.get(i,j,1) - Jzh[i][j][1]*dt*th*FourPI)/suszz[i][j];
        }
      delArr2(susxz,nxn);
      delArr2(susyz,nxn);
      delArr2(suszz,nxn);
      break;
  }
}

/*! Perfect conductor boundary conditions: RIGHT wall */
void EMfields3D::perfectConductorRight(arr3_double imageX, arr3_double imageY, arr3_double imageZ, const_arr3_double vectorX, const_arr3_double vectorY, const_arr3_double vectorZ, int dir)
{
  double beta, omcx, omcy, omcz, denom;
  double** susxy;
  double** susyy;
  double** suszy;
  double** susxx;
  double** susyx;
  double** suszx;
  double** susxz;
  double** susyz;
  double** suszz;
  switch(dir){
    case 0: // boundary condition on X-DIRECTION RIGHT
      susxx = newArr2(double,nyn,nzn);
      susyx = newArr2(double,nyn,nzn);
      suszx = newArr2(double,nyn,nzn);
      sustensorRightX(susxx, susyx, suszx);
      for (int i=1; i < nyn-1;i++)
        for (int j=1; j <  nzn-1;j++){
          imageX[nxn-2][i][j] = vectorX.get(nxn-2,i,j) - (Ex[nxn-2][i][j] - susyx[i][j]*vectorY.get(nxn-2,i,j) - suszx[i][j]*vectorZ.get(nxn-2,i,j) - Jxh[nxn-2][i][j]*dt*th*FourPI)/susxx[i][j];
          imageY[nxn-2][i][j] = vectorY.get(nxn-2,i,j) - 0.0 * vectorY.get(nxn-3,i,j);
          imageZ[nxn-2][i][j] = vectorZ.get(nxn-2,i,j) - 0.0 * vectorZ.get(nxn-3,i,j);
        }
      delArr2(susxx,nxn);
      delArr2(susyx,nxn);       
      delArr2(suszx,nxn);
      break;
    case 1: // boundary condition on Y-DIRECTION RIGHT
      susxy = newArr2(double,nxn,nzn);
      susyy = newArr2(double,nxn,nzn);
      suszy = newArr2(double,nxn,nzn);
      sustensorRightY(susxy, susyy, suszy);
      for (int i=1; i < nxn-1;i++)
        for (int j=1; j < nzn-1;j++){
          imageX[i][nyn-2][j] = vectorX.get(i,nyn-2,j) - 0.0*vectorX.get(i,nyn-3,j);
          imageY[i][nyn-2][j] = vectorY.get(i,nyn-2,j) - (Ey[i][nyn-2][j] - susxy[i][j]*vectorX.get(i,nyn-2,j) - suszy[i][j]*vectorZ.get(i,nyn-2,j) - Jyh[i][nyn-2][j]*dt*th*FourPI)/susyy[i][j];
          imageZ[i][nyn-2][j] = vectorZ.get(i,nyn-2,j) - 0.0*vectorZ.get(i,nyn-3,j);
        }
      delArr2(susxy,nxn);
      delArr2(susyy,nxn);
      delArr2(suszy,nxn);
      break;
    case 2: // boundary condition on Z-DIRECTION RIGHT
      susxz = newArr2(double,nxn,nyn);
      susyz = newArr2(double,nxn,nyn);
      suszz = newArr2(double,nxn,nyn);
      sustensorRightZ(susxz, susyz, suszz);
      for (int i=1; i < nxn-1;i++)
        for (int j=1; j < nyn-1;j++){
          imageX[i][j][nzn-2] = vectorX.get(i,j,nzn-2);
          imageY[i][j][nzn-2] = vectorY.get(i,j,nzn-2);
          imageZ[i][j][nzn-2] = vectorZ.get(i,j,nzn-2) - (Ez[i][j][nzn-2] - susxz[i][j]*vectorX.get(i,j,nzn-2) - susyz[i][j]*vectorY.get(i,j,nzn-2) - Jzh[i][j][nzn-2]*dt*th*FourPI)/suszz[i][j];
        }
      delArr2(susxz,nxn);
      delArr2(susyz,nxn);       
      delArr2(suszz,nxn);
      break;
  }
}

/*! Perfect conductor boundary conditions for source: LEFT WALL */
void EMfields3D::perfectConductorLeftS(arr3_double vectorX, arr3_double vectorY, arr3_double vectorZ, int dir) 
{

  double ebc[3];

  // Assuming E = - ve x B
  cross_product(ue0,ve0,we0,B0x,B0y,B0z,ebc);
  scale(ebc,-1.0,3);

  switch(dir){
    case 0: // boundary condition on X-DIRECTION LEFT
      for (int i=1; i < nyn-1;i++)
        for (int j=1; j < nzn-1;j++){
          vectorX[1][i][j] = 0.0;
          vectorY[1][i][j] = ebc[1];
          vectorZ[1][i][j] = ebc[2];
          //+//          vectorX[1][i][j] = 0.0;
          //+//          vectorY[1][i][j] = 0.0;
          //+//          vectorZ[1][i][j] = 0.0;
        }
      break;
    case 1: // boundary condition on Y-DIRECTION LEFT
      for (int i=1; i < nxn-1;i++)
        for (int j=1; j < nzn-1;j++){
          vectorX[i][1][j] = ebc[0];
          vectorY[i][1][j] = 0.0;
          vectorZ[i][1][j] = ebc[2];
          //+//          vectorX[i][1][j] = 0.0;
          //+//          vectorY[i][1][j] = 0.0;
          //+//          vectorZ[i][1][j] = 0.0;
        }
      break;
    case 2: // boundary condition on Z-DIRECTION LEFT
      for (int i=1; i < nxn-1;i++)
        for (int j=1; j <  nyn-1;j++){
          vectorX[i][j][1] = ebc[0];
          vectorY[i][j][1] = ebc[1];
          vectorZ[i][j][1] = 0.0;
          //+//          vectorX[i][j][1] = 0.0;
          //+//          vectorY[i][j][1] = 0.0;
          //+//          vectorZ[i][j][1] = 0.0;
        }
      break;
  }
}

/*! Perfect conductor boundary conditions for source: RIGHT WALL */
void EMfields3D::perfectConductorRightS(arr3_double vectorX, arr3_double vectorY, arr3_double vectorZ, int dir) 
{

  double ebc[3];

  // Assuming E = - ve x B
  cross_product(ue0,ve0,we0,B0x,B0y,B0z,ebc);
  scale(ebc,-1.0,3);

  switch(dir){
    case 0: // boundary condition on X-DIRECTION RIGHT
      for (int i=1; i < nyn-1;i++)
        for (int j=1; j < nzn-1;j++){
          vectorX[nxn-2][i][j] = 0.0;
          vectorY[nxn-2][i][j] = ebc[1];
          vectorZ[nxn-2][i][j] = ebc[2];
          //+//          vectorX[nxn-2][i][j] = 0.0;
          //+//          vectorY[nxn-2][i][j] = 0.0;
          //+//          vectorZ[nxn-2][i][j] = 0.0;
        }
      break;
    case 1: // boundary condition on Y-DIRECTION RIGHT
      for (int i=1; i < nxn-1;i++)
        for (int j=1; j < nzn-1;j++){
          vectorX[i][nyn-2][j] = ebc[0];
          vectorY[i][nyn-2][j] = 0.0;
          vectorZ[i][nyn-2][j] = ebc[2];
          //+//          vectorX[i][nyn-2][j] = 0.0;
          //+//          vectorY[i][nyn-2][j] = 0.0;
          //+//          vectorZ[i][nyn-2][j] = 0.0;
        }
      break;
    case 2:
      for (int i=1; i <  nxn-1;i++)
        for (int j=1; j <  nyn-1;j++){
          vectorX[i][j][nzn-2] = ebc[0];
          vectorY[i][j][nzn-2] = ebc[1];
          vectorZ[i][j][nzn-2] = 0.0;
          //+//          vectorX[i][j][nzn-2] = 0.0;
          //+//          vectorY[i][j][nzn-2] = 0.0;
          //+//          vectorZ[i][j][nzn-2] = 0.0;
        }
      break;
  }
}

void EMfields3D::OpenBoundaryInflowEImage(arr3_double imageX, arr3_double imageY, arr3_double imageZ, const_arr3_double vectorX, const_arr3_double vectorY, const_arr3_double vectorZ, int nx, int ny, int nz)
{
  const VirtualTopology3D *vct = &get_vct();
  // Assuming E = - ve x B
  double injE[3];
  cross_product(ue0,ve0,we0,B0x,B0y,B0z,injE);
  scale(injE,-1.0,3);

  if(vct->getXleft_neighbor()==MPI_PROC_NULL && bcEMfaceXleft == 2) 
  {
    for (int j=1; j < ny-1;j++)
      for (int k=1; k < nz-1;k++){
        imageX[0][j][k] = vectorX[0][j][k] - injE[0];
        imageY[0][j][k] = vectorY[0][j][k] - injE[1];
        imageZ[0][j][k] = vectorZ[0][j][k] - injE[2];
      }
  }
}

void EMfields3D::OpenBoundaryInflowB(arr3_double vectorX, arr3_double vectorY, arr3_double vectorZ, int nx, int ny, int nz)
{
  const VirtualTopology3D *vct = &get_vct();

  if(vct->getXleft_neighbor()==MPI_PROC_NULL && bcEMfaceXleft ==2 && nx>10) {
    for (int j=0; j < ny;j++)
      for (int k=0; k < nz;k++){
          
	vectorX[0][j][k] = B0x;
        vectorY[0][j][k] = B0y;
        vectorZ[0][j][k] = B0z;

	vectorX[1][j][k] = B0x;
	vectorY[1][j][k] = B0y;
	vectorZ[1][j][k] = B0z;
		
	vectorX[2][j][k] = B0x;
	vectorY[2][j][k] = B0y;
	vectorZ[2][j][k] = B0z;

	vectorX[3][j][k] = B0x;
	vectorY[3][j][k] = B0y;
	vectorZ[3][j][k] = B0z;

      }
  }

  if(vct->getXright_neighbor()==MPI_PROC_NULL && bcEMfaceXright ==2 && nx>10 ) {
    for (int j=0; j < ny;j++)
      for (int k=0; k < nz;k++){

        vectorX[nx-4][j][k] = vectorX[nx-5][j][k];
        vectorY[nx-4][j][k] = vectorY[nx-5][j][k];
        vectorZ[nx-4][j][k] = vectorZ[nx-5][j][k];

        vectorX[nx-3][j][k] = vectorX[nx-5][j][k];
        vectorY[nx-3][j][k] = vectorY[nx-5][j][k];
        vectorZ[nx-3][j][k] = vectorZ[nx-5][j][k];

        vectorX[nx-2][j][k] = vectorX[nx-5][j][k];
        vectorY[nx-2][j][k] = vectorY[nx-5][j][k];
        vectorZ[nx-2][j][k] = vectorZ[nx-5][j][k];

        vectorX[nx-1][j][k] = vectorX[nx-5][j][k];
        vectorY[nx-1][j][k] = vectorY[nx-5][j][k];
        vectorZ[nx-1][j][k] = vectorZ[nx-5][j][k];
      }
  }

  if(vct->getYleft_neighbor()==MPI_PROC_NULL && bcEMfaceYleft ==2 && ny> 10)  {
    for (int i=0; i < nx;i++)
      for (int k=0; k < nz;k++){

    	  vectorX[i][0][k] = vectorX[i][4][k];
    	  vectorY[i][0][k] = vectorY[i][4][k];
    	  vectorZ[i][0][k] = vectorZ[i][4][k];

    	  vectorX[i][1][k] = vectorX[i][4][k];
    	  vectorY[i][1][k] = vectorY[i][4][k];
    	  vectorZ[i][1][k] = vectorZ[i][4][k];

    	  vectorX[i][2][k] = vectorX[i][4][k];
    	  vectorY[i][2][k] = vectorY[i][4][k];
    	  vectorZ[i][2][k] = vectorZ[i][4][k];

    	  vectorX[i][3][k] = vectorX[i][4][k];
    	  vectorY[i][3][k] = vectorY[i][4][k];
    	  vectorZ[i][3][k] = vectorZ[i][4][k];
      } 
  }

  if(vct->getYright_neighbor()==MPI_PROC_NULL && bcEMfaceYright==2 && ny>10)  {
    for (int i=0; i < nx;i++)
      for (int k=0; k< nz;k++){

    	vectorX[i][ny-4][k] = vectorX[i][ny-5][k];
        vectorY[i][ny-4][k] = vectorY[i][ny-5][k];
        vectorZ[i][ny-4][k] = vectorZ[i][ny-5][k];

        vectorX[i][ny-3][k] = vectorX[i][ny-5][k];
        vectorY[i][ny-3][k] = vectorY[i][ny-5][k];
        vectorZ[i][ny-3][k] = vectorZ[i][ny-5][k];

        vectorX[i][ny-2][k] = vectorX[i][ny-5][k];
        vectorY[i][ny-2][k] = vectorY[i][ny-5][k];
        vectorZ[i][ny-2][k] = vectorZ[i][ny-5][k];

        vectorX[i][ny-1][k] = vectorX[i][ny-5][k];
        vectorY[i][ny-1][k] = vectorY[i][ny-5][k];
        vectorZ[i][ny-1][k] = vectorZ[i][ny-5][k];
      }
  }

  if(vct->getZleft_neighbor()==MPI_PROC_NULL && bcEMfaceZleft ==2 && nz > 10)  {
    for (int i=0; i < nx;i++)
      for (int j=0; j < ny;j++){

    	  vectorX[i][j][0] = vectorX[i][j][4];
    	  vectorY[i][j][0] = vectorY[i][j][4];
    	  vectorZ[i][j][0] = vectorZ[i][j][4];

    	  vectorX[i][j][1] = vectorX[i][j][4];
    	  vectorY[i][j][1] = vectorY[i][j][4];
    	  vectorZ[i][j][1] = vectorZ[i][j][4];

    	  vectorX[i][j][2] = vectorX[i][j][4];
    	  vectorY[i][j][2] = vectorY[i][j][4];
    	  vectorZ[i][j][2] = vectorZ[i][j][4];

    	  vectorX[i][j][3] = vectorX[i][j][4];
    	  vectorY[i][j][3] = vectorY[i][j][4];
    	  vectorZ[i][j][3] = vectorZ[i][j][4];

      } 
  }

  if(vct->getZright_neighbor()==MPI_PROC_NULL && bcEMfaceZright ==2 && nz>10)  {
    for (int i=0; i < nx;i++)
      for (int j=0; j < ny;j++){

    	vectorX[i][j][nz-4] = vectorX[i][j][nz-5];
        vectorY[i][j][nz-4] = vectorY[i][j][nz-5];
        vectorZ[i][j][nz-4] = vectorZ[i][j][nz-5];

        vectorX[i][j][nz-3] = vectorX[i][j][nz-5];
        vectorY[i][j][nz-3] = vectorY[i][j][nz-5];
        vectorZ[i][j][nz-3] = vectorZ[i][j][nz-5];

        vectorX[i][j][nz-2] = vectorX[i][j][nz-5];
        vectorY[i][j][nz-2] = vectorY[i][j][nz-5];
        vectorZ[i][j][nz-2] = vectorZ[i][j][nz-5];

        vectorX[i][j][nz-1] = vectorX[i][j][nz-5];
        vectorY[i][j][nz-1] = vectorY[i][j][nz-5];
        vectorZ[i][j][nz-1] = vectorZ[i][j][nz-5];
      }
  }

}

void EMfields3D::OpenBoundaryInflowE(arr3_double vectorX, arr3_double vectorY, arr3_double vectorZ, int nx, int ny, int nz)
{
  const VirtualTopology3D *vct = &get_vct();
  // Assuming E = - ve x B
  double injE[3];
  cross_product(ue0,ve0,we0,B0x,B0y,B0z,injE);
  scale(injE,-1.0,3);

    if(vct->getXleft_neighbor()==MPI_PROC_NULL && bcEMfaceXleft ==2) 
    {
        for (int j=0; j < ny;j++)
            for (int k=0; k < nz;k++)
            {
                vectorX[1][j][k] = injE[0];
                vectorY[1][j][k] = injE[1];
                vectorZ[1][j][k] = injE[2];

                vectorX[2][j][k] = injE[0];
                vectorY[2][j][k] = injE[1];
                vectorZ[2][j][k] = injE[2];

                vectorX[3][j][k] = injE[0];
                vectorY[3][j][k] = injE[1];
                vectorZ[3][j][k] = injE[2];
            } 
    }
}

//* =========================================================================================================== *//

//*** Get energies ***//

//! Electric field energy
double EMfields3D::get_E_field_energy(void) 
{
    double localEenergy = 0.0;
    double totalEenergy = 0.0;

    for (int i = 1; i < nxn - 2; i++)
        for (int j = 1; j < nyn - 2; j++)
            for (int k = 1; k < nzn - 2; k++)
                localEenergy += .5 * dx * dy * dz * (Ex[i][j][k] * Ex[i][j][k] + Ey[i][j][k] * Ey[i][j][k] + Ez[i][j][k] * Ez[i][j][k]) / (FourPI);

    MPI_Allreduce(&localEenergy, &totalEenergy, 1, MPI_DOUBLE, MPI_SUM, (&get_vct())->getFieldComm());
    return (totalEenergy);
}

double EMfields3D::get_Ex_field_energy(void) 
{
    double localEenergy = 0.0;
    double totalEenergy = 0.0;

    for (int i = 1; i < nxn - 2; i++)
        for (int j = 1; j < nyn - 2; j++)
            for (int k = 1; k < nzn - 2; k++)
                localEenergy += .5 * dx * dy * dz * (Ex[i][j][k] * Ex[i][j][k]) / (FourPI);

    MPI_Allreduce(&localEenergy, &totalEenergy, 1, MPI_DOUBLE, MPI_SUM, (&get_vct())->getFieldComm());
    return (totalEenergy);
}

double EMfields3D::get_Ey_field_energy(void) 
{
    double localEenergy = 0.0;
    double totalEenergy = 0.0;

    for (int i = 1; i < nxn - 2; i++)
        for (int j = 1; j < nyn - 2; j++)
            for (int k = 1; k < nzn - 2; k++)
                localEenergy += .5 * dx * dy * dz * (Ey[i][j][k] * Ey[i][j][k]) / (FourPI);

    MPI_Allreduce(&localEenergy, &totalEenergy, 1, MPI_DOUBLE, MPI_SUM, (&get_vct())->getFieldComm());
    return (totalEenergy);
}

double EMfields3D::get_Ez_field_energy(void) 
{
    double localEenergy = 0.0;
    double totalEenergy = 0.0;

    for (int i = 1; i < nxn - 2; i++)
        for (int j = 1; j < nyn - 2; j++)
            for (int k = 1; k < nzn - 2; k++)
                localEenergy += .5 * dx * dy * dz * (Ez[i][j][k] * Ez[i][j][k]) / (FourPI);

    MPI_Allreduce(&localEenergy, &totalEenergy, 1, MPI_DOUBLE, MPI_SUM, (&get_vct())->getFieldComm());
    return (totalEenergy);
}

//*! Get internal magnetic field energy
double EMfields3D::get_B_field_energy(void) 
{
    double localBenergy = 0.0;
    double totalBenergy = 0.0;
    double Bxt = 0.0;
    double Byt = 0.0;
    double Bzt = 0.0;

    for (int i = 1; i < nxc - 1; i++)
        for (int j = 1; j < nyc - 1; j++)
            for (int k = 1; k < nzc - 1; k++)
            {
                Bxt = Bxc[i][j][k];
                Byt = Byc[i][j][k];
                Bzt = Bzc[i][j][k];

                localBenergy += .5*dx*dy*dz*(Bxt*Bxt + Byt*Byt + Bzt*Bzt)/(FourPI);
            }

    MPI_Allreduce(&localBenergy, &totalBenergy, 1, MPI_DOUBLE, MPI_SUM, (&get_vct())->getFieldComm());
    return (totalBenergy);
}

double EMfields3D::get_Bx_field_energy(void) 
{
    double localBenergy = 0.0;
    double totalBenergy = 0.0;

    for (int i = 1; i < nxc - 1; i++)
        for (int j = 1; j < nyc - 1; j++)
            for (int k = 1; k < nzc - 1; k++)
                localBenergy += .5 * dx * dy * dz * (Bxc[i][j][k] * Bxc[i][j][k])/(FourPI);

    MPI_Allreduce(&localBenergy, &totalBenergy, 1, MPI_DOUBLE, MPI_SUM, (&get_vct())->getFieldComm());
    return (totalBenergy);
}

double EMfields3D::get_By_field_energy(void) 
{
    double localBenergy = 0.0;
    double totalBenergy = 0.0;

    for (int i = 1; i < nxc - 1; i++)
        for (int j = 1; j < nyc - 1; j++)
            for (int k = 1; k < nzc - 1; k++)
                localBenergy += .5 * dx * dy * dz * (Byc[i][j][k] * Byc[i][j][k])/(FourPI);

    MPI_Allreduce(&localBenergy, &totalBenergy, 1, MPI_DOUBLE, MPI_SUM, (&get_vct())->getFieldComm());
    return (totalBenergy);
}

double EMfields3D::get_Bz_field_energy(void) 
{
    double localBenergy = 0.0;
    double totalBenergy = 0.0;

    for (int i = 1; i < nxc - 1; i++)
        for (int j = 1; j < nyc - 1; j++)
            for (int k = 1; k < nzc - 1; k++)
                localBenergy += .5 * dx * dy * dz * (Bzc[i][j][k] * Bzc[i][j][k])/(FourPI);

    MPI_Allreduce(&localBenergy, &totalBenergy, 1, MPI_DOUBLE, MPI_SUM, (&get_vct())->getFieldComm());
    return (totalBenergy);
}

//*! Get external magnetic field energy
double EMfields3D::get_Bext_energy(void) 
{
    double localBenergy = 0.0;
    double totalBenergy = 0.0;
    double Bxt = 0.0;
    double Byt = 0.0;
    double Bzt = 0.0;

    for (int i = 1; i < nxc - 1; i++)
        for (int j = 1; j < nyc - 1; j++)
            for (int k = 1; k < nzc - 1; k++)
            {
                Bxt = Bxc_ext[i][j][k];
                Byt = Byc_ext[i][j][k];
                Bzt = Bzc_ext[i][j][k];

                localBenergy += .5*dx*dy*dz*(Bxt*Bxt + Byt*Byt + Bzt*Bzt)/(FourPI);
            }

    MPI_Allreduce(&localBenergy, &totalBenergy, 1, MPI_DOUBLE, MPI_SUM, (&get_vct())->getFieldComm());
    return (totalBenergy);
}

/*! get bulk kinetic energy*/
double EMfields3D::get_bulk_energy(int is) 
{
    double localBenergy = 0.0;
    double totalBenergy = 0.0;
    for (int i = 1; i < nxn - 2; i++)
        for (int j = 1; j < nyn - 2; j++)
            for (int k = 1; k < nzn - 2; k++)
                // Trying to avoid division by zero. Where rho iz 0, current must be 0.
                localBenergy += (fabs(rhons[is][i][j][k]) > 1.e-20) ? (0.5 * dx * dy * dz * (Jxs[is][i][j][k] * Jxs[is][i][j][k] + Jys[is][i][j][k] * Jys[is][i][j][k] + Jzs[is][i][j][k] * Jzs[is][i][j][k]) / rhons[is][i][j][k]) : 0.0;

    MPI_Allreduce(&localBenergy, &totalBenergy, 1, MPI_DOUBLE, MPI_SUM, (&get_vct())->getFieldComm());
    return (totalBenergy / qom[is]);
}

/*! Print info about electromagnetic field */
void EMfields3D::print(void) const { }

//* =========================================================================================================== *//

//*! Destructor !*//
EMfields3D::~EMfields3D() 
{
    delete [] qom;
    delete [] rhoINIT;
    for(int i=0;i<sizeMomentsArray;i++) { delete moments10Array[i]; }
    delete [] moments10Array;
    freeDataType();
}
