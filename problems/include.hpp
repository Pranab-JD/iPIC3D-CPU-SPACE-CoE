#pragma once

#include <mpi.h>
#include <random>
#include <cstring>
#include <iomanip>
#include <fstream>
#include <iostream>
#include <filesystem>

#include "CG.h"
#include "Basic.h"
#include "Alloc.h"
#include "debug.h"
#include "GMRES.h"
#include "string.h"
#include "ompdefs.h"
#include "Moments.h"
#include "asserts.h"
#include "Grid3DCU.h"
#include "ipichdf5.h"
#include "ipicmath.h"
#include "TimeTasks.h"
#include "EMfields3D.h"
#include "Collective.h"
#include "Parameters.h"
#include "Com3DNonblk.h"
#include "VCtopology3D.h"
#include "mic_particles.h"
#include "Particles3Dcomm.h"

#include "../LeXInt_Timer.hpp"

////? ============================================================= ?////

////? Some Generic Functions

//* sech^2(x) up to arbitrary precision
inline double sech_square(double x) 
{
    double y, res;
  
    if (fabs(x) > 354.0) 
        res = 1.31e-307;
    else 
    {                                                                                    
        y = 1.0/cosh(x);
        res = y*y;
    }
    return res;
}

//* Function to read fields from files to restart simulations
inline void EMfields3D::init_fields_restart()
{
    const Grid *grid = &get_grid();
    const Collective *col = &get_col();
    const VirtualTopology3D *vct = &get_vct();

    //! READ FROM RESTART
    #ifdef NO_HDF5
        eprintf("If you wish to restart simulations, you have to complie iPIC3D with HDF5");
    #else
        
        col->read_field_restart(vct, grid, Bxn, Byn, Bzn, Bxc, Byc, Bzc, Ex, Ey, Ez, rhoc_avg, divE_average, ns);

        //* Communicate ghost data for B at cell centres
        communicateCenterBC(nxc, nyc, nzc, Bxc, col->bcBx[0],col->bcBx[1],col->bcBx[2],col->bcBx[3],col->bcBx[4],col->bcBx[5], vct,this);
        communicateCenterBC(nxc, nyc, nzc, Byc, col->bcBy[0],col->bcBy[1],col->bcBy[2],col->bcBy[3],col->bcBy[4],col->bcBy[5], vct,this);
        communicateCenterBC(nxc, nyc, nzc, Bzc, col->bcBz[0],col->bcBz[1],col->bcBz[2],col->bcBz[3],col->bcBz[4],col->bcBz[5], vct,this);

        //* Communicate ghost data for rhoc_avg at cell centres
        communicateCenterBC(nxc, nyc, nzc, rhoc_avg, col->bcBx[0],col->bcBx[1],col->bcBx[2],col->bcBx[3],col->bcBx[4],col->bcBx[5], vct,this);

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
    
    #endif
}

//* Initialise uniform distribution of particles with a Maxellian velocity distribution
inline void Particles3D::maxwellian(Field * EMf)
{
    //! New initial setup
    if (col->getRestart_status() == 0)
    {
        //* Initialise random generator with different seed on different processor
        long long seed = (vct->getCartesian_rank() + 1)*20 + ns;
        srand(seed);
        srand48(seed);

        assert_eq(_pcls.size(), 0);

        const double q_sgn = (qom / fabs(qom));
        const double q_factor =  q_sgn * grid->getVOL() / npcel;

        for (int i = 1; i < grid->getNXC() - 1; i++)
            for (int j = 1; j < grid->getNYC() - 1; j++)
                for (int k = 1; k < grid->getNZC() - 1; k++)
                {
                    const double q = q_factor * fabs(EMf->getRHOcs(i, j, k, ns));
                    
                    for (int ii = 0; ii < npcelx; ii++)
                        for (int jj = 0; jj < npcely; jj++)
                            for (int kk = 0; kk < npcelz; kk++)
                            {
                                const double x = (ii + .5) * (dx / npcelx) + grid->getXN(i, j, k);
                                const double y = (jj + .5) * (dy / npcely) + grid->getYN(i, j, k);
                                const double z = (kk + .5) * (dz / npcelz) + grid->getZN(i, j, k);

                                double u, v, w;
                                sample_maxwellian(u, v, w, uth, vth, wth, u0, v0, w0);
                                create_new_particle(u, v, w, q, x, y, z);
                            }
                }

        fixPosition();
    }

    //! If col->getRestart_status() == 0 or 1, particle restart data is automatically read from the restart#.hdf files
}