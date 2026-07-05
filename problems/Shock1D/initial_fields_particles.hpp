//* This file needs to be included in main/iPIC3Dlib.cpp
//*
//* The functions used to define initial particles and field config need to be 
//* included in 'include/Particles3D.h' and 'include/EMfields3D.h', respectively.

#pragma once

#include "../include.hpp"

#ifndef NO_HDF5
#endif

using std::cout;
using std::endl;
using namespace iPic3D;


//? ========================================================================== ?//

//? Quasi-1D ion-electron shock (Relativistic and Non relativistic)
void Particles3D::Shock1D(Field * EMf) 
{
    //! New initial setup
    if (col->getRestart_status() == 0)
    {
        //* Initialise random generator with different seed on different processor
        long long seed = (vct->getCartesian_rank() + 1)*20 + ns;
        srand(seed);
        srand48(seed);

        assert_eq(_pcls.size(), 0);
    
        //? Parameters for relativistic cases - need to be defined outside of "if (col->getRelativistic())"
        double thermal_spread = col->getUth(ns);                                    //* Thermal spread
        double drift_velocity = col->getU0(ns);                                     //* Relativistic drift/bulk velocity
        double lorentz_factor = 1.0/sqrt(1.0 - drift_velocity*drift_velocity);      //* Lorentz factor

        const double Lx_half = Lx/2.0;
        const double q = (qom / fabs(qom)) * grid->getVOL() / npcel * col->getRHOinit(ns)/(4.0*M_PI);

        for (int i = 1; i < grid->getNXC() - 1; i++)
            for (int j = 1; j < grid->getNYC() - 1; j++)
                for (int k = 1; k < grid->getNZC() - 1; k++) 
                    for (int ii = 0; ii < npcelx; ii++)
                        for (int jj = 0; jj < npcely; jj++)
                            for (int kk = 0; kk < npcelz; kk++)
                            {
                                const double x = (ii + .5) * (dx / npcelx) + grid->getXN(i, j, k);
                                const double y = (jj + .5) * (dy / npcely) + grid->getYN(i, j, k);
                                const double z = (kk + .5) * (dz / npcelz) + grid->getZN(i, j, k);
    
                                //* Velocities of particles
                                double u, v, w;

                                if (col->getRelativistic())
                                {
                                    //? Relativistic (velocity of relativistic nondrifting Maxwellian)
                                    if(x < Lx_half) 
                                        sample_Maxwell_Juttner(u, v, w, thermal_spread, lorentz_factor, 1);
                                    else
                                        sample_Maxwell_Juttner(u, v, w, thermal_spread, lorentz_factor, -1);
                                }
                                else
                                {
                                    //? Non relativistic
                                    if(x < Lx_half)
                                        sample_maxwellian(u, v, w, uth, vth, wth, u0, v0, w0);
                                    else  
                                        sample_maxwellian(u, v, w, uth, vth, wth, -u0, v0, w0);
                                }         
                                
                                create_new_particle(u, v, w, q, x, y, z);
                            }

        fixPosition();
    }

    //! If col->getRestart_status() == 0 or 1, particle restart data is automatically read from the restart#.hdf files
}

//? ========================================================================== ?//

//? Quasi-1D ion-electron shock (Relativistic and Non relativistic)
void EMfields3D::initShock1D() 
{
    const Grid *grid = &get_grid();
    const Collective *col = &get_col();
    const VirtualTopology3D *vct = &get_vct();

    double v0  = col->getU0(1);
    double thb = col->getUth(1);

    //! New initial setup
    if (col->getRestart_status() == 0)
    {
        //! Initial setup (NOT RESTART)
        if (vct->getCartesian_rank() == 0) 
        {
            cout << "-------------------------------------------------------" << endl;
            cout << "Initialise quasi-1D double periodic ion-electron shock " << endl;
            cout << "-------------------------------------------------------" << endl;
            cout << "Background ion sigma                                 = " << (B0x*B0x+B0y*B0y+B0z*B0z)/sqrt(FourPI*rhoINIT[1]) << endl;
            if (col->getRelativistic())
            {
                cout << "Background theta_i                               = " << thb << endl;
                cout << "Background bulk velocity                         = " << v0 << endl;
            }
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
    
    else if (col->getRestart_status() == 1 || col->getRestart_status() == 2)
    {
        //! Read data from restart files
        init_fields_restart();
    }

    else
    {
        if (vct->getCartesian_rank() == 0)
        {
            cout << "Incorrect restart status!" << endl;
            cout << "   restart_status = 0 ---> NO RESTART!" << endl;
            cout << "   restart_status = 1 or 2 ---> RESTART!" << endl;
        }
        abort();
    }
}

//? ========================================================================== ?//