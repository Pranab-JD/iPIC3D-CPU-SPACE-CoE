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

//? Initialise uniform distribution of particles with a Maxellian velocity distribution
void Particles3D::maxwellian_Double_Harris_Hump(Field * EMf)
{
    //! New initial setup
    if (col->getRestart_status() == 0)
    {
        //* Initialise random generator with different seed on different processor
        long long seed = (vct->getCartesian_rank() + 1)*20 + ns;
        srand(seed);
        srand48(seed);

        assert_eq(_pcls.size(), 0);
        double prob, theta;

        const double q_factor =  (qom / fabs(qom)) * grid->getVOL() / npcel;

        for (int i = 1; i < grid->getNXC() - 1; i++)
            for (int j = 1; j < grid->getNYC() - 1; j++)
                for (int k = 1; k < grid->getNZC() - 1; k++)
                {
                    const double q = q_factor * fabs(EMf->getRHOcs(i, j, k, ns));
                    
                    for (int ii = 0; ii < npcelx; ii++)
                        for (int jj = 0; jj < npcely; jj++)
                            for (int kk = 0; kk < npcelz; kk++)
                            {
                                double global_y = grid->getYN(i, j, k) + grid->getDY();
                                double shaper_z = -tanh((global_y - Ly/2)/0.0001);

                                const double x = (ii + .5) * (dx / npcelx) + grid->getXN(i, j, k);
                                const double y = (jj + .5) * (dy / npcely) + grid->getYN(i, j, k);
                                const double z = (kk + .5) * (dz / npcelz) + grid->getZN(i, j, k);
                                
                                double u, v, w;
                                sample_maxwellian(u, v, w, uth, vth, wth, u0, v0, w0*shaper_z);
                                create_new_particle(u,v,w,q,x,y,z);
                            }
                }
        
        fixPosition();
    }

    //! If col->getRestart_status() == 0 or 1, particle restart data is automatically read from the restart#.hdf files
}

//? ========================================================================== ?//

void EMfields3D::init_Double_Harris_Hump()
{
    const Grid *grid = &get_grid();
    const Collective *col = &get_col();
    const VirtualTopology3D *vct = &get_vct();

    //* Custom input parameters
    const double perturbation               = input_param[0];       //* Amplitude of initial perturbation (localised in X)
    const double delta                      = input_param[1];       //* Half-thickness of current sheet

    const double delta_x = 8.0 * delta;
    const double delta_y = 4.0 * delta;
    
    //! New initial setup
    if (col->getRestart_status() == 0) 
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