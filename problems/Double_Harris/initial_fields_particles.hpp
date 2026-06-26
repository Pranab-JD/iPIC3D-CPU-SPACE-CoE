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
void Particles3D::maxwellian_Double_Harris(Field * EMf)
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

//* Initialise double Harris sheets for magnetic reconnection
void EMfields3D::init_Double_Harris()
{
    const Grid *grid = &get_grid();
    const Collective *col = &get_col();
    const VirtualTopology3D *vct = &get_vct();

    //* Custom input parameters
    const double perturbation               = input_param[0];       //* Amplitude of initial perturbation
    const double delta                      = input_param[1];       //* Half-thickness of current sheet

    //! New initial setup
    if (restart_status == 0)
    {
        if (vct->getCartesian_rank() ==0)
        {
            cout << "------------------------------------------" << endl;
            cout << "     Initialising Double Harris Sheet     " << endl;
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
    
    else if (restart_status == 1 || restart_status == 2)
    {
        //! Read data from restart files
        init_fields_restart();
    }

    else
    {
        if (vct->getCartesian_rank() == 0)
        {
            cout << "Incorrect restart status!" << endl;
            cout << "restart_status = 0 ---> NO RESTART!" << endl;
            cout << "restart_status = 1 ---> RESTART! SaveDirName and RestartDirName are different" << endl;
            cout << "restart_status = 1 ---> RESTART! SaveDirName and RestartDirName are the same" << endl;
        }
        abort();
    }
}

//? ========================================================================== ?//