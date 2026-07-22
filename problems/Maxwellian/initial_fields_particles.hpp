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

//! Particles are initialised with a uniform spatial distribution and a Maxellian velocity distribution

//? ========================================================================== ?//

//* Default electric and magnetic field configurations
void EMfields3D::init_Maxwellian()
{
    const Grid *grid = &get_grid();
    const Collective *col = &get_col();
    const VirtualTopology3D *vct = &get_vct();

    //! New initial setup
    if (col->getRestart_status() == 0)
    {
        if (vct->getCartesian_rank() == 0) 
            cout << "Default field initialisation; initial magnetic field components (Bx, By, Bz) = " << "(" << B0x << ", " << B0y << ", " << B0z << ")" << endl;

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