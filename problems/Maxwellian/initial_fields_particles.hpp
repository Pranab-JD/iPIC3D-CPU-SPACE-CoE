//* This file needs to be included in main/iPIC3Dlib.cpp

#include "../include.hpp"

#ifndef NO_HDF5
#endif

using std::cout;
using std::endl;
using namespace iPic3D;

//? ========================================================================== ?//

//? Initialise uniform distribution of particles with a Maxellian velocity distribution
void Particles3D::maxwellian(Field * EMf)
{
    //* Initialise random generator with different seed on different processor
    long long seed = (vct->getCartesian_rank() + 1)*20 + ns;
    srand(seed);
    srand48(seed);

    assert_eq(_pcls.size(), 0);

    const double q_sgn = (qom / fabs(qom));
    const double q_factor =  q_sgn * grid->getVOL() / npcel;

    long long counter = 0;
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
                            counter++;
                        }
            }

    fixPosition();
}

//? ========================================================================== ?//

//* Default electric and magnetic field configurations
void EMfields3D::init_Maxwellian()
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
        init_fields_restart();
}

//? ========================================================================== ?//