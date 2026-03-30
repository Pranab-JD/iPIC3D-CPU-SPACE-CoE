//* This file needs to be included in main/iPIC3Dlib.cpp

#include "../include.hpp"

#ifndef NO_HDF5
#endif

using std::cout;
using std::endl;
using namespace iPic3D;

//? ========================================================================== ?//

//? Initialise unifrom distribution of particles with relativistic Maxellian random velocity
void Particles3D::Maxwell_Juttner(Field * EMf) 
{
	//* Initialise random generator with different seed on different processor
	srand(vct->getCartesian_rank() + 2 + ns);

    assert_eq(_pcls.size(), 0);    

    double thermal_spread = uth;                                //* Thermal spread (isotropic along X, Y, Z)
	double lorentz_factor_x = u0;                               //* Lorentz factor (X)
	double lorentz_factor_y = v0;                               //* Lorentz factor (Y)
	double lorentz_factor_z = w0;                               //* Lorentz factor (Z)
	double lorentz_factor; int drift_direction;
	
    if (fabs(lorentz_factor_x) > 1.0) 
    {
		drift_direction = int(fabs(lorentz_factor_x)/lorentz_factor_x) * 1;
		lorentz_factor = fabs(lorentz_factor_x);
	}
	else if (fabs(lorentz_factor_y) > 1.0) 
    {
		drift_direction = int(fabs(lorentz_factor_y)/lorentz_factor_y) * 2;
		lorentz_factor = fabs(lorentz_factor_y);
	}
	else if (fabs(lorentz_factor_z) > 1.0) 
    {
		drift_direction = int(fabs(lorentz_factor_z)/lorentz_factor_z) * 3;
		lorentz_factor = fabs(lorentz_factor_z);
	}
	else 
    {
        drift_direction = 0;
		lorentz_factor = 1.0;
	}

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

                            double u, v, w;
                            sample_Maxwell_Juttner(u, v, w, thermal_spread, lorentz_factor, drift_direction);
                            
                            create_new_particle(u, v, w, q, x, y, z);
                        }

	fixPosition();
}

//? ========================================================================== ?//

//* Default electric and magnetic field configurations
void EMfields3D::init_Maxwell_Juttner()
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
        
        #endif // NO_HDF5
    }
}

//? ========================================================================== ?//