//* This file needs to be included in main/iPIC3Dlib.cpp

#include "../include.hpp"

#ifndef NO_HDF5
#endif

using std::cout;
using std::endl;
using namespace iPic3D;

//? ========================================================================== ?//

//? Quasi-1D double periodic ion-electron shock driven by a piston (Relativistic and Non relativistic)
void Particles3D::Shock1D_DoublePiston(Field * EMf) 
{
    //* Initialise random generator with different seed on different processor
	long long seed = (vct->getCartesian_rank() + 1)*20 + ns;
    srand(seed);
    srand48(seed);

    assert_eq(_pcls.size(), 0);
  
    double thermal_velocity = col->getUth(ns);
    const double Lx_half = Lx/2.0;
    const double dx_one_half = 1.5*dx; 

    const double q = (qom / fabs(qom)) * grid->getVOL() / npcel * col->getRHOinit(ns)/(4.0*M_PI);

    for (int i = 1; i < grid->getNXC() - 1; i++)
        for (int j = 1; j < grid->getNYC() - 1; j++)
            for (int k = 1; k < grid->getNZC() - 1; k++) 
                for (int ii = 0; ii < npcelx; ii++) 
                {
                    //* Skip first cell near Lx/2 so that the sudden appearance of a 
                    //* static piston doesn't cause particles to be shot away
                    double xp = (ii + .5) * (dx / npcelx) + grid->getXN(i, j, k);
                    if (fabs(xp - Lx_half) < 1.5*dx) continue;
  
                    for (int jj = 0; jj < npcely; jj++)
                        for (int kk = 0; kk < npcelz; kk++) 
                        {
                            const double x = (ii + .5) * (dx / npcelx) + grid->getXN(i, j, k);
                            const double y = (jj + .5) * (dy / npcely) + grid->getYN(i, j, k);
                            const double z = (kk + .5) * (dz / npcelz) + grid->getZN(i, j, k);

                            double u, v, w;
                            
                            if (col->getRelativistic()) 
                            {
                                //? Relativistic (velocity of relativistic nondrifting Maxwellian)
                                sample_Maxwell_Juttner(u, v, w, thermal_velocity, 1.0, 0);
                            }
                            else 
                            {
                                //? Non relativistic
                                sample_maxwellian(u, v, w, uth, vth, wth, u0, v0, w0);
                            }         
                            
                            create_new_particle(u, v, w, q, x, y, z);
                        }
                }

    fixPosition();
}

//? ========================================================================== ?//