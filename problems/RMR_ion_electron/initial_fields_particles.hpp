//* This file needs to be included in main/iPIC3Dlib.cpp

#include "../include.hpp"

#ifndef NO_HDF5
#endif

using std::cout;
using std::endl;
using namespace iPic3D;

//? ========================================================================== ?//

//? Relativistic double Harris for ion-electron plasma: Maxwellian background, drifting particles in the sheets
void Particles3D::Relativistic_Double_Harris_ion_electron(Field * EMf) 
{
	//* Initialise random generator with different seed on different processor
	long long seed = (vct->getCartesian_rank() + 1)*20 + ns;
    srand(seed);
    srand48(seed);

    assert_eq(_pcls.size(), 0);

    //* Custom input parameters for relativistic reconnection
    const double sigma                  = input_param[0];       //* Magnetisation parameter
    const double eta                    = input_param[1];       //* Ratio of current sheet density to upstream density (this is "alpha" in Fabio's paper; Eqs 52 and 53)
    const double delta_CS               = input_param[2];       //* Half-thickness of current sheet (free parameter)
    const double perturbation           = input_param[3];       //* Amplitude of initial perturbation
    const double guide_field_ratio      = input_param[4];       //* Ratio of guide field to in-plane magnetic field
    
    //* Background (BG) or upstream particles
    double thermal_spread_BG_electrons  = col->getUth(0);                           //* Thermal spread of electrons
    double thermal_spread_BG_ions       = col->getUth(1);                           //* Thermal spread of ions
    double rho_BG                       = col->getRHOinit(ns)/(4.0*M_PI);           //* Density (rho_BG = n * mc^2)
    double B_BG                         = sqrt(sigma*4.0*M_PI*rho_BG);              //* sigma = B^2/(4*pi*rho_electrons)
    
    //* Current sheet (CS) particles
    double rho_CS                       = eta*rho_BG;                                           //* Density (rho_CS = eta * n * mc^2)
    double drift_velocity               = B_BG/(8.0*M_PI*rho_CS*delta_CS/c);                    //* v = B*c/(8 * pi * rho_CS * delta_CS); Eq 52
    double lorentz_factor_CS            = 1.0/sqrt(1.0 - drift_velocity*drift_velocity);        //* Lorentz factor of the relativistic drifting particles
    double thermal_spread_CS_ions       = B_BG*B_BG*lorentz_factor_CS/(16.0*M_PI*rho_CS);       //* Thermal spread of ions (B^2 * Gamma/(16 * pi * eta * n * mc^2)); Eq 53
    double thermal_spread_CS_electrons  = thermal_spread_CS_ions * fabs(col->getQOM(0));        //* Thermal spread of electrons (Ratio of thermal spread = mass ratio)
    
    //* Additional params needed for setting up a current sheet
    double y_half           = Ly/2.0;
    double y_quarter        = Ly/4.0;
    double y_three_quarters = 3.0*y_quarter;

    const double q_factor = (qom/fabs(qom)) * grid->getVOL()/npcel;
    
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

                            //* Velocities and charges of particles
                            double u, v, w, q;
                        
                            //* Distinguish between background and drifting species
                            if (ns < 2) 
                            {
                                //? Background species (these are the particles that get accelerated)
                                q = q_factor * rho_BG;
                                
                                //* Velocity of relativistic nondrifting Maxwellian
								if (qom < 0.0) 
                                    sample_Maxwell_Juttner(u, v, w, thermal_spread_BG_electrons, 1.0, 0);
								else        
                                    sample_Maxwell_Juttner(u, v, w, thermal_spread_BG_ions, 1.0, 0);

                                
                            }
                            else 
                            {
                                //? Current sheet species (necessary to initialise a current sheet)
                                double fs;

                                if (y < y_half)   
                                    fs = sech_square((y - y_quarter)/delta_CS);
                                else              
                                    fs = sech_square((y - y_three_quarters)/delta_CS);
                        
                                //* Skip the particle if its weight is too small
                                if (fabs(fs) < 1.e-8) continue;

                                q = q_factor * rho_CS * fs;

                                //* Velocity of relativistic drifting (along the Z-axis) Maxwellian
                                if (qom < 0.0) 
                                    sample_Maxwell_Juttner(u, v, w, thermal_spread_CS_electrons, lorentz_factor_CS, -3);    //* Negative charges (e.g., electrons)
                                else
                                    sample_Maxwell_Juttner(u, v, w, thermal_spread_CS_ions, lorentz_factor_CS, 3);          //* Positive charges (e.g., ions)
                                
                                //* Flip sign of drift velocity for particles in the second layer
                                if (y > y_half)
                                {
                                    u = -u; 
                                    v = -v; 
                                    w = -w;
                                }
                            }

                            create_new_particle(u, v, w, q, x, y, z);
                        }

	fixPosition();
}

//? ========================================================================== ?//

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
        init_fields_restart();
}

//? ========================================================================== ?//