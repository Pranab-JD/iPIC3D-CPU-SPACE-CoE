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

//? Kelvin--Helmholtz Instability (Finite Larmor Radius (FLR); Cerri 2013, https://doi.org/10.1063/1.4828981)
void Particles3D::maxwellian_KHI_FLR(Field* EMf)
{
    //* Custom input parameters
    const double velocity_shear         = input_param[0];       //* Initial velocity shear
    const double perturbation           = input_param[1];       //* Amplitude of initial perturbation
    const double gamma_electrons        = input_param[2];       //* Gamma for isothermal electrons (FLR corrections)
    const double gamma_ions_perp        = input_param[3];       //* Gamma (perpendicular) for ions (FLR corrections)
    const double gamma_ions_parallel    = input_param[4];       //* Gamma (parallel) for ions (FLR corrections)
    const double s3                     = input_param[5];       //* +/-1 (Here -1 : Ux(y) or 1 : Uy(x)) (FLR corrections)
    const double delta                  = input_param[6];       //* Thickness of shear layer (FLR corrections)

    //! New initial setup
    if (col->getRestart_status() == 0)
    {
        //* Initial incompressible velocity perturbation on the first modes
        double TwoPI = 8*atan(1.0);
        double kx_pert = TwoPI/Lx;
        int nbpert = 5;                                             //* Number of initial perturbation modes
        array2_double phase(nbpert, 2);

        double Vthi = col->getUth(1);                               //* Ion thermal velocity (supposed isotropic far from velocity shear layer)
        double qomi = col->getQOM(1);                               //* Ion charge to mass ratio
        double Vthe = col->getUth(0);                               //* Electron thermal velocity (supposed isotropic far from velocity shear layer)
        double qome = col->getQOM(0);                               //* Electron charge to mass ratio
        double TeTi = -qomi/qome * (Vthe/Vthi) * (Vthe/Vthi);       //* Electron to ion temperature ratio (computed from input file parameters)
    
        //* For FLR corrections
        double B0x = col->getB0x(); double B0y = col->getB0y(); double B0z = col->getB0z();
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

        //* Initialise random generator with different seed on different processor
        srand (vct->getCartesian_rank()+1+ns);

        //* Initialise phase for initial random noise
        for (int iipert=0; iipert < 2; iipert++)
            for (int ipert=0; ipert < nbpert; ipert++)
                phase[ipert][iipert] = 2.0*M_PI*(0.5*ipert/nbpert+0.5*iipert);

        //* Constant factor (to be multiplied to charge)
        const double q_factor = (qom / fabs(qom)) * grid->getVOL() / npcel;

        for (int i = 1; i < grid->getNXC() - 1; i++)
            for (int j = 1; j < grid->getNYC() - 1; j++)
                for (int k = 1; k < grid->getNZC() - 1; k++)
                {
                    const double q = q_factor * fabs(EMf->getRHOcs(i, j, k, ns));

                    for (int ii = 0; ii < npcelx; ii++)
                        for (int jj = 0; jj < npcely; jj++)
                            for (int kk = 0; kk < npcelz; kk++)
                            {
                                //* For ion FLR corrections
                                double ay  = 1.0/pow((cosh((grid->getYC(i,j,k)-0.25*Ly)/delta)), 2.0) - 1.0/pow((cosh((grid->getYC(i,j,k)-0.75*Ly)/delta)), 2.0);
                                double finf = 1.0/(1.0 - Cinf*ay);

                                const double x = (ii + .5) * (dx / npcelx) + grid->getXN(i, j, k);
                                const double y = (jj + .5) * (dy / npcely) + grid->getYN(i, j, k);
                                const double z = (kk + .5) * (dz / npcelz) + grid->getZN(i, j, k);

                                //* Thermal velocities
                                double vthperp, vthpar, vthx, vthy, power;
                                
                                //? Thermal velocity (assumed isotropic in input - only uth is used!!!)
                                if (qom < 0.0) 
                                {   
                                    //! Electrons
                                    vthx = uth; vthy = uth; vthpar = uth;
                                }
                                else 
                                { 
                                    //! Ions
                                    power   = (gamma_ions_perp-1.0)/(2.0*gamma_ions_perp);
                                    vthperp = uth * pow(finf, power);
                                    vthx    = vthperp * sqrt(1.0+s3*0.5*velocity_shear/(Omega_ci*delta)*ay);        // ion FLR along x
                                    vthy    = vthperp * sqrt(1.0-s3*0.5*velocity_shear/(Omega_ci*delta)*ay);        // ion FLR along y
                                    power   = (gamma_ions_parallel-1.0)/(2.0*gamma_ions_perp); 
                                    vthpar  = uth * pow(finf, power);                                               // ion FLR along z
                                }
                                
                                double u = c, v = c, w = c;

                                while ((fabs(u)>=c) | (fabs(v)>=c) | (fabs(w)>=c))
                                    sample_maxwellian(u, v, w, vthx, vthy, vthpar, 0, 0, 0);
                
                                //* Add drift velocity
                                double udrift = velocity_shear * (tanh((y-0.25*Ly)/delta) - tanh((y-0.75*Ly)/delta)-1.0);   //* X velocity drift (identical for electrons and ions)
                                u += udrift;

                                //* Add initial velocity perturbation at y = Ly/4
                                double u_pert = 0.0, v_pert = 0.0;
                                for (int ipert = 1; ipert < (nbpert+1); ipert++)
                                {
                                    u_pert += cos(ipert*kx_pert*x+phase[ipert-1][0]);
                                    v_pert += (ipert*kx_pert)*sin(ipert*kx_pert*x+phase[ipert-1][0]);
                                }
                                
                                double fy_pert = perturbation * exp( - (y-0.25*Ly)*(y-0.25*Ly) / (delta*delta) );
                                double dyfy_pert = -2.0*(y-0.25*Ly)/(delta*delta)*fy_pert;
                                u += dyfy_pert*u_pert;
                                v += fy_pert*v_pert;
                                
                                //* Add initial velocity perturbation at y = 3*Ly/4
                                u_pert = 0.0; v_pert = 0.0;
                                for (int ipert = 1; ipert < (nbpert+1); ipert++)
                                {
                                    u_pert += cos(ipert*kx_pert*x+phase[ipert-1][1]);
                                    v_pert += (ipert*kx_pert)*sin(ipert*kx_pert*x+phase[ipert-1][1]);
                                }
                                
                                fy_pert = perturbation * exp( - (y-0.75*Ly)*(y-0.75*Ly) / (delta*delta) );
                                dyfy_pert = -2.0*(y-0.75*Ly)/(delta*delta)*fy_pert;
                                u += dyfy_pert*u_pert;
                                v += fy_pert*v_pert;

                                if (u != u) 
                                    MPI_Abort(MPI_COMM_WORLD, -1); 

                                create_new_particle(u, v, w, q, x, y, z);
                            }
                }

        fixPosition();
    }

    //! If col->getRestart_status() == 0 or 1, particle restart data is automatically read from the restart#.hdf files
}

//? ========================================================================== ?//

//* Initialise fields for shear velocity in fluid finite Larmor radius (FLR) equilibrium (Cerri et al. 2013)
//* The charge is set to 1/(4 pi) in order to satisfy the omega_pi = 1. The 2 species have same charge density to guarantee plasma neutrality
void EMfields3D::init_KHI_FLR()
{
    const Grid *grid = &get_grid();
    const Collective *col = &get_col();
    const VirtualTopology3D *vct = &get_vct();

    //* Custom input parameters
    const double velocity_shear         = input_param[0];       //* Initial velocity shear
    const double perturbation           = input_param[1];       //* Amplitude of initial perturbation
    const double gamma_electrons        = input_param[2];       //* Gamma for isothermal electrons (FLR corrections)
    const double gamma_ions_perp        = input_param[3];       //* Gamma (perpendicular) for ions (FLR corrections)
    const double gamma_ions_parallel    = input_param[4];       //* Gamma (parallel) for ions (FLR corrections)
    const double s3                     = input_param[5];       //* +/-1 (Here -1 : Ux(y) or 1 : Uy(x)) (FLR corrections)
    const double delta                  = input_param[6];       //* Thickness of shear layer (FLR corrections)

    //! New initial setup
    if (restart_status == 0)
    {
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

        if (col->getRestart_status() == 0)
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