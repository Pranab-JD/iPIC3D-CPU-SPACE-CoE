/* iPIC3D was originally developed by Stefano Markidis and Giovanni Lapenta. 
 * This release was contributed by Alec Johnson and Ivy Bo Peng.
 * Publications that use results from iPIC3D need to properly cite  
 * 'S. Markidis, G. Lapenta, and Rizwan-uddin. "Multi-scale simulations of 
 * plasma with iPIC3D." Mathematics and Computers in Simulation 80.7 (2010): 1509-1519.'
 *
 *        Copyright 2015 KTH Royal Institute of Technology
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at 
 *
 *         http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*******************************************************************************************
  Particles3D.cpp  -  Class for particles of the same species, in a 3D space and 3component velocity
  -------------------
developers: Stefano Markidis, Giovanni Lapenta
 ********************************************************************************************/


#include <mpi.h>
#include <iostream>
#include <math.h>
#include <limits.h>
#include "asserts.h"
#include "VCtopology3D.h"
#include "Collective.h"
#include "Basic.h"
#include "Grid3DCU.h"
#include "Field.h"
#include "MPIdata.h"
#include "ipicdefs.h"
#include "TimeTasks.h"
#include "parallel.h"
#include "Particles3D.h"

#include "mic_particles.h"
#include "debug.h"
#include <complex>
#include <iomanip>
#include <fstream>
#include "../LeXInt_Timer.hpp"
#include <cstdlib>

using std::cout;
using std::cerr;
using std::endl;

#define min(a,b) (((a)<(b))?(a):(b));
#define max(a,b) (((a)>(b))?(a):(b));
#define MIN_VAL   1E-16
// particles processed together
#define P_SAME_TIME 2

// if true then mover cannot move particle 
// more than one processor subdomain
//
static bool cap_velocity(){return false;}

//* sech^2(x) up to arbitrary precision
double sech_square(double x) 
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

/**
 * 
 * Class for particles of the same species
 * @date Fri Jun 4 2009
 * @author Stefano Markidis, Giovanni Lapenta
 * @version 2.0
 *
 */

//! ============================================================================= !//

//! Initial particle distributions (Non Relativistic) !//

//? Particles are uniformly distributed with zero velocity
void Particles3D::uniform_background(Field * EMf)
{
    for (int i = 1; i < grid->getNXC() - 1; i++)
        for (int j = 1; j < grid->getNYC() - 1; j++)
            for (int k = 1; k < grid->getNZC() - 1; k++)
                for (int ii = 0; ii < npcelx; ii++)
                    for (int jj = 0; jj < npcely; jj++)
                        for (int kk = 0; kk < npcelz; kk++)
                        {
                            double x = (ii + .5) * (dx / npcelx) + grid->getXN(i, j, k);
                            double y = (jj + .5) * (dy / npcely) + grid->getYN(i, j, k);
                            double z = (kk + .5) * (dz / npcelz) + grid->getZN(i, j, k);
                            double u = uth + u0;
                            double v = vth + v0;
                            double w = wth + w0;
                            double q = (qom / fabs(qom)) * (EMf->getRHOcs(i, j, k, ns) / npcel) * (1.0 / grid->getInvVOL());
                            _pcls.push_back(SpeciesParticle(u,v,w,q,x,y,z,0));
                        }

    fixPosition();
    
}

//? Initialize particles with a constant velocity along "dim" direction
void Particles3D::constantVelocity(double vel, int dim, Field * EMf) 
{
    switch (dim) 
    {
        case 0:
        for (int i = 0; i < getNOP(); i++)
        {
            setU(i,vel);
            setV(i,0.);
            setW(i,0.);
        }
        break;
        case 1:
        for (int i = 0; i < getNOP(); i++)
        {
            setU(i,0.);
            setV(i,vel);
            setW(i,0.);
        }
        break;
        case 2:
        for (int i = 0; i < getNOP(); i++)
        {
            setU(i,0.);
            setV(i,0.);
            setW(i,vel);
        }
        break;

    }
}

#ifdef BATSRUS
/** Maxellian random velocity and uniform spatial distribution */
void Particles3D::MaxwellianFromFluid(Field* EMf,Collective *col, int is)
{
    //* Constuctiong the distribution function from a Fluid model

    // loop over grid cells and set position, velociy and charge of all particles indexed by counter
    // there are multiple (27 or so) particles per grid cell.
    int i,j,k,counter=0;
    for (i=1; i< grid->getNXC()-1;i++)
        for (j=1; j< grid->getNYC()-1;j++)
            for (k=1; k< grid->getNZC()-1;k++)
                MaxwellianFromFluidCell(col,is, i,j,k,counter,x,y,z,q,u,v,w,ParticleID);
}

void Particles3D::MaxwellianFromFluidCell(Collective *col, int is, int i, int j, int k, int &ip, double *x, double *y, double *z, double *q, double *vx, double *vy, double *vz, longid* ParticleID)
{
    /*
    * grid           : local grid object (in)
    * col            : collective (global) object (in)
    * is             : species index (in)
    * i,j,k          : grid cell index on proc (in)
    * ip             : particle number counter (inout)
    * x,y,z          : particle position (out)
    * q              : particle charge (out)
    * vx,vy,vz       : particle velocity (out)
    * ParticleID     : particle tracking ID (out)
    */

    // loop over particles inside grid cell i,j,k
    for (int ii=0; ii < npcelx; ii++)
        for (int jj=0; jj < npcely; jj++)
            for (int kk=0; kk < npcelz; kk++){
                // Assign particle positions: uniformly spaced. x_cellnode + dx_particle*(0.5+index_particle)
                fetchX(ip) = (ii + .5)*(dx/npcelx) + grid->getXN(i,j,k);
                fetchY(ip) = (jj + .5)*(dy/npcely) + grid->getYN(i,j,k);
                fetchZ(ip) = (kk + .5)*(dz/npcelz) + grid->getZN(i,j,k);
                // q = charge
                fetchQ(ip) =  (qom/fabs(qom))*(col->getFluidRhoCenter(i,j,k,is)/npcel)*(1.0/grid->getInvVOL());
                // u = X velocity
                sample_maxwellian(
                fetchU(ip),fetchV(ip),fetchW(ip),
                col->getFluidUthx(i,j,k,is),
                col->getFluidVthx(i,j,k,is),
                col->getFluidWthx(i,j,k,is),
                col->getFluidUx(i,j,k,is),
                col->getFluidVx(i,j,k,is),
                col->getFluidWx(i,j,k,is));
                ip++ ;
        }
}
#endif

//? Initialise unifrom distribution of particles with a Maxellian velocity distribution
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

    if(0)
    {
        dprintf("number of particles of species %d: %d", ns, getNOP());
        const int num_ids = 1;
        longid id_list[num_ids] = {0};
        print_pcls(_pcls,ns,id_list, num_ids);
    }
}

/** Maxellian velocity from currents and uniform spatial distribution */
void Particles3D::maxwellianNullPoints(Field * EMf)
{
	//* Initialise random generator with different seed on different processor
	srand(vct->getCartesian_rank()+2);

	const double q_sgn = (qom / fabs(qom));
	const double q_factor =  q_sgn * grid->getVOL() / npcel;

	for (int i=1; i< grid->getNXC()-1;i++)
	for (int j=1; j< grid->getNYC()-1;j++)
	for (int k=1; k< grid->getNZC()-1;k++){
		const double q = q_factor * EMf->getRHOcs(i, j, k, ns);

		// determine the drift velocity from current X
		u0 = EMf->getJxs(i,j,k,ns)/EMf->getRHOns(i,j,k,ns);
		if (u0 > c){
			cout << "DRIFT VELOCITY x > c : B init field too high!" << endl;
			MPI_Abort(MPI_COMM_WORLD,2);
		}
		// determine the drift velocity from current Y
		v0 = EMf->getJys(i,j,k,ns)/EMf->getRHOns(i,j,k,ns);
		if (v0 > c){
			cout << "DRIFT VELOCITY y > c : B init field too high!" << endl;
			MPI_Abort(MPI_COMM_WORLD,2);
		}
		// determine the drift velocity from current Z
		w0 = EMf->getJzs(i,j,k,ns)/EMf->getRHOns(i,j,k,ns);
		if (w0 > c){
			cout << "DRIFT VELOCITY z > c : B init field too high!" << endl;
			MPI_Abort(MPI_COMM_WORLD,2);
		}
		for (int ii=0; ii < npcelx; ii++)
		for (int jj=0; jj < npcely; jj++)
		for (int kk=0; kk < npcelz; kk++){
			double u,v,w;
			sample_maxwellian(u, v, w, uth, vth, wth, u0, v0, w0);

			const double x = (ii + .5)*(dx/npcelx) + grid->getXN(i,j,k);
			const double y = (jj + .5)*(dy/npcely) + grid->getYN(i,j,k);
			const double z = (kk + .5)*(dz/npcelz) + grid->getZN(i,j,k);

			create_new_particle(u,v,w,q,x,y,z);
		}
	}
}

/** Maxellian random velocity and uniform spatial distribution - invert w0 for the upper current sheet */
void Particles3D::maxwellian_Double_Harris(Field * EMf)
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
                            
                            if (fabs(q) < 1.e-8) 
                                continue;                   //* skip this particle if weight is too small
                            
                            double u, v, w;
                            sample_maxwellian(u, v, w, uth, vth, wth, u0, v0, w0*shaper_z);
                            create_new_particle(u,v,w,q,x,y,z);
                        }
            }
    
    fixPosition();
}

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

/** pitch_angle_energy initialization (Assume B on z only) for test particles */
void Particles3D::pitch_angle_energy(Field * EMf) 
{
    //* Initialise random generator with different seed on different processor
    srand(vct->getCartesian_rank() + 3 + ns);
    assert_eq(_pcls.size(),0);

    double p0, pperp0, gyro_phase;

    const double q_factor =  (qom / fabs(qom)) * grid->getVOL() / npcel;

    long long counter=0;

    for (int i=1; i< grid->getNXC()-1;i++)
        for (int j=1; j< grid->getNYC()-1;j++)
            for (int k=1; k< grid->getNZC()-1;k++){

            	// q = charge following electron (species 0)
            	const double q = q_factor * EMf->getRHOcs(i, j, k, 0);

                for (int ii=0; ii < npcelx; ii++)
                    for (int jj=0; jj < npcely; jj++)
                        for (int kk=0; kk < npcelz; kk++){
                        	const double x= (ii + .5)*(dx/npcelx) + grid->getXN(i,j,k);
                        	const double y= (jj + .5)*(dy/npcely) + grid->getYN(i,j,k);
                        	const double z= (kk + .5)*(dz/npcelz) + grid->getZN(i,j,k);

                            // velocity - assumes B is along z
                            p0=sqrt((energy+1)*(energy+1)-1);
                            const double w =p0*cos(pitch_angle);
                            pperp0=p0*sin(pitch_angle);
                            gyro_phase = 2*M_PI* rand()/(double)RAND_MAX;
                            const double u=pperp0*cos(gyro_phase);
                            const double v=pperp0*sin(gyro_phase);
                            counter++ ;

                            create_new_particle(u,v,w,q,x,y,z);
                        }
            }
    const int num_ids = 1;
    longid id_list[num_ids] = {0};
    if (vct->getCartesian_rank() == 0){
    	cout << "------------------------------------------" << endl;
        cout << "Initialize Test Particle "<< ns << " with pitch angle "<< pitch_angle << ", energy " << energy << ", qom " << qom << ", npcel "<< counter<< endl;
        cout << "------------------------------------------" << endl;
    }
}

/** Force Free initialization (JxB=0) for particles */
void Particles3D::force_free(Field * EMf)
{
    eprintf("this function was not properly implemented and needs to be revised.");

    #if 0
    /* initialize random generator */
    srand(vct->getCartesian_rank() + 1 + ns);
    
    for (int i = 1; i < grid->getNXC() - 1; i++)
        for (int j = 1; j < grid->getNYC() - 1; j++)
            for (int k = 1; k < grid->getNZC() - 1; k++)
                for (int ii = 0; ii < npcelx; ii++)
                    for (int jj = 0; jj < npcely; jj++)
                        for (int kk = 0; kk < npcelz; kk++)
                        {
                        double x = (ii + .5) * (dx / npcelx) + grid->getXN(i, j, k);
                        double y = (jj + .5) * (dy / npcely) + grid->getYN(i, j, k);
                        double z = (kk + .5) * (dz / npcelz) + grid->getZN(i, j, k);
                        // q = charge
                        double q = (qom / fabs(qom)) * (EMf->getRHOcs(i, j, k, ns) / npcel) * (1.0 / invVOL);
                        double shaperx = tanh((y - Ly / 2) / delta) / cosh((y - Ly / 2) / delta) / delta;
                        double shaperz = 1.0 / (cosh((y - Ly / 2) / delta) * cosh((y - Ly / 2) / delta)) / delta;
                        eprintf("shapery needs to be initialized.");
                        eprintf("flvx etc. need to be initialized.");
                        double shapery;
                        // new drift velocity to satisfy JxB=0
                        const double flvx = u0 * flvx * shaperx;
                        const double flvz = w0 * flvz * shaperz;
                        const double flvy = v0 * flvy * shapery;
                        double u = c;
                        double v = c;
                        double w = c;
                        while ((fabs(u) >= c) || (fabs(v) >= c) || (fabs(w) >= c))
                        {
                            sample_maxwellian(
                            u, v, w,
                            uth, vth, wth,
                            flvx, flvy, flvz);
                        }
                        create_new_particle(u,v,w,q,x,y,z);
                        }
    #endif
}

/**Add a periodic perturbation in J exp i(kx - \omega t); deltaBoB is the ratio (Delta B / B0) **/
void Particles3D::AddPerturbationJ(double deltaBoB, double kx, double ky, double Bx_mod, double By_mod, double Bz_mod, double jx_mod, double jx_phase, double jy_mod, double jy_phase, double jz_mod, double jz_phase, double B0) {

  // rescaling of amplitudes according to deltaBoB //
  double alpha;
  alpha = deltaBoB * B0 / sqrt(Bx_mod * Bx_mod + By_mod * By_mod + Bz_mod * Bz_mod);
  jx_mod *= alpha;
  jy_mod *= alpha;
  jz_mod *= alpha;
  for (int i = 0; i < getNOP(); i++) {
    fetchU(i) += jx_mod / q[i] / npcel / invVOL * cos(kx * x[i] + ky * y[i] + jx_phase);
    fetchV(i) += jy_mod / q[i] / npcel / invVOL * cos(kx * x[i] + ky * y[i] + jy_phase);
    fetchW(i) += jz_mod / q[i] / npcel / invVOL * cos(kx * x[i] + ky * y[i] + jz_phase);
  }
}


//! Initial particle distributions (Relativistic) !//

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

//? Relativistic double Harris for pair plasma: Maxwellian background, drifting particles in the sheets
void Particles3D::Relativistic_Double_Harris_pairs(Field * EMf) 
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
    double thermal_spread_BG            = col->getUth(0);                           //* Thermal spread
    double rho_BG                       = col->getRHOinit(ns)/(4.0*M_PI);           //* Density (rho_BG = n * mc^2)
    double B_BG                         = sqrt(sigma*4.0*M_PI*rho_BG*2.0);          //* sigma = B^2/(4*pi*rho_electron*rho_prositron)

    //* Current sheet (CS) particles
    double rho_CS                       = eta*rho_BG;                                            //* Density (rho_CS = eta * n * mc^2)
    double drift_velocity               = B_BG/(8.0*M_PI*rho_CS*delta_CS/c);                     //* v = B*c/(8 * pi * rho_CS * delta_CS); Eq 52
    double lorentz_factor_CS            = 1.0/sqrt(1.0 - drift_velocity*drift_velocity);         //* Lorentz factor of the relativistic drifting particles
    double thermal_spread_CS            = B_BG*B_BG*lorentz_factor_CS/(16.0*M_PI*rho_CS);        //* Thermal spread (B^2 * Gamma/(16 * pi * eta * n * mc^2)); Eq 53
  
    //* Additional params needed for setting up a current sheet
    double y_half           = Ly/2.0;
    double y_quarter        = Ly/4.0;
    double y_three_quarters = 3.0*y_quarter;

    const double q_factor = (qom / fabs(qom)) * grid->getVOL()/npcel;

	for (int i = 1; i < grid->getNXC() - 1; i++)
        for (int j = 1; j < grid->getNYC() - 1; j++)
            for (int k = 1; k < grid->getNZC() - 1; k++)
                for (int ii = 0; ii < npcelx; ii++)
                    for (int jj = 0; jj < npcely; jj++)
                        for (int kk = 0; kk < npcelz; kk++) 
                        {
                            double x = (ii + .5) * (dx / npcelx) + grid->getXN(i, j, k);
                            double y = (jj + .5) * (dy / npcely) + grid->getYN(i, j, k);
                            double z = (kk + .5) * (dz / npcelz) + grid->getZN(i, j, k);

                            //* Velocities and charges of particles
                            double u, v, w, q, fs;
                        
                            //* Distinguish between background and drifting species
                            if (ns < 2) 
                            {
                                //? Background species (these are the particles that get accelerated)
                                q = q_factor * rho_BG;

                                //* Velocity of relativistic nondrifting Maxwellian
                                sample_Maxwell_Juttner(u, v, w, thermal_spread_BG, 1.0, 0);
                            }
                            else 
                            {
                                //? Current sheet species (necessary to initialise a current sheet)
                                if (y < y_half)
                                    fs = sech_square((y - y_quarter)/delta_CS);
                                else              
                                    fs = sech_square((y - y_three_quarters)/delta_CS);
                        
                                //* Skip the particle if its weight is too small
                                if (fabs(fs) < 1.e-8) continue;

                                q = q_factor * rho_CS * fs;

                                //* Velocity of relativistic drifting (along the Z-axis) Maxwellian
                                if (qom < 0.0) 
                                    sample_Maxwell_Juttner(u, v, w, thermal_spread_CS, lorentz_factor_CS, -3);  //* Negative charges (e.g., electrons)
                                else
                                    sample_Maxwell_Juttner(u, v, w, thermal_spread_CS, lorentz_factor_CS, 3);   //* Positive charges (e.g., positrons)
                                
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

//! Initial particle distributions (Non Relativistic and Relativistic) !//

//? Quasi-1D ion-electron shock (Relativistic and Non relativistic)
void Particles3D::Shock1D(Field * EMf) 
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
    //TODO: Ask Fabio -- LF, drift

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


//! ============================================================================= !//

// Create a vectorized version of this mover as follows.
//
// Let N be the number of doubles that fit in the vector unit.
//
// Vectorization can be done in one of two ways:
// 1. trivially by writing loops over stride-1 data,
//    particularly if data is in SoA format, and
// 2. with intrinsics, e.g. by loading the components
//    of physical vectors in the vector unit and
//    performing physical vector operations.
//
// Process N:=sizeof(vector_unit)/sizeof(double) particles at a time:
//
// A. Initialize average position with old position.
// B. repeat the following steps:
//    1. Use average position to generate 8 node weights and indices:
//       for pcl=1:N: (3) position -> (8) weights and (8) node_indices
//    2. Use weights and fields at node indices to calculate sampled field:
//       for pcl=1:N: (8) weights, field at nodes -> (6 or 8) sampled field
//    3. Use sampled field to advance velocity:
//       for pcl=1:N: sampled field, velocity -> (3) new velocity
//    4. Use velocity to advance position:
//       for pcl=1:N: velocity, old position -> (3) average position
// C. Use old position and average position to update the position.
//
// Since the fields used by the pusher are in AoS format,
// step 2 can be vectorized trivially, independent of whether
// the particle representation is AoS or SoA, and is expected
// to dominate in the absence of vectorization.
//
// If the particle representation is AoS, then
// vectorizing steps 1, 3, and 4 requires intrinsics.
//
// If the particle representation is SoA then
// vectorizing step 3 requires intrinsics unless:
// 1. the Nx(6 or 8) array of sampled fields is transposed
//    from AoS format to SoA format or
// 2. particles are sorted by mesh cell and are subcycled
//    or resorted with each iteration to keep the average
//    position within the mesh cell;
//    in this case the cell nodes are the same for all
//    N particles, so the sampled fields can be directly
//    calculated in SoA format.
//
// Compare the vectorization notes at the top of sumMoments()
//

//! ECSIM - energy conserving semi-implicit method !//
void Particles3D::ECSIM_velocity(Field *EMf) 
{
    #pragma omp parallel
    {
        convertParticlesToAoS();

        #pragma omp master
        if (vct->getCartesian_rank() == 0) 
            cout << "*** ECSIM MOVER (velocities) for species " << ns << " ***" << endl;

        const_arr4_double fieldForPcls = EMf->get_fieldForPcls();

        const double qdto2mc = 0.5 * dt * qom/c;

        #pragma omp for schedule(static)
        for (int pidx = 0; pidx < getNOP(); pidx++) 
        {
            //* Copy the particle
		    SpeciesParticle* pcl = &_pcls[pidx];
		    ALIGNED(pcl);

            //* --------------------------------------- *//

            //* Copy particles' positions and velocities at the 'n^th' time step
            const double x_n = pcl->get_x();
            const double y_n = pcl->get_y();
            const double z_n = pcl->get_z();
            const double u_n = pcl->get_u();
            const double v_n = pcl->get_v();
            const double w_n = pcl->get_w();
            
            //* Additional variables for storing old and new positions and velocities
            double x_old = x_n; double y_old = y_n; double z_old = z_n;
            double uavg, vavg, wavg;

            //* --------------------------------------- *//

            //* Compute weights for field components
            double weights[8] ALLOC_ALIGNED;
            int cx, cy, cz;
            grid->get_safe_cell_and_weights(x_old, y_old, z_old, cx, cy, cz, weights);

            const double* field_components[8] ALLOC_ALIGNED;
            get_field_components_for_cell(field_components, fieldForPcls, cx, cy, cz);

            double sampled_field[8] ALLOC_ALIGNED;
            for(int i=0;i<8;i++) sampled_field[i]=0;
            
            double& Bxl = sampled_field[0];
            double& Byl = sampled_field[1];
            double& Bzl = sampled_field[2];
            double& Exl = sampled_field[0+DFIELD_3or4];
            double& Eyl = sampled_field[1+DFIELD_3or4];
            double& Ezl = sampled_field[2+DFIELD_3or4];
            
            const int num_field_components = 2*DFIELD_3or4;

            for(int c = 0; c < 8; c++)
            {
                const double* field_components_c = field_components[c];
                ASSUME_ALIGNED(field_components_c);
                const double weights_c = weights[c];
                
                #pragma simd
                for(int i=0; i<num_field_components; i++)
                {
                    sampled_field[i] += weights_c*field_components_c[i];
                }
            }

            //* --------------------------------------- *//

            //? Update (temporary) velocities
            const pfloat u_temp = u_n + qdto2mc*Exl;
            const pfloat v_temp = v_n + qdto2mc*Eyl;
            const pfloat w_temp = w_n + qdto2mc*Ezl;

            const double Omx = qdto2mc*Bxl;
            const double Omy = qdto2mc*Byl;
            const double Omz = qdto2mc*Bzl;

            const pfloat omsq = (Omx * Omx + Omy * Omy + Omz * Omz);
            const pfloat denom = 1.0 / (1.0 + omsq);

            const pfloat udotOm = u_temp * Omx + v_temp * Omy + w_temp * Omz;

            uavg = (u_temp + (v_temp * Omz - w_temp * Omy + udotOm * Omx)) * denom;
            vavg = (v_temp + (w_temp * Omx - u_temp * Omz + udotOm * Omy)) * denom;
            wavg = (w_temp + (u_temp * Omy - v_temp * Omx + udotOm * Omz)) * denom;

            //? Update new velocities at the (n+1)^th time step
            pcl->set_u(2.0 * uavg - u_n);
            pcl->set_v(2.0 * vavg - v_n);
            pcl->set_w(2.0 * wavg - w_n);
        }
    }
    //! End of #pragma omp parallel
}

//! RelSIM - Relativistic semi-implicit method !//
void Particles3D::RelSIM_velocity(Field *EMf) 
{
    #pragma omp parallel
    {
        convertParticlesToAoS();

        #pragma omp master
        if (vct->getCartesian_rank() == 0) 
            cout << "*** RelSIM MOVER (velocities) for species " << ns << " ***" << endl;

        const_arr4_double fieldForPcls = EMf->get_fieldForPcls();

        //* q*dt/(2*m*c)
        const double q_dt_2mc = 0.5*dt*qom/c;

        #pragma omp for schedule(static)
        for (int pidx = 0; pidx < getNOP(); pidx++) 
        {
            //* Copy the particle
		    SpeciesParticle* pcl = &_pcls[pidx];
		    ALIGNED(pcl);

            //* Copy particles' positions and velocities at the 'n^th' time step
            const double x_n = pcl->get_x();
            const double y_n = pcl->get_y();
            const double z_n = pcl->get_z();
            const double u_n = pcl->get_u();
            const double v_n = pcl->get_v();
            const double w_n = pcl->get_w();
            
            //* Additional variables for storing old and new positions and velocities
            double x_old = x_n; double y_old = y_n; double z_old = z_n;
            double uavg, vavg, wavg;

            //? Lorentz factor for each particle (relativistic cases)
            double lorentz_factor = sqrt(1.0 + (u_n*u_n + v_n*v_n + w_n*w_n)/(c*c));

            //* Variables for storing temporary velocity variables for the "Lapenta_Markidis" particle pusher
            const double u_old = u_n;
            const double v_old = v_n;
            const double w_old = w_n;
            const double lorentz_factor_old = lorentz_factor;
            double u_bar, v_bar, w_bar, lorentz_factor_bar;

            //* --------------------------------------- *//

            //* Compute weights for field components
            double weights[8] ALLOC_ALIGNED;
            int cx, cy, cz;
            grid->get_safe_cell_and_weights(x_old, y_old, z_old, cx, cy, cz, weights);

            const double* field_components[8] ALLOC_ALIGNED;
            get_field_components_for_cell(field_components, fieldForPcls, cx, cy, cz);

            double sampled_field[8] ALLOC_ALIGNED;
            for(int i = 0; i < 8; i++) sampled_field[i] = 0;
            
            double& Bxl = sampled_field[0];
            double& Byl = sampled_field[1];
            double& Bzl = sampled_field[2];
            double& Exl = sampled_field[0+DFIELD_3or4];
            double& Eyl = sampled_field[1+DFIELD_3or4];
            double& Ezl = sampled_field[2+DFIELD_3or4];

            //TODO: External forces are to be implemented
            double Fxl = 0.0, Fyl = 0.0, Fzl = 0.0;
            
            const int num_field_components = 2*DFIELD_3or4;

            for(int c = 0; c < 8; c++)
            {
                const double* field_components_c=field_components[c];
                ASSUME_ALIGNED(field_components_c);
                const double weights_c = weights[c];
                
                #pragma simd
                for(int i=0; i<num_field_components; i++)
                {
                    sampled_field[i] += weights_c*field_components_c[i];
                }
            }

            //* --------------------------------------- *//
            
            if (Relativistic_pusher == "Boris")
            {
                //? Update (temporary) velocities
                const double u_temp = u_n + q_dt_2mc*c*Exl + 0.5*dt*Fxl;
                const double v_temp = v_n + q_dt_2mc*c*Eyl + 0.5*dt*Fyl;
                const double w_temp = w_n + q_dt_2mc*c*Ezl + 0.5*dt*Fzl;

                double lorentz_factor_new = sqrt(1.0 + (u_temp*u_temp + v_temp*v_temp + w_temp*w_temp)/(c*c));

                //! TODO: Division by c?? check in paper
                Bxl = Bxl*q_dt_2mc/lorentz_factor_new;
                Byl = Byl*q_dt_2mc/lorentz_factor_new;
                Bzl = Bzl*q_dt_2mc/lorentz_factor_new;
                const double B_squared = Bxl*Bxl + Byl*Byl + Bzl*Bzl;

                //* Solve velocity equation (relativistic Boris)
                const double u_new = -(Byl*Byl*u_temp) - (Bzl*Bzl*u_temp) + (Bxl*Byl*v_temp) + (Bxl*Bzl*w_temp) - Byl*w_temp + Bzl*v_temp;
                const double v_new = -(Bxl*Bxl*v_temp) - (Bzl*Bzl*v_temp) + (Bxl*Byl*u_temp) + (Byl*Bzl*w_temp) - Bzl*u_temp + Bxl*w_temp;
                const double w_new = -(Bxl*Bxl*w_temp) - (Byl*Byl*w_temp) + (Bxl*Bzl*u_temp) + (Byl*Bzl*v_temp) - Bxl*v_temp + Byl*u_temp;

                //* New velocities at the (n+1)^th time step
                uavg = u_temp + u_new * 2.0/(1.0+B_squared) + q_dt_2mc * Exl + 0.5 * dt * Fxl;
                vavg = v_temp + v_new * 2.0/(1.0+B_squared) + q_dt_2mc * Eyl + 0.5 * dt * Fyl;
                wavg = w_temp + w_new * 2.0/(1.0+B_squared) + q_dt_2mc * Ezl + 0.5 * dt * Fzl;

            }
            else if (Relativistic_pusher == "Lapenta_Markidis")
            {
                //? The equations for the Lapenta-Markidis pusher can be found in Bacchini (2023), ApJS, 268, 60

                //* beta = q*dt*B^n/(2*m*c)
                double beta_x = q_dt_2mc*Bxl;
                double beta_y = q_dt_2mc*Byl;
                double beta_z = q_dt_2mc*Bzl;
                const double B_squared = beta_x*beta_x + beta_y*beta_y + beta_z*beta_z;

                //* epsilon = q*dt*E^(n+theta)/(2*m)
                double eps_x = q_dt_2mc*Exl + 0.5*dt*Fxl;
                double eps_y = q_dt_2mc*Eyl + 0.5*dt*Fyl;
                double eps_z = q_dt_2mc*Ezl + 0.5*dt*Fzl;
                
                //* u' = u + epsilon
                const double u_prime = u_n + eps_x;
                const double v_prime = v_n + eps_y;
                const double w_prime = w_n + eps_z;

                //* Polynomial coefficients
                double u_dot_eps  = u_prime*eps_x + v_prime*eps_y + w_prime*eps_z;
                double beta_dot_e = beta_x*eps_x + beta_y*eps_y + beta_z*eps_z;
                double u_dot_beta = u_prime*beta_x + v_prime*beta_y + w_prime*beta_z;
                
                double u_cross_beta_x =  v_prime*beta_z - w_prime*beta_y;
                double v_cross_beta_y = -u_prime*beta_z + w_prime*beta_x;
                double w_cross_beta_z =  u_prime*beta_y - v_prime*beta_x;
                
                double aa = u_dot_eps - B_squared;
                double bb = u_cross_beta_x*eps_x + v_cross_beta_y*eps_y + w_cross_beta_z*eps_z + lorentz_factor_old*B_squared;
                double cc = u_dot_beta*beta_dot_e;

                //* Solution coefficients
                double AA = 2.*aa/3.+lorentz_factor_old*lorentz_factor_old/4.;
                double BB = 4.*aa*lorentz_factor_old+8.*bb+lorentz_factor_old*lorentz_factor_old*lorentz_factor_old;
                double DD = aa*aa-3.*bb*lorentz_factor_old-12.*cc;
                double FF = -2.*aa*aa*aa+9.*aa*bb*lorentz_factor_old-72.*aa*cc+27.*bb*bb-27.*cc*lorentz_factor_old*lorentz_factor_old;
                std::complex<double> GG = FF*FF-4.*DD*DD*DD;
                std::complex<double> EE;
                if (std::real((FF+sqrt(GG))/2.)<0.) EE = -pow(-(FF+sqrt(GG))/2.,1./3.);
                else EE = pow((FF+sqrt(GG))/2.,1./3.);
                std::complex<double> CC = DD/(EE+1.e-20)/3.+EE/3.;
                
                //* Solution
                std::complex<double> lorentz_factor_bar_complex = lorentz_factor_old/4.+sqrt(2.*AA+BB/4./sqrt(AA+CC+1.e-20)-CC)/2.+sqrt(AA+CC)/2.;
                lorentz_factor_bar = (double) std::real(lorentz_factor_bar_complex);

                u_bar = (u_prime + (u_prime*beta_x + v_prime*beta_y + w_prime*beta_z)*beta_x/(lorentz_factor_bar*lorentz_factor_bar) + ( v_prime*beta_z - w_prime*beta_y)/lorentz_factor_bar)/(1.0 + B_squared/lorentz_factor_bar/lorentz_factor_bar);
                v_bar = (v_prime + (u_prime*beta_x + v_prime*beta_y + w_prime*beta_z)*beta_y/(lorentz_factor_bar*lorentz_factor_bar) + (-u_prime*beta_z + w_prime*beta_x)/lorentz_factor_bar)/(1.0 + B_squared/lorentz_factor_bar/lorentz_factor_bar);
                w_bar = (w_prime + (u_prime*beta_x + v_prime*beta_y + w_prime*beta_z)*beta_z/(lorentz_factor_bar*lorentz_factor_bar) + ( u_prime*beta_y - v_prime*beta_x)/lorentz_factor_bar)/(1.0 + B_squared/lorentz_factor_bar/lorentz_factor_bar);

                //* New velocities at the (n+1)^th time step
                uavg = 2.0 * u_bar - u_old;
                vavg = 2.0 * v_bar - v_old;
                wavg = 2.0 * w_bar - w_old;
            }
            else
            {
                if (vct->getCartesian_rank() == 0) 
                    cout << "Incorrect relativistic pusher! Please choose either 'Boris' or 'Lapenta_Markidis'" << endl;
                exit(1);
            }

            //* --------------------------------------- *//

            //? Update new velocities at the (n+1)^th time step
            pcl->set_u(uavg);
            pcl->set_v(vavg);
            pcl->set_w(wavg);
        }
    }
}

void Particles3D::ECSIM_position(Field *EMf) 
{
    #pragma omp parallel
    {
        convertParticlesToAoS();

        #pragma omp master
        if (Relativistic)
        {
            if (vct->getCartesian_rank() == 0) 
                cout << "*** RelSIM MOVER (positions) for species " << ns << " ***" << endl;
        }
        else
        {
            if (vct->getCartesian_rank() == 0) 
                cout << "*** ECSIM MOVER (positions) for species " << ns << " ***" << endl;
        }

        double correct_x = 1.0, correct_y = 1.0, correct_z = 1.0;
        const double inv_dx = 1.0/dx, inv_dy = 1.0/dy, inv_dz = 1.0/dz;
        double dxp = 0.0, dyp = 0.0, dzp = 0.0; double lorentz_factor = 1.0;

        //* Parameters for charge conservation
        double eta0, eta1, zeta0, zeta1, xi0, xi1;
        double weight00, weight01, weight10, weight11;
        double invSURF; double limiter = 1.0; 
        
        array3_double temp_R(nxc, nyc, nzc);
        eqValue(0.0, temp_R, nxc, nyc, nzc);

        for (int ii = 0; ii < nxc; ii++)
            for (int jj = 0; jj < nyc; jj++)
                for (int kk = 0; kk < nzc; kk++)
                    temp_R.fetch(ii, jj, kk) = EMf->getResDiv(ii, jj, kk, ns);

        #pragma omp for schedule(static)
        for (int pidx = 0; pidx < getNOP(); pidx++) 
        {
            //* Copy the particle
		    SpeciesParticle* pcl = &_pcls[pidx];
		    ALIGNED(pcl);

            //* --------------------------------------- *//

            //* Copy particles' positions and velocities at the 'n^th' time step
            const double x_n = pcl->get_x();
            const double y_n = pcl->get_y();
            const double z_n = pcl->get_z();
            const double u_n = pcl->get_u();
            const double v_n = pcl->get_v();
            const double w_n = pcl->get_w();
            
            //* Additional variables for storing old and new positions and velocities
            double xavg = x_n; double yavg = y_n; double zavg = z_n;

            //? Lorentz factor at time step "n"
            if (Relativistic) 
                lorentz_factor = sqrt(1.0 + (u_n*u_n + v_n*v_n + w_n*w_n)/(c*c));

            //* --------------------------------------- *//

            //! This may be further optimised - PJD
            //? Charge Conservation (energy conservation inherently violates charge conservation --> charge has to be conserved separately)

            const double ixd = floor((xavg - dx/2.0 - xstart) * inv_dx);
			const double iyd = floor((yavg - dy/2.0 - ystart) * inv_dy);
			const double izd = floor((zavg - dz/2.0 - zstart) * inv_dz);
			int ix = 2 + int (ixd);
			int iy = 2 + int (iyd);
			int iz = 2 + int (izd);
		
			//* Difference along X
			eta0  = yavg - grid->getYC(ix, iy - 1, iz);
			zeta0 = zavg - grid->getZC(ix, iy, iz - 1);
			eta1  = grid->getYC(ix, iy, iz) - yavg;
			zeta1 = grid->getZC(ix, iy, iz) - zavg;
			invSURF = 1.0/(dy*dz);

            double RxP = 0.0;
			double RxM = 0.0;
		
			weight00 = eta0 * zeta0 * invSURF;
			weight01 = eta0 * zeta1 * invSURF;
			weight10 = eta1 * zeta0 * invSURF;
			weight11 = eta1 * zeta1 * invSURF;

            RxP  = weight00 * temp_R.get(ix, iy, iz);
			RxP += weight01 * temp_R.get(ix, iy, iz - 1);
			RxP += weight10 * temp_R.get(ix, iy - 1, iz);
			RxP += weight11 * temp_R.get(ix, iy - 1, iz - 1);

            RxM  = weight00 * temp_R.get(ix - 1, iy, iz);
			RxM += weight01 * temp_R.get(ix - 1, iy, iz - 1);
			RxM += weight10 * temp_R.get(ix - 1, iy - 1, iz);
			RxM += weight11 * temp_R.get(ix - 1, iy - 1, iz - 1);

            //* Difference along Y
			xi0   = xavg - grid->getXC(ix - 1, iy, iz);
			zeta0 = zavg - grid->getZC(ix, iy, iz - 1);
			xi1   = grid->getXC(ix, iy, iz) - xavg;
			zeta1 = grid->getZC(ix, iy, iz) - zavg;
			invSURF = 1.0/(dx*dz);
		
			double RyP = 0.0;
			double RyM = 0.0;

            weight00 = xi0 * zeta0 * invSURF;
			weight01 = xi0 * zeta1 * invSURF;
			weight10 = xi1 * zeta0 * invSURF;
			weight11 = xi1 * zeta1 * invSURF;

            RyP  = weight00 * temp_R.get(ix, iy, iz);
			RyP += weight01 * temp_R.get(ix, iy, iz - 1);
			RyP += weight10 * temp_R.get(ix - 1, iy, iz);
			RyP += weight11 * temp_R.get(ix - 1, iy, iz - 1);
		
            RyM  = weight00 * temp_R.get(ix, iy - 1, iz);
			RyM += weight01 * temp_R.get(ix, iy - 1, iz - 1);
			RyM += weight10 * temp_R.get(ix - 1, iy - 1, iz);
			RyM += weight11 * temp_R.get(ix - 1, iy - 1, iz - 1);

            //* Difference along Z
			double RzP = 0.0;
			double RzM = 0.0;
		
			xi0  = xavg - grid->getXC(ix - 1, iy, iz);
			eta0 = yavg - grid->getYC(ix, iy - 1, iz);
			xi1  = grid->getXC(ix, iy, iz) - xavg;
			eta1 = grid->getYC(ix, iy, iz) - yavg;
			invSURF = 1.0/(dx*dy);

            weight00 = xi0 * eta0 * invSURF;
			weight01 = xi0 * eta1 * invSURF;
			weight10 = xi1 * eta0 * invSURF;
			weight11 = xi1 * eta1 * invSURF;

            RzP  = weight00 * temp_R.get(ix, iy, iz);
			RzP += weight01 * temp_R.get(ix, iy - 1, iz);
			RzP += weight10 * temp_R.get(ix - 1, iy, iz);
			RzP += weight11 * temp_R.get(ix - 1, iy - 1, iz);
		
            RzM  = weight00 * temp_R.get(ix, iy, iz - 1);
			RzM += weight01 * temp_R.get(ix, iy - 1, iz - 1);
			RzM += weight10 * temp_R.get(ix - 1, iy, iz - 1);
			RzM += weight11 * temp_R.get(ix - 1, iy - 1, iz - 1);

            dxp = 0.25 * (RxP - RxM) * dx;
			dxp = -dxp/abs(dxp+1e-10) * min(abs(dxp), abs(u_n/lorentz_factor*dt)/limiter);
		
			dyp = 0.25 * (RyP - RyM) * dy;
			dyp = -dyp/abs(dyp+1e-10) * min(abs(dyp), abs(v_n/lorentz_factor*dt)/limiter);
		
			dzp = 0.25 * (RzP - RzM) * dz;
			dzp = -dzp/abs(dzp+1e-10) * min(abs(dzp), abs(w_n/lorentz_factor*dt)/limiter);

            //* --------------------------------------- *//

            //? Update the positions with the new velocity
            pcl->set_x(xavg + u_n/lorentz_factor * dt * correct_x + dxp);
            pcl->set_y(yavg + v_n/lorentz_factor * dt * correct_y + dyp);
            pcl->set_z(zavg + w_n/lorentz_factor * dt * correct_z + dzp);
        }                             
        //! END OF ALL THE PARTICLES

        //* 1D: particle positions for Y & Z = 0; 2D: particle positions for Z = 0
        fixPosition();
    }
}

//? Set particles' poitions to 0 along unused dimensions
void Particles3D::fixPosition()
{
    if (col->getDim() == 1) 
    {    
        for (int pidx = 0; pidx < getNOP(); pidx++) 
        {
            SpeciesParticle* pcl = &_pcls[pidx];
		    ALIGNED(pcl);
            pcl->set_y(0.0);
            pcl->set_z(0.0);
        }
    } 
    else if (col->getDim() == 2) 
    {
        for (int pidx = 0; pidx < getNOP(); pidx++) 
        {
            SpeciesParticle* pcl = &_pcls[pidx];
		    ALIGNED(pcl);
            pcl->set_z(0.0);
        }
    }
}

//? Compute ECSIM moments (rho, J_hat, and mass matrix)
void Particles3D::computeMoments(Field *EMf) 
{
    //! This function is the computational bottleneck and should be further optimised - PJD

    #ifdef __PROFILE_MOMENTS__
    LeXInt::timer time_fc, time_add, time_mm, time_total;

    time_total.start();
    #endif

    #pragma omp master
    if (vct->getCartesian_rank() == 0) 
        cout << "Number of particles of species " << ns << " per MPI process: " << getNOP() << endl;

    #pragma omp parallel
    {
        convertParticlesToAoS();

        //TODO: External forces are to be implemented
        double Fxl = 0.0, Fyl = 0.0, Fzl = 0.0;

        const_arr4_double fieldForPcls = EMf->get_fieldForPcls();
        
        //* q*dt/(2*m*c)
        const double q_dt_2mc = 0.5*dt*qom/c;

        #pragma omp for schedule(static)
        for (int pidx = 0; pidx < getNOP(); pidx++)
        {
            //* Copy the particle
		    SpeciesParticle* pcl = &_pcls[pidx];
		    ALIGNED(pcl);

            //* Copy particles' positions and velocities at the 'n^th' time step
            const double x_n = pcl->get_x();
            const double y_n = pcl->get_y();
            const double z_n = pcl->get_z();
            const double u_n = pcl->get_u();
            const double v_n = pcl->get_v();
            const double w_n = pcl->get_w();
            const double q   = pcl->get_q();

            //* Additional variables for storing old and new positions and velocities
            double x_old = x_n; double y_old = y_n; double z_old = z_n;

            //* --------------------------------------- *//

            #ifdef __PROFILE_MOMENTS__
            time_fc.start();
            #endif

            //* Compute weights for field components
            double weights[8] ALLOC_ALIGNED;
            int cx, cy, cz;
            grid->get_safe_cell_and_weights(x_old, y_old, z_old, cx, cy, cz, weights);

            const double* field_components[8] ALLOC_ALIGNED;
            get_field_components_for_cell(field_components, fieldForPcls, cx, cy, cz);

            double sampled_field[8] ALLOC_ALIGNED;
            for(int i = 0; i < 8;i++) sampled_field[i] = 0;
            
            double& Bxl = sampled_field[0];
            double& Byl = sampled_field[1];
            double& Bzl = sampled_field[2];
            double& Exl = sampled_field[0+DFIELD_3or4];
            double& Eyl = sampled_field[1+DFIELD_3or4];
            double& Ezl = sampled_field[2+DFIELD_3or4];
            
            const int num_field_components = 2*DFIELD_3or4;

            //TODO: External force are to be implemented in "sampled_field"
            for(int c = 0; c < 8; c++)
            {
                const double* field_components_c = field_components[c];
                ASSUME_ALIGNED(field_components_c);
                const double weights_c = weights[c];
                
                #pragma simd
                for(int i = 0; i < num_field_components; i++)
                {
                    sampled_field[i] += weights_c*field_components_c[i];
                }
            }

            #ifdef __PROFILE_MOMENTS__
            time_fc.stop();
            #endif

            //* --------------------------------------- *//

            //? Compute the rotation matrix, "alpha"
            double lorentz_factor = 1.0;
            if (Relativistic)
            {
                if (Relativistic_pusher == "Boris")
                {
                    //* u_temp = u + q*dt*E^(n + theta)/(2*m) + F*dt/2 (F --> external force)
                    const double u_temp = u_n + q_dt_2mc*c*Exl + 0.5*dt*Fxl;
                    const double v_temp = v_n + q_dt_2mc*c*Eyl + 0.5*dt*Fyl;
                    const double w_temp = w_n + q_dt_2mc*c*Ezl + 0.5*dt*Fzl;

                    //* lorentz_factor = [1 + ((u^2 + v^2 + w^2)/c^2)]^0.5
                    lorentz_factor = sqrt(1.0 + (u_temp*u_temp + v_temp*v_temp + w_temp*w_temp)/(c*c));
                }
                else if (Relativistic_pusher == "Lapenta_Markidis")
                {
                    //* lorentz_factor = [1 + ((u^2 + v^2 + w^2)/c^2)]^0.5
                    double lorentz_factor_old = sqrt(1.0 + (u_n*u_n + v_n*v_n + w_n*w_n)/(c*c));

                    //? The equations for the Lapenta-Markidis pusher can be found in Bacchini (2023), ApJ, 268, 60

                    //* beta = q*dt*B^n/(2*m*c)
                    double beta_x = q_dt_2mc*Bxl;
                    double beta_y = q_dt_2mc*Byl;
                    double beta_z = q_dt_2mc*Bzl;
                    const double B_squared = beta_x*beta_x + beta_y*beta_y + beta_z*beta_z;

                    //* epsilon = q*dt*E^(n+theta)/(2*m)
                    double eps_x = q_dt_2mc*Exl + 0.5*dt*Fxl;
                    double eps_y = q_dt_2mc*Eyl + 0.5*dt*Fyl;
                    double eps_z = q_dt_2mc*Ezl + 0.5*dt*Fzl;
                    
                    //* u' = u + epsilon
                    const double u_prime = u_n + eps_x;
                    const double v_prime = v_n + eps_y;
                    const double w_prime = w_n + eps_z;

                    //* Polynomial coefficients
                    double u_dot_eps  = u_prime*eps_x + v_prime*eps_y + w_prime*eps_z;
                    double beta_dot_e = beta_x*eps_x + beta_y*eps_y + beta_z*eps_z;
                    double u_dot_beta = u_prime*beta_x + v_prime*beta_y + w_prime*beta_z;
                    
                    double u_cross_beta_x =  v_prime*beta_z - w_prime*beta_y;
                    double v_cross_beta_y = -u_prime*beta_z + w_prime*beta_x;
                    double w_cross_beta_z =  u_prime*beta_y - v_prime*beta_x;
                    
                    double aa = u_dot_eps - B_squared;
                    double bb = u_cross_beta_x*eps_x + v_cross_beta_y*eps_y + w_cross_beta_z*eps_z + lorentz_factor_old*B_squared;
                    double cc = u_dot_beta*beta_dot_e;

                    //* Solution coefficients
                    double AA = 2.*aa/3.+lorentz_factor_old*lorentz_factor_old/4.;
                    double BB = 4.*aa*lorentz_factor_old+8.*bb+lorentz_factor_old*lorentz_factor_old*lorentz_factor_old;
                    double DD = aa*aa-3.*bb*lorentz_factor_old-12.*cc;
                    double FF = -2.*aa*aa*aa+9.*aa*bb*lorentz_factor_old-72.*aa*cc+27.*bb*bb-27.*cc*lorentz_factor_old*lorentz_factor_old;
                    std::complex<double> GG = FF*FF-4.*DD*DD*DD;
                    std::complex<double> EE;
                    if (std::real((FF+sqrt(GG))/2.)<0.) EE = -pow(-(FF+sqrt(GG))/2.,1./3.);
                    else EE = pow((FF+sqrt(GG))/2.,1./3.);
                    std::complex<double> CC = DD/(EE+1.e-20)/3.+EE/3.;
                    
                    //* Solution
                    std::complex<double> lorentz_factor_bar_complex = lorentz_factor_old/4.+sqrt(2.*AA+BB/4./sqrt(AA+CC+1.e-20)-CC)/2.+sqrt(AA+CC)/2.;
                    lorentz_factor = (double) std::real(lorentz_factor_bar_complex);
                }
                else
                {
                    if (vct->getCartesian_rank() == 0) 
                        cout << "Incorrect relativistic pusher! Please choose either 'Boris' or 'Lapenta_Markidis'" << endl;
                    exit(1);
                }
            }

            const double Omx = q_dt_2mc*Bxl/lorentz_factor;
            const double Omy = q_dt_2mc*Byl/lorentz_factor;
            const double Omz = q_dt_2mc*Bzl/lorentz_factor;

            const pfloat omsq = (Omx * Omx + Omy * Omy + Omz * Omz);
            const pfloat denom = 1.0 / (1.0 + omsq)/lorentz_factor;

            double alpha[3][3];
            alpha[0][0] = ( 1.0 + (Omx*Omx))*denom;
            alpha[0][1] = ( Omz + (Omx*Omy))*denom;
            alpha[0][2] = (-Omy + (Omx*Omz))*denom;

            alpha[1][0] = (-Omz + (Omx*Omy))*denom;
            alpha[1][1] = ( 1.0 + (Omy*Omy))*denom;
            alpha[1][2] = ( Omx + (Omy*Omz))*denom;

            alpha[2][0] = ( Omy + (Omx*Omz))*denom;
            alpha[2][1] = (-Omx + (Omy*Omz))*denom;
            alpha[2][2] = ( 1.0 + (Omz*Omz))*denom;

            double qau = q * (alpha[0][0]*(u_n + dt/2.*Fxl) + alpha[0][1]*(v_n + dt/2.*Fyl) + alpha[0][2]*(w_n + dt/2.*Fzl));
            double qav = q * (alpha[1][0]*(u_n + dt/2.*Fxl) + alpha[1][1]*(v_n + dt/2.*Fyl) + alpha[1][2]*(w_n + dt/2.*Fzl));
            double qaw = q * (alpha[2][0]*(u_n + dt/2.*Fxl) + alpha[2][1]*(v_n + dt/2.*Fyl) + alpha[2][2]*(w_n + dt/2.*Fzl));

            //* --------------------------------------- *//

            //* Temporary variable used to add density and current density
            double temp[8];
            
            //* Index of cell of particles
            const int ix = cx + 1;
            const int iy = cy + 1;
            const int iz = cz + 1;

            #ifdef __PROFILE_MOMENTS__
            time_add.start();
            #endif

            //* Add charge density                 
            for (int ii = 0; ii < 8; ii++)
                temp[ii] = q * weights[ii];
            EMf->add_Rho(temp, ix, iy, iz, ns);

            //* Add implicit current density - X
            for (int ii = 0; ii < 8; ii++)
                temp[ii] = qau * weights[ii];
            EMf->add_Jxh(temp, ix, iy, iz, ns);
            
            //* Add implicit current density - Y
            for (int ii = 0; ii < 8; ii++)
                temp[ii] = qav * weights[ii];
            EMf->add_Jyh(temp, ix, iy, iz, ns);

            //* Add implicit current density - Z
            for (int ii = 0; ii < 8; ii++)
                temp[ii] = qaw * weights[ii];
            EMf->add_Jzh(temp, ix, iy, iz, ns);

            #ifdef __PROFILE_MOMENTS__
            time_add.stop();
            time_mm.start();
            #endif
            
            //? Compute exact Mass Matrix
            for (int i = 0; i < 2; i++) 
                for (int j = 0; j < 2; j++) 
                    for (int k = 0; k < 2; k++) 
                    {
                        int ni = ix-i;
                        int nj = iy-j;
                        int nk = iz-k;

                        //* Iterate over half of the 27 neighbouring nodes as M is symmetric
                        for (int n_node = 0; n_node < 14; n_node++)
                        {
                            int n2i = ni + NeNo.getX(n_node);
                            int n2j = nj + NeNo.getY(n_node);
                            int n2k = nk + NeNo.getZ(n_node);

                            int i2 = ix - n2i;
                            int j2 = iy - n2j;
                            int k2 = iz - n2k;

                            //TODO: What does this part actually do? - Ask Fabio
                            //* Check if this node is one of the cell where the particle is
                            if (i2 >= 0 && i2 < 2 && j2 >= 0 && j2 < 2 && k2 >= 0 && k2 < 2) 
                            {
                                // Map (i, j, k) & (i2, j2, k2) to 1D
                                int index1 = i * 4 + j * 2 + k;
                                int index2 = i2 * 4 + j2 * 2 + k2; 
                                double qww = q * q_dt_2mc * weights[index1] * weights[index2];
                                double value[3][3];
                                
                                for (int ind1 = 0; ind1 < 3; ind1++)
                                    for (int ind2 = 0; ind2 < 3; ind2++) 
                                        value[ind1][ind2] = alpha[ind2][ind1]*qww;

                                EMf->add_Mass(value, ni, nj, nk, n_node);
                            }
                        }
                    }
            
            #ifdef __PROFILE_MOMENTS__
            time_mm.stop();
            #endif
        }
    }

    #ifdef __PROFILE_MOMENTS__
    time_total.stop();

    if(MPIdata::get_rank() == 0)
    {
        cout << endl << "   MOMENT GATHERER (computeMoments())" << endl; 
        cout << "       Get field components        : " << time_fc.total()    << " s, fraction of time taken in computeMoments(): " << time_fc.total()/time_total.total() << endl;
        cout << "       Add rho & J                 : " << time_add.total()   << " s, fraction of time taken in computeMoments(): " << time_add.total()/time_total.total() << endl;
        cout << "       Mass Matrix                 : " << time_mm.total()    << " s, fraction of time taken in computeMoments(): " << time_mm.total()/time_total.total() << endl;
        cout << "       computeMoments()            : " << time_total.total() << " s" << endl;
    }
    #endif   
}

//? Supplementary ECSIM moments (not computed at every time step; only written to files)
void Particles3D::compute_supplementary_moments(Field * EMf)
{
    #pragma omp parallel
    {
        convertParticlesToAoS();

        #pragma omp for schedule(static)
        for (int pidx = 0; pidx < getNOP(); pidx++)
        {
            //* Copy the particle
            SpeciesParticle* pcl = &_pcls[pidx];
            ALIGNED(pcl);

            //* Copy particles' positions and velocities at the 'n^th' time step
            const double x_n = pcl->get_x();
            const double y_n = pcl->get_y();
            const double z_n = pcl->get_z();
            const double u_n = pcl->get_u();
            const double v_n = pcl->get_v();
            const double w_n = pcl->get_w();
            const double q   = pcl->get_q();

            //* Compute weights for field components
            double weights[8] ALLOC_ALIGNED;
            int cx, cy, cz;
            grid->get_safe_cell_and_weights(x_n, y_n, z_n, cx, cy, cz, weights);

            //* --------------------------------------- *//

            double lorentz_factor = 1.0;
            double lorentz_factor_E_flux = 1.0;     //* Used to compute energy 
            
            if (Relativistic) 
            {
                lorentz_factor = sqrt(1.0 + (u_n*u_n + v_n*v_n + w_n*w_n)/(c*c));
                lorentz_factor_E_flux = lorentz_factor;
            }
            else 
                lorentz_factor_E_flux = 0.5 * (u_n*u_n + v_n*v_n + w_n*w_n)/(c*c);

            //* --------------------------------------- *//

            //* Temporary variable used to add density and current density
            double temp[8];
            
            //* Index of cell of particles
            const int ix = cx + 1;
            const int iy = cy + 1;
            const int iz = cz + 1;

            //! Add charge density
            for (int ii = 0; ii < 8; ii++)
                temp[ii] = q * weights[ii];
            EMf->add_Rho(temp, ix, iy, iz, ns);

            //! Add current density - X, Y, Z
            for (int ii = 0; ii < 8; ii++)
                temp[ii] = q * u_n/lorentz_factor * weights[ii];
            EMf->add_Jx(temp, ix, iy, iz, ns);
            
            for (int ii = 0; ii < 8; ii++)
                temp[ii] = q * v_n/lorentz_factor * weights[ii];
            EMf->add_Jy(temp, ix, iy, iz, ns);

            for (int ii = 0; ii < 8; ii++)
                temp[ii] = q * w_n/lorentz_factor * weights[ii];
            EMf->add_Jz(temp, ix, iy, iz, ns);

            //! Add pressure tensor - X, Y, Z
            for (int ii = 0; ii < 8; ii++)
                temp[ii] = q * u_n*u_n/lorentz_factor * weights[ii];
            EMf->add_Pxx(temp, ix, iy, iz, ns);
            
            for (int ii = 0; ii < 8; ii++)
                temp[ii] = q * u_n*v_n/lorentz_factor * weights[ii];
            EMf->add_Pxy(temp, ix, iy, iz, ns);

            for (int ii = 0; ii < 8; ii++)
                temp[ii] = q * u_n*w_n/lorentz_factor * weights[ii];
            EMf->add_Pxz(temp, ix, iy, iz, ns);

            for (int ii = 0; ii < 8; ii++)
                temp[ii] = q * v_n*v_n/lorentz_factor * weights[ii];
            EMf->add_Pyy(temp, ix, iy, iz, ns);
            
            for (int ii = 0; ii < 8; ii++)
                temp[ii] = q * v_n*w_n/lorentz_factor * weights[ii];
            EMf->add_Pyz(temp, ix, iy, iz, ns);

            for (int ii = 0; ii < 8; ii++)
                temp[ii] = q * w_n*w_n/lorentz_factor * weights[ii];
            EMf->add_Pzz(temp, ix, iy, iz, ns);

            //! Add energy flux density - X, Y, Z
            for (int ii = 0; ii < 8; ii++)
                temp[ii] = q/qom * u_n * lorentz_factor_E_flux * weights[ii];
            EMf->add_E_flux_x(temp, ix, iy, iz, ns);
            
            for (int ii = 0; ii < 8; ii++)
                temp[ii] = q/qom * v_n * lorentz_factor_E_flux * weights[ii];
            EMf->add_E_flux_y(temp, ix, iy, iz, ns);

            for (int ii = 0; ii < 8; ii++)
                temp[ii] = q/qom * w_n * lorentz_factor_E_flux * weights[ii];
            EMf->add_E_flux_z(temp, ix, iy, iz, ns);

            //TODO: Ask Fabio (no Lorentz factor?)
            if (SaveHeatFluxTensor)
            {
                for (int ii = 0; ii < 8; ii++)
                    temp[ii] = q/qom * u_n*u_n*u_n * weights[ii];
                EMf->add_Qxxx(temp, ix, iy, iz, ns);

                for (int ii = 0; ii < 8; ii++)
                    temp[ii] = q/qom * v_n*v_n*v_n * weights[ii];
                EMf->add_Qyyy(temp, ix, iy, iz, ns);

                for (int ii = 0; ii < 8; ii++)
                    temp[ii] = q/qom * w_n*w_n*w_n * weights[ii];
                EMf->add_Qzzz(temp, ix, iy, iz, ns);

                for (int ii = 0; ii < 8; ii++)
                    temp[ii] = q/qom * u_n*v_n*w_n * weights[ii];
                EMf->add_Qxyz(temp, ix, iy, iz, ns);

                for (int ii = 0; ii < 8; ii++)
                    temp[ii] = q/qom * u_n*u_n*v_n * weights[ii];
                EMf->add_Qxxy(temp, ix, iy, iz, ns);

                for (int ii = 0; ii < 8; ii++)
                    temp[ii] = q/qom * u_n*u_n*w_n * weights[ii];
                EMf->add_Qxxz(temp, ix, iy, iz, ns);

                for (int ii = 0; ii < 8; ii++)
                    temp[ii] = q/qom * u_n*v_n*v_n * weights[ii];
                EMf->add_Qxyy(temp, ix, iy, iz, ns);

                for (int ii = 0; ii < 8; ii++)
                    temp[ii] = q/qom * u_n*w_n*w_n * weights[ii];
                EMf->add_Qxzz(temp, ix, iy, iz, ns);

                for (int ii = 0; ii < 8; ii++)
                    temp[ii] = q/qom * v_n*v_n*w_n * weights[ii];
                EMf->add_Qyyz(temp, ix, iy, iz, ns);

                for (int ii = 0; ii < 8; ii++)
                    temp[ii] = q/qom * v_n*w_n*w_n * weights[ii];
                EMf->add_Qyzz(temp, ix, iy, iz, ns);
            }
        }
    }
}

//! End of ECSIM & RelSIM

//! ============================================================================= !//

//! IMM - Implicit moment method - mover with a Predictor-Corrector scheme !//
void Particles3D::mover_PC(Field * EMf) 
{
    #pragma omp parallel
    {
        convertParticlesToSoA();

        #pragma omp master
        if (vct->getCartesian_rank() == 0) 
        {
            cout << "***PC MOVER species " << ns << " ***" << NiterMover << " ITERATIONS   ****" << endl;
        }
        const_arr4_double fieldForPcls = EMf->get_fieldForPcls();
        
        const double dto2 = .5 * dt, qdto2mc = qom * dto2 / c;

        //? Integrate over all particles
        #pragma omp for schedule(static)
        // why does single precision make no difference in execution speed?
        //#pragma simd vectorlength(VECTOR_WIDTH)
        for (int pidx = 0; pidx < getNOP(); pidx++) 
        {
            //* copy the particle
            const double xorig = getX(pidx);
            const double yorig = getY(pidx);
            const double zorig = getZ(pidx);
            const double uorig = getU(pidx);
            const double vorig = getV(pidx);
            const double worig = getW(pidx);
            double xavg = xorig;
            double yavg = yorig;
            double zavg = zorig;
            double uavg;
            double vavg;
            double wavg;
            
            //? Calculate the average velocity iteratively
            for (int innter = 0; innter < NiterMover; innter++) 
            {
                //* interpolation G-->P
                const double ixd = floor((xavg - xstart) * inv_dx);
                const double iyd = floor((yavg - ystart) * inv_dy);
                const double izd = floor((zavg - zstart) * inv_dz);
                
                //* interface of index to right of cell
                int ix = 2 + int(ixd);
                int iy = 2 + int(iyd);
                int iz = 2 + int(izd);

                //* use field data of closest cell in domain
                if (ix < 1) ix = 1;
                if (iy < 1) iy = 1;
                if (iz < 1) iz = 1;
                if (ix > nxc) ix = nxc;
                if (iy > nyc) iy = nyc;
                if (iz > nzc) iz = nzc;
                
                //* index of cell of particle;
                const int cx = ix - 1;
                const int cy = iy - 1;
                const int cz = iz - 1;

                const double xi0   = xavg - grid->getXN(ix-1);
                const double eta0  = yavg - grid->getYN(iy-1);
                const double zeta0 = zavg - grid->getZN(iz-1);
                const double xi1   = grid->getXN(ix) - xavg;
                const double eta1  = grid->getYN(iy) - yavg;
                const double zeta1 = grid->getZN(iz) - zavg;

                double weights[8] ALLOC_ALIGNED;
                const pfloat weight0 = invVOL*xi0;
                const pfloat weight1 = invVOL*xi1;
                const pfloat weight00 = weight0*eta0;
                const pfloat weight01 = weight0*eta1;
                const pfloat weight10 = weight1*eta0;
                const pfloat weight11 = weight1*eta1;
                weights[0] = weight00*zeta0; // weight000
                weights[1] = weight00*zeta1; // weight001
                weights[2] = weight01*zeta0; // weight010
                weights[3] = weight01*zeta1; // weight011
                weights[4] = weight10*zeta0; // weight100
                weights[5] = weight10*zeta1; // weight101
                weights[6] = weight11*zeta0; // weight110
                weights[7] = weight11*zeta1; // weight111
                //weights[0] = xi0 * eta0 * zeta0 * qi * invVOL; // weight000
                //weights[1] = xi0 * eta0 * zeta1 * qi * invVOL; // weight001
                //weights[2] = xi0 * eta1 * zeta0 * qi * invVOL; // weight010
                //weights[3] = xi0 * eta1 * zeta1 * qi * invVOL; // weight011
                //weights[4] = xi1 * eta0 * zeta0 * qi * invVOL; // weight100
                //weights[5] = xi1 * eta0 * zeta1 * qi * invVOL; // weight101
                //weights[6] = xi1 * eta1 * zeta0 * qi * invVOL; // weight110
                //weights[7] = xi1 * eta1 * zeta1 * qi * invVOL; // weight111

                // creating these aliases seems to accelerate this method by about 30%
                // on the Xeon host, processor, suggesting deficiency in the optimizer.
                //
                const double* field_components[8];
                get_field_components_for_cell(field_components, fieldForPcls, cx, cy, cz);

                //double Exl=0,Exl=0,Ezl=0,Bxl=0,Byl=0,Bzl=0;
                //for(int c=0; c<8; c++)
                //{
                //  Bxl += weights[c] * field_components[c][0];
                //  Byl += weights[c] * field_components[c][1];
                //  Bzl += weights[c] * field_components[c][2];
                //  Exl += weights[c] * field_components[c][0+DFIELD_3or4];
                //  Eyl += weights[c] * field_components[c][1+DFIELD_3or4];
                //  Ezl += weights[c] * field_components[c][2+DFIELD_3or4];
                //}
                // causes compile error on icpc
                //double sampled_field[8]={0,0,0,0,0,0,0,0} ALLOC_ALIGNED;
                double sampled_field[8] ALLOC_ALIGNED;
                for(int i=0;i<8;i++) 
                    sampled_field[i]=0;
                
                double& Bxl=sampled_field[0];
                double& Byl=sampled_field[1];
                double& Bzl=sampled_field[2];
                double& Exl=sampled_field[0+DFIELD_3or4];
                double& Eyl=sampled_field[1+DFIELD_3or4];
                double& Ezl=sampled_field[2+DFIELD_3or4];
                
                const int num_field_components = 2*DFIELD_3or4;
                
                for(int c = 0; c < 8; c++)
                {
                    const double* field_components_c=field_components[c];
                    ASSUME_ALIGNED(field_components_c);
                    const double weights_c = weights[c];
                    
                    #pragma simd
                    for(int i=0; i<num_field_components; i++)
                    {
                        sampled_field[i] += weights_c*field_components_c[i];
                    }
                }

                const double Omx = qdto2mc*Bxl;
                const double Omy = qdto2mc*Byl;
                const double Omz = qdto2mc*Bzl;

                //* end interpolation
                const pfloat omsq = (Omx * Omx + Omy * Omy + Omz * Omz);
                const pfloat denom = 1.0 / (1.0 + omsq);
                
                //? Solve position equation
                const pfloat ut = uorig + qdto2mc * Exl;
                const pfloat vt = vorig + qdto2mc * Eyl;
                const pfloat wt = worig + qdto2mc * Ezl;
                //const pfloat udotb = ut * Bxl + vt * Byl + wt * Bzl;
                const pfloat udotOm = ut * Omx + vt * Omy + wt * Omz;
                
                //? Update velocity
                uavg = (ut + (vt * Omz - wt * Omy + udotOm * Omx)) * denom;
                vavg = (vt + (wt * Omx - ut * Omz + udotOm * Omy)) * denom;
                wavg = (wt + (ut * Omy - vt * Omx + udotOm * Omz)) * denom;
                
                //? Update average position
                xavg = xorig + uavg * dto2;
                yavg = yorig + vavg * dto2;
                zavg = zorig + wavg * dto2;
            }                           
            //! End of iterations
            
            //? Update final positions and velocities
            fetchX(pidx) = xorig + uavg * dt;
            fetchY(pidx) = yorig + vavg * dt;
            fetchZ(pidx) = zorig + wavg * dt;
            fetchU(pidx) = 2.0 * uavg - uorig;
            fetchV(pidx) = 2.0 * vavg - vorig;
            fetchW(pidx) = 2.0 * wavg - worig;
        }                             
        //! END OF ALL PARTICLES
    }
}

void Particles3D::mover_PC_AoS(Field * EMf)
{
	#pragma omp parallel
	{
	  convertParticlesToAoS();

#ifdef PRINTPCL   
          #pragma omp master
	  if (vct->getCartesian_rank() == 0) {
		cout << "***AoS MOVER species " << ns << " *** Max." << NiterMover << " ITERATIONS   ****" << endl;
	  }
	  int sum_innter = 0;
#endif
	  const_arr4_pfloat fieldForPcls = EMf->get_fieldForPcls();
	

	  const double dto2 = .5 * dt, qdto2mc = qom * dto2 / c;
	  #pragma omp for schedule(static)
	  for (int pidx = 0; pidx < getNOP(); pidx++) {
		// copy the particle
		SpeciesParticle* pcl = &_pcls[pidx];
		ALIGNED(pcl);
		const double xorig = pcl->get_x();
		const double yorig = pcl->get_y();
		const double zorig = pcl->get_z();
		const double uorig = pcl->get_u();
		const double vorig = pcl->get_v();
		const double worig = pcl->get_w();
		double xavg = xorig;
		double yavg = yorig;
		double zavg = zorig;
		double uavg, vavg, wavg;
		double uavg_old=uorig;
		double vavg_old=vorig;
		double wavg_old=worig;

		// calculate the average velocity iteratively
		//for (int innter = 0; innter < NiterMover; innter++) {
		int   innter = 0;
		const double PC_err_2 = 1E-12;//square of error tolerance
		double currErr = PC_err_2+1.; //initialize to a larger value

		// calculate the average velocity iteratively
		while(currErr> PC_err_2 &&  innter < NiterMover){

		  // compute weights for field components
		  //
		  double weights[8] ALLOC_ALIGNED;
		  int cx,cy,cz;
		  grid->get_safe_cell_and_weights(xavg,yavg,zavg,cx,cy,cz,weights);

		  const double* field_components[8] ALLOC_ALIGNED;
		  get_field_components_for_cell(field_components,fieldForPcls,cx,cy,cz);

		  //double Exl=0,Exl=0,Ezl=0,Bxl=0,Byl=0,Bzl=0;
		  //for(int c=0; c<8; c++)
		  //{
		  //  Bxl += weights[c] * field_components[c][0];
		  //  Byl += weights[c] * field_components[c][1];
		  //  Bzl += weights[c] * field_components[c][2];
		  //  Exl += weights[c] * field_components[c][0+DFIELD_3or4];
		  //  Eyl += weights[c] * field_components[c][1+DFIELD_3or4];
		  //  Ezl += weights[c] * field_components[c][2+DFIELD_3or4];
		  //}
		  // causes compile error on icpc
		  //double sampled_field[8]={0,0,0,0,0,0,0,0} ALLOC_ALIGNED;
		  double sampled_field[8] ALLOC_ALIGNED;
		  for(int i=0;i<8;i++) sampled_field[i]=0;
		  double& Bxl=sampled_field[0];
		  double& Byl=sampled_field[1];
		  double& Bzl=sampled_field[2];
		  double& Exl=sampled_field[0+DFIELD_3or4];
		  double& Eyl=sampled_field[1+DFIELD_3or4];
		  double& Ezl=sampled_field[2+DFIELD_3or4];
		  const int num_field_components=2*DFIELD_3or4;
		  for(int c=0; c<8; c++)
		  {
			const double* field_components_c=field_components[c];
			ASSUME_ALIGNED(field_components_c);
			const double weights_c = weights[c];
			#pragma simd
			for(int i=0; i<num_field_components; i++)
			{
			  sampled_field[i] += weights_c*field_components_c[i];
			}
		  }
          
		  const double Omx = qdto2mc*Bxl;
		  const double Omy = qdto2mc*Byl;
		  const double Omz = qdto2mc*Bzl;

		  // end interpolation
		  const pfloat omsq = (Omx * Omx + Omy * Omy + Omz * Omz);
		  const pfloat denom = 1.0 / (1.0 + omsq);
		  // solve the position equation
		  const pfloat ut = uorig + qdto2mc * Exl;
		  const pfloat vt = vorig + qdto2mc * Eyl;
		  const pfloat wt = worig + qdto2mc * Ezl;
		  //const pfloat udotb = ut * Bxl + vt * Byl + wt * Bzl;
		  const pfloat udotOm = ut * Omx + vt * Omy + wt * Omz;
		  // solve the velocity equation
		  uavg = (ut + (vt * Omz - wt * Omy + udotOm * Omx)) * denom;
		  vavg = (vt + (wt * Omx - ut * Omz + udotOm * Omy)) * denom;
		  wavg = (wt + (ut * Omy - vt * Omx + udotOm * Omz)) * denom;
		  // update average position
		  xavg = xorig + uavg * dto2;
		  yavg = yorig + vavg * dto2;
		  zavg = zorig + wavg * dto2;

		  innter ++;
		  currErr = ((uavg_old-uavg)*(uavg_old-uavg)+(vavg_old-vavg)*(vavg_old-vavg)+(wavg_old-wavg)*(wavg_old-wavg)) /
								 (uavg_old*uavg_old+vavg_old*vavg_old+wavg_old*wavg_old);
		  // capture the new velocity for the next iteration
		  uavg_old = uavg;
		  vavg_old = vavg;
		  wavg_old = wavg;

		}// end of iteration
#ifdef PRINTPCL
		sum_innter +=innter;
#endif
		// update the final position and velocity
		if(cap_velocity())
		{
		  bool cap = (abs(uavg)>umax || abs(vavg)>vmax || abs(wavg)>wmax)? true : false;
		  // we could do something more smooth or sophisticated
		  if(builtin_expect(cap,false))
		  {
			if(true)
			{
			  dprintf("capping velocity: abs(%g,%g,%g)>(%g,%g,%g)",
				uavg,vavg,wavg,umax,vmax,wmax);
			}
			if(uavg>umax) uavg=umax; else if(uavg<umin) uavg=umin;
			if(vavg>vmax) vavg=vmax; else if(vavg<vmin) vavg=vmin;
			if(wavg>wmax) wavg=wmax; else if(wavg<wmin) wavg=wmin;
		  }
		}
		//
		pcl->set_x(xorig + uavg * dt);
		pcl->set_y(yorig + vavg * dt);
		pcl->set_z(zorig + wavg * dt);
		pcl->set_u(2.0 * uavg - uorig);
		pcl->set_v(2.0 * vavg - vorig);
		pcl->set_w(2.0 * wavg - worig);
	  }// END OF ALL THE PARTICLES
	
#ifdef PRINTPCL  
	  if (vct->getCartesian_rank() == 0) {
		cout << "***AoS MOVER species " << ns << " *** Avg." << (double)sum_innter/((double)getNOP()) << " ITERATIONS   ****" << endl;
	  }
#endif
	}
}

void Particles3D::mover_PC_AoS_Relativistic(Field * EMf)
{
    // The local average number of PC iterations
	// The sum over all processes of the avg. numb. of PC iter
	#ifdef PRINTPCL
	  double innter_sum = 0.0,subcycle_sum=0.0;
	#endif
#pragma omp parallel
{
  convertParticlesToAoS();
  const_arr4_pfloat fieldForPcls = EMf->get_fieldForPcls();

  #pragma omp master
  { timeTasks_begin_task(TimeTasks::MOVER_PCL_MOVING); }


  const double PC_err = 1E-6;
  const int    nop=getNOP(); //for OpenMP
  int 		   pidx = 0;      //for OpenMP

#ifdef PRINTPCL
	#pragma omp for schedule(static) reduction(+:innter_sum,subcycle_sum)
#else
	#pragma omp for schedule(static)
#endif
  for (pidx = 0; pidx < nop; pidx++) {

	  // copy the particle
	  SpeciesParticle* pcl = &_pcls[pidx];
	  ALIGNED(pcl);

	  //determine # of subcyles
	  double weights[8] ALLOC_ALIGNED;
	  int cx,cy,cz;
	  grid->get_safe_cell_and_weights(pcl->get_x(),pcl->get_y(),pcl->get_z(),cx,cy,cz,weights);
	  const double* field_components[8] ALLOC_ALIGNED;
	  get_field_components_for_cell(field_components,fieldForPcls,cx,cy,cz);
	  double Bxl = 0.0;
	  double Byl = 0.0;
	  double Bzl = 0.0;
	  for(int c=0; c<8; c++)
	  {
		Bxl += weights[c] * field_components[c][0];
		Byl += weights[c] * field_components[c][1];
		Bzl += weights[c] * field_components[c][2];
	  }
	  const double B_mag= sqrt(Bxl*Bxl+Byl*Byl+Bzl*Bzl);
	  double dt_sub = M_PI*c/(4*abs(qom)*B_mag);
	  const int sub_cycles = int(dt/dt_sub)+1;
	  dt_sub = dt/double(sub_cycles);
	  double dto2_sub = .5 * dt_sub, qdto2mc_sub = qom*dto2_sub/c;

#ifdef PRINTPCL
	  subcycle_sum += sub_cycles;
	  double innter_avg = 0.0;
#endif

	  //start subcycling
	  for(int cyc_cnt=0;cyc_cnt<sub_cycles;cyc_cnt++)
	  {

		    const double xorig = pcl->get_x();
       	    const double yorig = pcl->get_y();
       	    const double zorig = pcl->get_z();
       	    const double uorig = pcl->get_u();
       	    const double vorig = pcl->get_v();
       	    const double worig = pcl->get_w();
       	    double xavg = xorig;
       	    double yavg = yorig;
       	    double zavg = zorig;
       	    double uavg_old, uavg = uorig;
       	    double vavg_old, vavg = vorig;
       	    double wavg_old, wavg = worig;
       	    const double gamma0 = 1.0/(sqrt(1.0 - uorig*uorig - vorig*vorig - worig*worig));
       	    double gamma1;
       	    pfloat ut,vt,wt;

            int innter = 0;
            double currErr = PC_err+1;//initialize to a larger value

            // calculate the average velocity iteratively
            while(currErr> PC_err*PC_err &&  innter < NiterMover){

				  // Save old v_avg
				  uavg_old = uavg;
				  vavg_old = vavg;
				  wavg_old = wavg;

				  // compute weights for field components
				  double weights[8] ALLOC_ALIGNED;
				  int cx,cy,cz;
				  grid->get_safe_cell_and_weights(xavg,yavg,zavg,cx,cy,cz,weights);

				  const double* field_components[8] ALLOC_ALIGNED;
				  get_field_components_for_cell(field_components,fieldForPcls,cx,cy,cz);

				  double Exl = 0.0;
				  double Eyl = 0.0;
				  double Ezl = 0.0;
				  double Bxl = 0.0;
				  double Byl = 0.0;
				  double Bzl = 0.0;
				  for(int c=0; c<8; c++)
				  {
					Bxl += weights[c] * field_components[c][0];
					Byl += weights[c] * field_components[c][1];
					Bzl += weights[c] * field_components[c][2];
					Exl += weights[c] * field_components[c][0+DFIELD_3or4];
					Eyl += weights[c] * field_components[c][1+DFIELD_3or4];
					Ezl += weights[c] * field_components[c][2+DFIELD_3or4];
				  }
				  const double Omx = qdto2mc_sub*Bxl;
				  const double Omy = qdto2mc_sub*Byl;
				  const double Omz = qdto2mc_sub*Bzl;

				  // end interpolation
				  const pfloat omsq = (Omx * Omx + Omy * Omy + Omz * Omz);
				  pfloat denom = 1.0 / (1.0 + omsq);


				  //relativistic part

				  // solve the position equation
				  const pfloat ut = uorig*gamma0 + qdto2mc_sub * Exl;
				  const pfloat vt = vorig*gamma0 + qdto2mc_sub * Eyl;
				  const pfloat wt = worig*gamma0 + qdto2mc_sub * Ezl;

				  gamma1 = sqrt(1.0 + ut*ut + vt*vt + wt*wt);
				  Bxl /=gamma1;
				  Byl /=gamma1;
				  Bzl /=gamma1;
				  denom /=gamma1;

				  const pfloat udotOm = ut * Omx + vt * Omy + wt * Omz;

				  // solve the velocity equation
				  uavg = (ut + (vt * Omz - wt * Omy + udotOm * Omx)) * denom;
				  vavg = (vt + (wt * Omx - ut * Omz + udotOm * Omy)) * denom;
				  wavg = (wt + (ut * Omy - vt * Omx + udotOm * Omz)) * denom;

				  // update average position
				  xavg = xorig + uavg * dto2_sub;
				  yavg = yorig + vavg * dto2_sub;
				  zavg = zorig + wavg * dto2_sub;

				  currErr = ((uavg_old-uavg)*(uavg_old-uavg)+(vavg_old-vavg)*(vavg_old-vavg)+(wavg_old-wavg)*(wavg_old-wavg)) /
					     (uavg_old*uavg_old+vavg_old*vavg_old+wavg_old*wavg_old);
				  innter++;
            }  // end of iteration
#ifdef PRINTPCL
            innter_avg += innter;
#endif

            // relativistic velocity update
		    ut = uorig*gamma0;
		    vt = vorig*gamma0;
		    wt = worig*gamma0;
		    double vt_sq = ut*ut + vt*vt + wt*wt;
		    double vavg_sq = uavg*uavg + vavg*vavg + wavg*wavg;
		    double vt_vavg = ut*uavg + vt*vavg + wt*wavg;

		    double cfa = 1.0 - vavg_sq;
		    double cfb =-2.0*(-vt_vavg+gamma0*vavg_sq);
		    double cfc =-1.0-gamma0*gamma0*vavg_sq+2.0*gamma0*vt_vavg - vt_sq;

		    double delta_rel= cfb*cfb -4.0*cfa*cfc;
		    // update velocity
		    if (delta_rel < 0.0){
		        cout << "Relativity violated: gamma0=" << gamma0 << ",  vavg_sq=" << vavg_sq;
		        pcl->set_u((2.0*gamma1)*uavg - uorig*gamma0);
		        pcl->set_v((2.0*gamma1)*vavg - vorig*gamma0);
		        pcl->set_w((2.0*gamma1)*wavg - worig*gamma0);
		    } else {
		        gamma1 = (-cfb+sqrt(delta_rel))/2.0/cfa;
		        pcl->set_u((1.0 + gamma0/gamma1)*uavg - ut/gamma1);
		        pcl->set_v((1.0 + gamma0/gamma1)*vavg - vt/gamma1);
		        pcl->set_w((1.0 + gamma0/gamma1)*wavg - wt/gamma1);
		    }

		    // update the final position and velocity
		    pcl->set_x(xorig + uavg * dt_sub);
		    pcl->set_y(yorig + vavg * dt_sub);
		    pcl->set_z(zorig + wavg * dt_sub);

	  }//END OF subcycling
#ifdef PRINTPCL
	  innter_sum += innter_avg/sub_cycles;
#endif
  }// END OF ALL THE PARTICLES


  #pragma omp master 
  {
	  timeTasks_end_task(TimeTasks::MOVER_PCL_MOVING);

#ifdef PRINTPCL
	  double local_subcycle = subcycle_sum/nop;
	  double local_innter   = innter_sum/nop;
	  double localAvgArr[2];
	  localAvgArr[0]=local_subcycle;
	  localAvgArr[1]=local_innter;
	  double globalAvgArr[2];
	  MPI_Reduce(&localAvgArr, &globalAvgArr, 2, MPI_DOUBLE, MPI_SUM, 0, mpi_comm);
	  if (MPIdata::get_rank() == 0)
		  cout << "*** Relativistic AoS MOVER with Subcycling  species " << ns << " ***" 
		  << globalAvgArr[0]/MPIdata::get_nprocs()  << " subcyles ***" << globalAvgArr[1]/MPIdata::get_nprocs()<< " ITERATIONS   ****" << endl;
#endif
  }
}
}

// move the particle using MIC vector intrinsics
void Particles3D::mover_PC_AoS_vec_intr(Field * EMf)
{
 #ifndef __MIC__
	   if (MPIdata::get_rank() == 0)
			  cout << "*** MIC not enabled, switch to AoS mover " <<endl;
   	   mover_PC_AoS(EMf);
 #else
 #pragma omp parallel
 {
  convertParticlesToAoS();
  // Here and below x stands for all 3 physical position coordinates
  // and u stands for all 3 velocity coordinates.
  const F64vec8 dx_inv = make_F64vec8(get_invdx(), get_invdy(), get_invdz());
  // starting physical position of proper subdomain ("pdom", without ghosts)
  const F64vec8 pdom_xlow = make_F64vec8(get_xstart(),get_ystart(), get_zstart());
  F64vec8 umaximum = make_F64vec8(umax,vmax,wmax);
  F64vec8 uminimum = make_F64vec8(umin,vmin,wmin);
  //
  // compute canonical coordinates of subdomain (including ghosts)
  // relative to global coordinates.
  // x = physical position, X = canonical coordinates.
  //
  // starting position of cell in lower corner
  // of proper subdomain (without ghosts);
  // probably this is an integer value, but we won't rely on it.
  const F64vec8 pdom_Xlow = dx_inv*pdom_xlow;
  // g = including ghosts
  // starting position of cell in low corner
  const F64vec8 gdom_Xlow = pdom_Xlow - F64vec8(1.);
  // starting position of cell in high corner of ghost domain
  // in canonical coordinates
  const F64vec8 nXc = make_F64vec8(nxc,nyc,nzc);
  #pragma omp master
  if (vct->getCartesian_rank() == 0) {
    cout << "***AoS_vec_intr MOVER species " << ns << " ***" << NiterMover << " ITERATIONS   ****" << endl;
  }
  const_arr4_pfloat fieldForPcls = EMf->get_fieldForPcls();

  SpeciesParticle * pcls = &_pcls[0];
  ALIGNED(pcls);
  #pragma omp master
  { timeTasks_begin_task(TimeTasks::MOVER_PCL_MOVING); }
  const double dto2_d = .5 * dt;
  const double qdto2mc_d = qom * dto2_d / c;
  const F64vec8 dto2 = F64vec8(dto2_d);
  const F64vec8 qdto2mc = F64vec8(qdto2mc_d);
  #pragma omp for schedule(static)
  for (int pidx = 0; pidx < getNOP(); pidx+=2)
  {
    // copy the particle
    SpeciesParticle* pcl = &(pcls[pidx]);

    // gather position and velocity data from particles
    //
    F64vec8 pcl0 = *(F64vec8*)&(pcl[0]);
    F64vec8 pcl1 = *(F64vec8*)&(pcl[1]);
    const F64vec8 xorig = cat_hgh_halves(pcl0,pcl1);
    F64vec8 xavg = xorig;
    const F64vec8 uorig = cat_low_halves(pcl0,pcl1);

    // calculate the average velocity iteratively
    //
    // (could stop iteration when it is determined that
    // both particles are converged, e.g. if change in
    // xavg is sufficiently small)
    F64vec8 uavg;
    for (int iter = 0; iter < NiterMover; iter++)
    {
      // convert to canonical coordinates relative to subdomain with ghosts
      const F64vec8 gX = dx_inv*xavg - gdom_Xlow;
      F64vec8 cellXstart = floor(gX);
      // map to cell within the process subdomain (including ghosts);
      // this is triggered if xavg is outside the ghost subdomain
      // and results in extrapolation from the nearest ghost cell
      // rather than interpolation as in the usual case.
      cellXstart = maximum(cellXstart,F64vec8(0.));
      cellXstart = minimum(cellXstart,nXc);
      // get cell coordinates.
      const I32vec16 cell = round_to_nearest(cellXstart);
      // get field_components for each particle
      F64vec8 field_components0[8]; // first pcl
      F64vec8 field_components1[8]; // second pcl
      ::get_field_components_for_cell(
        field_components0,field_components1,fieldForPcls,cell);

      // get weights for field_components based on particle position
      //
      F64vec8 weights[2];
      const F64vec8 X = gX - cellXstart;
      construct_weights_for_2pcls(weights, X);

      // interpolate field to get fields
      F64vec8 fields[2];
      // sample fields for first particle
      fields[0] = sample_field_mic(weights[0],field_components0);
      // sample fields for second particle
      fields[1] = sample_field_mic(weights[1],field_components1);
      const F64vec8 B = cat_low_halves(fields[0],fields[1]);
      const F64vec8 E = cat_hgh_halves(fields[0],fields[1]);

      // use sampled field to push particle block
      //
      uavg = compute_uvg_for_2pcls(uorig, B, E, qdto2mc);
      // update average position
      xavg = xorig + uavg*dto2;
    } // end of iterative particle advance
    // update the final position and velocity
    if(cap_velocity())
    {
      uavg = minimum(uavg,umaximum);
      uavg = maximum(uavg,uminimum);
    }
    const F64vec8 xnew = xavg+(xavg - xorig);
    const F64vec8 unew = uavg+(uavg - uorig);
    const F64vec8 pcl0new = cat_low_halves(unew, xnew);
    const F64vec8 pcl1new = cat_hgh_halves(unew, xnew);
    copy012and456(pcl0,pcl0new);
    copy012and456(pcl1,pcl1new);

    // could save using no-read stores( _mm512_storenr_pd),
    // but we just read this, so presumably it is still in cache.
    _mm512_store_pd(&pcl[0], pcl0);
    _mm512_store_pd(&pcl[1], pcl1);
  }
  #pragma omp master
  { timeTasks_end_task(TimeTasks::MOVER_PCL_MOVING); }
 }
 #endif
}

void Particles3D::mover_PC_AoS_vec(Field * EMf)
{
#pragma omp parallel
{
  convertParticlesToAoS();
  #pragma omp master
  if (vct->getCartesian_rank() == 0) {
    cout << "***AoS_vec MOVER species " << ns << " ***" << NiterMover << " ITERATIONS   ****" << endl;
  }
  const_arr4_pfloat fieldForPcls = EMf->get_fieldForPcls();

  const int NUM_PCLS_MOVED_AT_A_TIME = 8;
  // make sure that we won't overrun memory
  int needed_capacity = roundup_to_multiple(getNOP(),NUM_PCLS_MOVED_AT_A_TIME);
  assert_le(needed_capacity,_pcls.capacity());

  #pragma omp master
  { timeTasks_begin_task(TimeTasks::MOVER_PCL_MOVING); }
  const double dto2 = .5 * dt, qdto2mc = qom * dto2 / c;
  #pragma omp for schedule(static)
  for (int pidx = 0; pidx < getNOP(); pidx+=NUM_PCLS_MOVED_AT_A_TIME)
  {
    // copy the particles
    SpeciesParticle* pcl[NUM_PCLS_MOVED_AT_A_TIME];
    for(int i=0;i<NUM_PCLS_MOVED_AT_A_TIME;i++)
    {
      pcl[i] = &_pcls[pidx+i];
    }
    // actually, all the particles are aligned,
    // but the compiler should be able to see that.
    ALIGNED(pcl[0]);
    double xorig[NUM_PCLS_MOVED_AT_A_TIME][3] __attribute__((aligned(64)));
    double uorig[NUM_PCLS_MOVED_AT_A_TIME][3] __attribute__((aligned(64)));
    double  xavg[NUM_PCLS_MOVED_AT_A_TIME][3] __attribute__((aligned(64)));
    double  uavg[NUM_PCLS_MOVED_AT_A_TIME][3] __attribute__((aligned(64)));
    // gather data into vectors
    // #pragma simd collapse(2)
    for(int i=0;i<NUM_PCLS_MOVED_AT_A_TIME;i++)
    for(int j=0;j<3;j++)
    {
      xavg[i][j] = xorig[i][j] = pcl[i]->get_x(j);
      uorig[i][j] = pcl[i]->get_u(j);
    }
    // calculate the average velocity iteratively
    for (int innter = 0; innter < NiterMover; innter++) {

      // compute weights for field components
      //
      double weights[NUM_PCLS_MOVED_AT_A_TIME][8] __attribute__((aligned(64)));
      int cx[NUM_PCLS_MOVED_AT_A_TIME][3] __attribute__((aligned(64)));
      for(int i=0;i<NUM_PCLS_MOVED_AT_A_TIME;i++)
      {
        grid->get_safe_cell_and_weights(xavg[i],cx[i],weights[i]);
      }

      const double* field_components[NUM_PCLS_MOVED_AT_A_TIME][8] __attribute__((aligned(64)));
      for(int i=0;i<NUM_PCLS_MOVED_AT_A_TIME;i++)
      {
        get_field_components_for_cell(field_components[i],fieldForPcls,
          cx[i][0],cx[i][1],cx[i][2]);
      }

      double E[NUM_PCLS_MOVED_AT_A_TIME][3] __attribute__((aligned(64)));
      double B[NUM_PCLS_MOVED_AT_A_TIME][3] __attribute__((aligned(64)));
      // could do this with memset
      // #pragma simd collapse(2)
      for(int i=0;i<NUM_PCLS_MOVED_AT_A_TIME;i++)
      for(int j=0;j<3;j++)
      {
        E[i][j]=0;
        B[i][j]=0;
      }
      for(int i=0; i<NUM_PCLS_MOVED_AT_A_TIME;i++)
      for(int j=0;j<3;j++)
      for(int c=0; c<8; c++)
      {
        B[i][j] += weights[i][c] * field_components[i][c][j];
        E[i][j] += weights[i][c] * field_components[i][c][j+DFIELD_3or4];
      }
      double Om[NUM_PCLS_MOVED_AT_A_TIME][3] __attribute__((aligned(64)));
      for(int i=0; i<NUM_PCLS_MOVED_AT_A_TIME;i++)
      for(int j=0;j<3;j++)
      {
        Om[i][j] = qdto2mc*B[i][j];
      }

      // can these dot products vectorize if
      // NUM_PCLS_MOVED_AT_A_TIME is large enough?
      double omsq[NUM_PCLS_MOVED_AT_A_TIME] __attribute__((aligned(64)));
      double denom[NUM_PCLS_MOVED_AT_A_TIME] __attribute__((aligned(64)));
      for(int i=0; i<NUM_PCLS_MOVED_AT_A_TIME;i++)
      {
        omsq[i] = Om[i][0] * Om[i][0]
                + Om[i][1] * Om[i][1]
                + Om[i][2] * Om[i][2];
        denom[i] = 1.0 / (1.0 + omsq[i]);
      }
      // solve the position equation
      double ut[NUM_PCLS_MOVED_AT_A_TIME][3] __attribute__((aligned(64)));
      for(int i=0; i<NUM_PCLS_MOVED_AT_A_TIME;i++)
      for(int j=0;j<3;j++)
      {
        ut[i][j] = uorig[i][j] + qdto2mc * E[i][j];
      }
      double udotOm[NUM_PCLS_MOVED_AT_A_TIME] __attribute__((aligned(64)));
      for(int i=0; i<NUM_PCLS_MOVED_AT_A_TIME;i++)
      {
        udotOm[i] = ut[i][0] * Om[i][0]
                  + ut[i][1] * Om[i][1]
                  + ut[i][2] * Om[i][2];
      }
      // solve the velocity equation 
      for(int i=0;i<NUM_PCLS_MOVED_AT_A_TIME;i++)
      {
        // these cross-products might not vectorize so well...
        uavg[i][0] = (ut[i][0] + (ut[i][1] * Om[i][2] - ut[i][2] * Om[i][1] + udotOm[i] * Om[i][0])) * denom[i];
        uavg[i][1] = (ut[i][1] + (ut[i][2] * Om[i][0] - ut[i][0] * Om[i][2] + udotOm[i] * Om[i][1])) * denom[i];
        uavg[i][2] = (ut[i][2] + (ut[i][0] * Om[i][1] - ut[i][1] * Om[i][0] + udotOm[i] * Om[i][2])) * denom[i];
      }
      // update average position
      // #pragma simd collapse(2)
      for(int i=0;i<NUM_PCLS_MOVED_AT_A_TIME;i++)
      for(int j=0;j<3;j++)
      {
        xavg[i][j] = xorig[i][j] + uavg[i][j] * dto2;
      }
    } // end of iteration
    // update the final position and velocity (scatter)
    for(int i=0;i<NUM_PCLS_MOVED_AT_A_TIME;i++)
    for(int j=0;j<3;j++)
    {
      pcl[i]->set_x(j, xorig[i][j] + uavg[i][j] * dt);
      pcl[i]->set_u(j, 2.*uavg[i][j] - uorig[i][j]);
    }
  }
  #pragma omp master
  { timeTasks_end_task(TimeTasks::MOVER_PCL_MOVING); }
}
}

/** relativistic mover with a Predictor-Corrector scheme */
int Particles3D::mover_relativistic(Field * EMf)
{
  eprintf("Mover_relativistic not implemented");
  return (0);
}

//! End of IMM

//! ============================================================================= !//

inline void Particles3D::populate_cell_with_particles(int i, int j, int k, double q_per_particle,
                                                    double dx_per_pcl, double dy_per_pcl, double dz_per_pcl)
{
  const double cell_low_x = grid->getXN(i,j,k);
  const double cell_low_y = grid->getYN(i,j,k);
  const double cell_low_z = grid->getZN(i,j,k);
  for (int ii=0; ii < npcelx; ii++)
  for (int jj=0; jj < npcely; jj++)
  for (int kk=0; kk < npcelz; kk++)
  {
    double u,v,w,q,x,y,z;
    sample_maxwellian(
      u,v,w,
      uth, vth, wth,
      u0, v0, w0);
    x = (ii + sample_u_double())*dx_per_pcl + cell_low_x;
    y = (jj + sample_u_double())*dy_per_pcl + cell_low_y;
    z = (kk + sample_u_double())*dz_per_pcl + cell_low_z;
    create_new_particle(u,v,w,q_per_particle,x,y,z);
  }
}

// This could be generalized to use fluid moments
// to generate particles.
//
void Particles3D::repopulate_particles()
{
  using namespace BCparticles;

  // if this is not a boundary process then there is nothing to do
  if(!vct->isBoundaryProcess_P()) return;

  // if there are no reemission boundaries then no one has anything to do
  const bool repop_bndry_in_X = !vct->getPERIODICX_P() &&
        (bcPfaceXleft == REEMISSION || bcPfaceXright == REEMISSION);
  const bool repop_bndry_in_Y = !vct->getPERIODICY_P() &&
        (bcPfaceYleft == REEMISSION || bcPfaceYright == REEMISSION);
  const bool repop_bndry_in_Z = !vct->getPERIODICZ_P() &&
        (bcPfaceZleft == REEMISSION || bcPfaceZright == REEMISSION);
  const bool repopulation_boundary_exists =
        repop_bndry_in_X || repop_bndry_in_Y || repop_bndry_in_Z;

  if(!repopulation_boundary_exists) return;


  // boundaries to repopulate
  //
  const bool repopulateXleft = (vct->noXleftNeighbor_P() && bcPfaceXleft == REEMISSION);
  const bool repopulateYleft = (vct->noYleftNeighbor_P() && bcPfaceYleft == REEMISSION);
  const bool repopulateZleft = (vct->noZleftNeighbor_P() && bcPfaceZleft == REEMISSION);
  const bool repopulateXrght = (vct->noXrghtNeighbor_P() && bcPfaceXright == REEMISSION);
  const bool repopulateYrght = (vct->noYrghtNeighbor_P() && bcPfaceYright == REEMISSION);
  const bool repopulateZrght = (vct->noZrghtNeighbor_P() && bcPfaceZright == REEMISSION);
  const bool do_repopulate = 
       repopulateXleft || repopulateYleft || repopulateZleft
    || repopulateXrght || repopulateYrght || repopulateZrght;
  // if this process has no reemission boundaries then there is nothing to do
  if(!do_repopulate)
    return;

  // there are better ways to obtain these values...
  //
  double  FourPI =16*atan(1.0);
  const double q_per_particle
    = (qom/fabs(qom))*(Ninj/FourPI/npcel)*(1.0/grid->getInvVOL());

  const int nxc = grid->getNXC();
  const int nyc = grid->getNYC(); const int nzc = grid->getNZC();
  // number of cell layers to repopulate at boundary
  const int num_layers = 3;
  const double xLow = num_layers*dx;
  const double yLow = num_layers*dy;
  const double zLow = num_layers*dz;
  const double xHgh = Lx-xLow;
  const double yHgh = Ly-yLow;
  const double zHgh = Lz-zLow;
  if(repopulateXleft || repopulateXrght) assert_gt(nxc, 2*num_layers);
  if(repopulateYleft || repopulateYrght) assert_gt(nyc, 2*num_layers);
  if(repopulateZleft || repopulateZrght) assert_gt(nzc, 2*num_layers);

  // delete particles in repopulation layers
  //
  const int nop_orig = getNOP();
  int pidx = 0;
  while(pidx < getNOP())
  {
    SpeciesParticle& pcl = _pcls[pidx];
    // determine whether to delete the particle
    const bool delete_pcl =
      (repopulateXleft && pcl.get_x() < xLow) ||
      (repopulateYleft && pcl.get_y() < yLow) ||
      (repopulateZleft && pcl.get_z() < zLow) ||
      (repopulateXrght && pcl.get_x() > xHgh) ||
      (repopulateYrght && pcl.get_y() > yHgh) ||
      (repopulateZrght && pcl.get_z() > zHgh);
    if(delete_pcl)
      delete_particle(pidx);
    else
      pidx++;
  }
  const int nop_remaining = getNOP();

  const double dx_per_pcl = dx/npcelx;
  const double dy_per_pcl = dy/npcely;
  const double dz_per_pcl = dz/npcelz;

  // starting coordinate of upper layer
  const int upXstart = nxc-1-num_layers;
  const int upYstart = nyc-1-num_layers;
  const int upZstart = nzc-1-num_layers;

  // inject new particles.
  //
  {
    // we shrink the imagined boundaries of the array as we go along to ensure
    // that we never inject particles twice in a single mesh cell.
    //
    // initialize imagined boundaries to full subdomain excluding ghost cells.
    //
    int xbeg = 1;
    int xend = nxc-2;
    int ybeg = 1;
    int yend = nyc-2;
    int zbeg = 1;
    int zend = nzc-2;
    if (repopulateXleft)
    {
      //cout << "*** Repopulate Xleft species " << ns << " ***" << endl;
      for (int i=1; i<= num_layers; i++)
      for (int j=ybeg; j<=yend; j++)
      for (int k=zbeg; k<=zend; k++)
      {
        populate_cell_with_particles(i,j,k,q_per_particle,
          dx_per_pcl, dy_per_pcl, dz_per_pcl);
      }
      // these have all been filled, so never touch them again.
      xbeg += num_layers;
    }
    if (repopulateXrght)
    {      
      cout << "*** Repopulate Xright species " << ns << " ***" << endl;
      for (int i=upXstart; i<=xend; i++)
      for (int j=ybeg; j<=yend; j++)
      for (int k=zbeg; k<=zend; k++)
      {
        populate_cell_with_particles(i,j,k,q_per_particle,
          dx_per_pcl, dy_per_pcl, dz_per_pcl);
      }
      // these have all been filled, so never touch them again.
      xend -= num_layers;
    }
    if (repopulateYleft)
    {     
      // cout << "*** Repopulate Yleft species " << ns << " ***" << endl;
      for (int i=xbeg; i<=xend; i++)
      for (int j=1; j<=num_layers; j++)
      for (int k=zbeg; k<=zend; k++)
      {
        populate_cell_with_particles(i,j,k,q_per_particle,
          dx_per_pcl, dy_per_pcl, dz_per_pcl);
      }
      // these have all been filled, so never touch them again.
      ybeg += num_layers;
    }
    if (repopulateYrght)
    {     
      // cout << "*** Repopulate Yright species " << ns << " ***" << endl;
      for (int i=xbeg; i<=xend; i++)
      for (int j=upYstart; j<=yend; j++)
      for (int k=zbeg; k<=zend; k++)
      {
        populate_cell_with_particles(i,j,k,q_per_particle,
          dx_per_pcl, dy_per_pcl, dz_per_pcl);
      }
      // these have all been filled, so never touch them again.
      yend -= num_layers;
    }
    if (repopulateZleft)
    {   
      //   cout << "*** Repopulate Zleft species " << ns << " ***" << endl;
      for (int i=xbeg; i<=xend; i++)
      for (int j=ybeg; j<=yend; j++)
      for (int k=1; k<=num_layers; k++)
      {
        populate_cell_with_particles(i,j,k,q_per_particle,
          dx_per_pcl, dy_per_pcl, dz_per_pcl);
      }
    }
    if (repopulateZrght)
    {   
      //   cout << "*** Repopulate Zright species " << ns << " ***" << endl;
      for (int i=xbeg; i<=xend; i++)
      for (int j=ybeg; j<=yend; j++)
      for (int k=upZstart; k<=zend; k++)
      {
        populate_cell_with_particles(i,j,k,q_per_particle,
          dx_per_pcl, dy_per_pcl, dz_per_pcl);
      }
    }
  }
  const int nop_final = getNOP();
  const int nop_deleted = nop_orig - nop_remaining;
  const int nop_created = nop_final - nop_remaining;

  //dprintf("change in # particles: %d - %d + %d = %d",nop_orig, nop_deleted, nop_created, nop_final);

  //if (vct->getCartesian_rank()==0){
  //  cout << "*** number of particles " << getNOP() << " ***" << endl;
  //}
}

// Open BC for particles: duplicate particles on the boundary,.
// shift outside the box and update location to test if inside box
// if so, add to particle list
void Particles3D::openbc_particles_inflow()
{
    eprintf("openbc_particles_inflow has not yet been implemented");
}

// Open BC for particles: duplicate particles on the boundary,.
// shift outside the box and update location to test if inside box
// if so, add to particle list
void Particles3D::openbc_particles_outflow()
{
  // if this is not a boundary process then there is nothing to do
  if(!vct->isBoundaryProcess_P()) return;

  //The below is OpenBC outflow for all other boundaries
  using namespace BCparticles;

  const bool openXleft = !vct->getPERIODICX_P() && vct->noXleftNeighbor_P() &&  bcPfaceXleft == OPENBCOut;
  const bool openYleft = !vct->getPERIODICY_P() && vct->noYleftNeighbor_P() &&  bcPfaceYleft == OPENBCOut;
  const bool openZleft = !vct->getPERIODICZ_P() && vct->noZleftNeighbor_P() &&  bcPfaceZleft == OPENBCOut;

  const bool openXright = !vct->getPERIODICX_P() && vct->noXrghtNeighbor_P() && bcPfaceXright == OPENBCOut;
  const bool openYright = !vct->getPERIODICY_P() && vct->noYrghtNeighbor_P() && bcPfaceYright == OPENBCOut;
  const bool openZright = !vct->getPERIODICZ_P() && vct->noZrghtNeighbor_P() && bcPfaceZright == OPENBCOut;

  if(!openXleft && !openYleft && !openZleft && !openXright && !openYright && !openZright)  return;

  const int num_layers = 3;
  assert_gt(nxc-2, (openXleft+openXright)*num_layers); //excluding 2 ghost cells, #of cells should be larger than total # of openBC layers
  assert_gt(nyc-2, (openYleft+openYright)*num_layers);
  assert_gt(nzc-2, (openZleft+openZright)*num_layers);

  const double xLow = num_layers*dx;
  const double yLow = num_layers*dy;
  const double zLow = num_layers*dz;
  const double xHgh = Lx-xLow;
  const double yHgh = Ly-yLow;
  const double zHgh = Lz-zLow;

  const bool   apply_openBC[6]    = {openXleft, openXright,openYleft, openYright,openZleft, openZright};
  const double delete_boundary[6] = {0, Lx,0, Ly,0, Lz};
  const double open_boundary[6]   = {xLow, xHgh,yLow, yHgh,zLow, zHgh};

  const int nop_orig = getNOP();
  const int capacity_out = roundup_to_multiple(nop_orig*0.1,DVECWIDTH);
  vector_SpeciesParticle injpcls(capacity_out);


  for(int dir_cnt=0;dir_cnt<6;dir_cnt++){

    if(apply_openBC[dir_cnt]){
    	  //dprintf( "*** OpenBC for Direction %d on particle species %d",dir_cnt, ns);

		  int pidx = 0;
		  int direction = dir_cnt/2;
		  double delbry  = delete_boundary[dir_cnt];
		  double openbry = open_boundary[dir_cnt];
		  double location;
		  while(pidx < getNOP())
		  {
		     SpeciesParticle& pcl = _pcls[pidx];
		     location = pcl.get_x(direction);

		     // delete the exiting particle if out of box on the direction of OpenBC
		     if((dir_cnt%2==0 && location<delbry) ||(dir_cnt%2==1 && location>delbry))
		       delete_particle(pidx);
		     else{
		       pidx++;

		       //copy the particle within open boundary to inject particle list if their shifted location after 1 time step is within simulation box
		       if ((dir_cnt%2==0 && location<openbry) ||(dir_cnt%2==1 && location>openbry)){
		    	   double injx=pcl.get_x(0), injy=pcl.get_x(1), injz=pcl.get_x(2);
		    	   double inju=pcl.get_u(0), injv=pcl.get_u(1), injw=pcl.get_u(2);
		    	   double injq=pcl.get_q();

		    	   //shift 3 layers out, not mirror
		    	   if(direction == 0) injx = (dir_cnt%2==0) ?(injx-xLow):(injx+xLow);
		    	   if(direction == 1) injy = (dir_cnt%2==0) ?(injy-yLow):(injy+yLow);
		    	   if(direction == 2) injz = (dir_cnt%2==0) ?(injz-zLow):(injz+zLow);

		    	   injx = injx + inju*dt;
		    	   injy = injy + injv*dt;
		    	   injz = injz + injw*dt;

		    	   //Add particle if it enter that sub-domain or the domain box?
		    	   //assume create particle as long as it enters the domain box
		    	   if(injx>0 && injx<Lx && injy>0 && injy<Ly && injz>0 && injz<Lz){
		    		    injpcls.push_back(SpeciesParticle(inju,injv,injw,injq,injx,injy,injz,pclIDgenerator.generateID()));
		    	   }
		       }
		     }
		   }
	  }
  }

  //const int nop_remaining = getNOP();
  //const int nop_deleted = nop_orig - nop_remaining;
  const int nop_created = injpcls.size();

  //dprintf("change in # particles: %d - %d + %d = %d",nop_orig, nop_deleted, nop_created, nop_remaining);

  for(int outId=0;outId<nop_created;outId++)
	  _pcls.push_back(injpcls[outId]);
}

//Simply delete exiting test particles if openBC
void Particles3D::openbc_delete_testparticles()
{
  // if this is not a boundary process then there is nothing to do
  if(!vct->isBoundaryProcess_P()) return;

  //The below is OpenBC outflow for all other boundaries
  using namespace BCparticles;

  const bool openXleft = !vct->getPERIODICX_P() && vct->noXleftNeighbor_P() &&  (bcPfaceXleft == OPENBCOut||bcPfaceXleft == REEMISSION);
  const bool openYleft = !vct->getPERIODICY_P() && vct->noYleftNeighbor_P() &&  (bcPfaceYleft == OPENBCOut||bcPfaceYleft == REEMISSION);
  const bool openZleft = !vct->getPERIODICZ_P() && vct->noZleftNeighbor_P() &&  (bcPfaceZleft == OPENBCOut||bcPfaceZleft == REEMISSION);

  const bool openXright = !vct->getPERIODICX_P() && vct->noXrghtNeighbor_P() && (bcPfaceXright == OPENBCOut||bcPfaceXright == REEMISSION);
  const bool openYright = !vct->getPERIODICY_P() && vct->noYrghtNeighbor_P() && (bcPfaceYright == OPENBCOut||bcPfaceYright == REEMISSION);
  const bool openZright = !vct->getPERIODICZ_P() && vct->noZrghtNeighbor_P() && (bcPfaceZright == OPENBCOut||bcPfaceZright == REEMISSION);
  //dprintf( "Entered openbc_delete_testparticles %d %d %d %d %d %d", openXleft,openXright,openYleft,openYright,openZleft,openZright);
  if(!openXleft && !openYleft && !openZleft && !openXright && !openYright && !openZright)  return;

  const bool   apply_openBC[6]    = {openXleft, openXright,openYleft, openYright,openZleft, openZright};
  const double delete_boundary[6] = {0, Lx,0, Ly,0, Lz};

  for(int dir_cnt=0;dir_cnt<6;dir_cnt++){

    if(apply_openBC[dir_cnt]){

		  int pidx = 0;
		  int delnop = 0;
		  int direction = dir_cnt/2;
		  double delbry  = delete_boundary[dir_cnt];
		  double location;
		  while(pidx < getNOP())
		  {
		     SpeciesParticle& pcl = _pcls[pidx];
		     location = pcl.get_x(direction);

		     // delete the exiting particle if out of box on the direction of OpenBC
		     if((dir_cnt%2==0 && location<delbry) ||(dir_cnt%2==1 && location>delbry)){
		    	 delete_particle(pidx);
		    	 delnop++;
		     }
		     else {
		    	 pidx ++;
		     }
		   }
		  dprintf( "*** Delete %d Test Particle for OpenBC for Direction %d on particle species %d",delnop, dir_cnt, ns);
	  }

  }

}

/** Linear delta f for bi-maxwellian plasma */
double Particles3D::delta_f(double u, double v, double w, double x, double y, double kx, double ky, double omega_re, double omega_i, double Ex_mod, double Ex_phase, double Ey_mod, double Ey_phase, double Ez_mod, double Ez_phase, double theta, Field * EMf) {
  const complex < double >I(0.0, 1.0);
  const double vperp = sqrt(v * v + w * w);
  const double vpar = u;
  const double kpar = kx;
  double kperp;
  if (ky == 0.0)                // because this formula is not valid for exactly parallel
    kperp = 1e-9;
  else
    kperp = ky;
  const double om_c = qom / c * sqrt(EMf->getBx(1, 1, 0) * EMf->getBx(1, 1, 0) + EMf->getBy(1, 1, 0) * EMf->getBy(1, 1, 0)) / 2 / M_PI;
  const double phi = atan2(w, v);
  const double lambda = kperp * vperp / om_c;
  const complex < double >omega(omega_re, omega_i);

  const int lmax = 5;           // sum from -lmax to lmax

  double bessel_Jn_array[lmax + 2];
  double bessel_Jn_prime_array[lmax + 1];
  complex < double >a1[2 * lmax + 1], a2[2 * lmax + 1], a3[2 * lmax + 1];
  complex < double >factor, deltaf;

  // rotation of x,y
  double temp;
  temp = x;
  x = x * cos(theta) - y * sin(theta);
  y = temp * sin(theta) + y * cos(theta);


  /** for compilation issues comment this part: PUT in the math stuff */
  // calc_bessel_Jn_seq(lambda, lmax, bessel_Jn_array, bessel_Jn_prime_array);
  factor = (kpar * vperp / omega * df0_dvpar(vpar, vperp) + (1.0 - (kpar * vpar / omega)) * df0_dvperp(vpar, vperp));
  for (int l = -lmax; l < 0; l++) {  // negative index
    a1[l + lmax] = factor / lambda * pow(-1.0, -l) * bessel_Jn_array[-l];
    a1[l + lmax] *= (double) l;
    a2[l + lmax] = factor * I * 0.5 * pow(-1.0, -l) * (bessel_Jn_array[-l - 1] - bessel_Jn_array[-l + 1]);
    a3[l + lmax] = kperp / omega * (vpar * df0_dvperp(vpar, vperp) - vperp * df0_dvpar(vpar, vperp)) / lambda * pow(-1.0, -l) * bessel_Jn_array[-l];
    a3[l + lmax] *= (double) l;
    a3[l + lmax] += df0_dvpar(vpar, vperp) * pow(-1.0, -l) * bessel_Jn_array[-l];
  }

  for (int l = 0; l < lmax + 1; l++) { // positive index
    a1[l + lmax] = factor / lambda * bessel_Jn_array[l];
    a1[l + lmax] *= (double) l;
    a2[l + lmax] = factor * I * bessel_Jn_prime_array[l];
    a3[l + lmax] = kperp / omega * (vpar * df0_dvperp(vpar, vperp) - vperp * df0_dvpar(vpar, vperp)) / lambda * bessel_Jn_array[l];
    a3[l + lmax] *= (double) l;
    a3[l + lmax] += df0_dvpar(vpar, vperp) * bessel_Jn_array[l];
  }

  //deltaf = (0.0, 0.0);
  for (int l = -lmax; l < lmax + 1; l++) {
    deltaf += (a3[l + lmax] * Ex_mod * exp(I * Ex_phase) + a1[l + lmax] * Ey_mod * exp(I * Ey_phase) + a2[l + lmax] * Ez_mod * exp(I * Ez_phase)) / (kpar * vpar + l * om_c - omega) * exp(-I * phi * (double) l);
  }
  deltaf *= I * qom * exp(I * lambda * sin(phi)) * exp(I * (2 * M_PI * kx * x + 2 * M_PI * ky * y));

  return (real(deltaf));
}

double Particles3D::df0_dvpar(double vpar, double vperp) {
  double result;
  result = -2 * (vpar - u0) / uth / uth * exp(-(vperp * vperp / vth / vth + (vpar - u0) * (vpar - u0) / uth / uth));
  result *= 3.92e6 / pow(M_PI, 3 / 2) / vth / vth / uth;
  return (result);
}

double Particles3D::df0_dvperp(double vpar, double vperp) {
  double result;
  result = -2 * (vperp) / vth / vth * exp(-(vperp * vperp / vth / vth + (vpar - u0) * (vpar - u0) / uth / uth));
  result *= 3.92e6 / pow(M_PI, 3 / 2) / vth / vth / uth;
  return (result);
}

double Particles3D::f0(double vpar, double vperp) {
  double result;
  result = exp(-(vperp * vperp / vth / vth + (vpar - u0) * (vpar - u0) / uth / uth));
  result *= 3.92e6 / pow(M_PI, 3 / 2) / vth / vth / uth;
  return (result);
}

void Particles3D::RotatePlaneXY(double theta) {
  double temp, temp2;
  for (int s = 0; s < getNOP(); s++) {
    temp = u[s];
    temp2 = v[s];
    u[s] = temp * cos(theta) + v[s] * sin(theta);
    v[s] = -temp * sin(theta) + temp2 * cos(theta);
  }
}

/*! Delete the particles inside the sphere with radius R and center x_center y_center and return the total charge removed */
double Particles3D::deleteParticlesInsideSphere(double R, double x_center, double y_center, double z_center)
{
  int pidx = 0;
  double Q_removed=0.;
  while (pidx < _pcls.size())
  {
    SpeciesParticle& pcl = _pcls[pidx];
    double xd = pcl.get_x() - x_center;
    double yd = pcl.get_y() - y_center;
    double zd = pcl.get_z() - z_center;

    if ( (xd*xd+yd*yd+zd*zd) < R*R ){
      Q_removed += pcl.get_q();
      delete_particle(pidx);
    } else {
      pidx++;
    }
  }
  return(Q_removed);
}

double Particles3D::deleteParticlesInsideSphere2DPlaneXZ(double R, double x_center, double z_center)
{
  int pidx = 0;
  double Q_removed=0.;
  while (pidx < _pcls.size())
  {
    SpeciesParticle& pcl = _pcls[pidx];
    double xd = pcl.get_x() - x_center;
    double zd = pcl.get_z() - z_center;

    if ( (xd*xd+zd*zd) < R*R ){
      Q_removed += pcl.get_q();
      delete_particle(pidx);
    } else {
      pidx++;
    }
  }
  return(Q_removed);
}
