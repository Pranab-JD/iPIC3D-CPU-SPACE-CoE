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
  Particles3D.h  -  Class for particles of the same species, in a 2D space and 3 component velocity
  -------------------
developers: Stefano Markidis, Enrico Camporeale, Giovanni Lapenta, David Burgess
 ********************************************************************************************/

#ifndef Part2D_H
#define Part2D_H

#include "Particles3Dcomm.h"
//#include "TimeTasks.h"

/**
 * 
 * Class for particles of the same species, in a 2D space and 3 component velocity
 * 
 * @date Fri Jun 4 2007
 * @author Stefano Markidis, Giovanni Lapenta
 * @version 2.0
 *
 */
class Particles3D:public Particles3Dcomm 
{
    public:
    
    //! Constructor !//
    Particles3D(int species, CollectiveIO *col, VirtualTopology3D *vct, Grid * grid): Particles3Dcomm(species, col, vct, grid) {}
    
    //! Destructor !//
    ~Particles3D(){}
    
    //? ============================================================================ ? //
    
    //? ---------- Initial particle distributions (Non Relativistic) ---------- ?//

    void maxwellian(Field * EMf);
    void maxwellian_KHI_FLR(Field* EMf);
    void uniform_background(Field * EMf);
    void maxwellian_Double_Harris(Field * EMf);
    void maxwellian_Double_Harris_Hump(Field * EMf);

    //? ---------- Initial particle distributions (Relativistic) ---------- ?//

    void Maxwell_Juttner(Field * EMf);
    void Shock1D(Field * EMf);
    void Shock1D_DoublePiston(Field * EMf);
    void Relativistic_Double_Harris_pairs(Field * EMf);
    void Relativistic_Double_Harris_ion_electron(Field * EMf);

    //? ============================================================================ ? //

    /** mover with the esplicit non relativistic scheme */
    void mover_explicit(Field * EMf);
   
    /** mover with a Predictor-Corrector Scheme */
    void mover_PC(Field * EMf);
    
    //* ECSIM (RelSIM) - velocity and position update
    void ECSIM_velocity(Field * EMf);
    void RelSIM_velocity(Field * EMf);
    void ECSIM_position(Field * EMf);

    //* ECSIM - moments
    void computeMoments(Field * EMf);

    //* ECSIM - supplementary moments
    void computeCharge(Field * EMf);
    void compute_supplementary_moments(Field * EMf);

    //* ECSIM - fix dimension of the problem
    void fixPosition();

    /** array-of-structs version of mover_PC */
    void mover_PC_AoS(Field * EMf);
    /** Relativistic array-of-structs version of mover_PC with adaptive Subcycling and PC*/
    void mover_PC_AoS_Relativistic(Field * EMf);
    /* vectorized version of previous */
    void mover_PC_AoS_vec(Field * EMf);
    /* mic particle mover */
    void mover_PC_AoS_vec_intr(Field * EMf);
    /* this computes garbage */
    void mover_PC_AoS_vec_onesort(Field * EMf);
    /** vectorized version of mover_PC **/
    void mover_PC_vectorized(Field * EMf);
    /** relativistic mover with a Predictor-Corrector scheme */
    int mover_relativistic(Field * EMf);
   
    private:
    /** repopulate particles in a single cell */
    void populate_cell_with_particles(int i, int j, int k, double q, double dx_per_pcl, double dy_per_pcl, double dz_per_pcl);
   
    public:
    /** repopulate particles in boundary layer */
    void repopulate_particles();
    /*! Delete the particles inside the sphere with radius R and center x_center y_center and return the total charge removed */
    double deleteParticlesInsideSphere(double R, double x_center, double y_center, double z_center);
    double deleteParticlesInsideSphere2DPlaneXZ(double R, double x_center, double z_center);
    /**Particles Open Boundary */
    void openbc_particles_outflow();
    void openbc_delete_testparticles();
    void openbc_particles_inflow();

    double delta_f(double u, double v, double w, double x, double y, double kx, double ky, double omega_re, double omega_i, double Ex_ampl, double Ex_phase, double Ey_ampl, double Ey_phase, double Ez_ampl, double Ez_phase, double theta, Field * EMf);
    double df0_dvpar(double vpar, double vperp);
    double df0_dvperp(double vpar, double vperp);
    double f0(double vpar, double vperp);
    void RotatePlaneXY(double theta);
};

#endif
