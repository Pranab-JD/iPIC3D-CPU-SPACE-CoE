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

/***************************************************************************
  Collective.h  -  Stefano Markidis, Giovanni Lapenta
  -------------------------------------------------------------------------- */


/*! Collective properties. Input physical parameters for the simulation.  Use ConfigFile to parse the input file @date Wed Jun 8 2011 @par Copyright: (C) 2011 K.U. LEUVEN @author Pierre Henri, Stefano Markidis @version 1.0 */

#ifndef Collective_H
#define Collective_H

#ifdef BATSRUS
#include "InterfaceFluid.h"
#endif
#include <iostream>
#include <string>
#include <memory>
#include "VCtopology3D.h"
#include "Grid3DCU.h"
#include "aligned_vector.h"

class ConfigFile;
using namespace std;

class Collective
#ifdef BATSRUS
: public InterfaceFluid
#endif
{
  private:
    enum Enum{
      thedefault=0,
      initial,
      final,
      // used by ImplSusceptMode
      explPredict,
      implPredict,
      NUMBER_OF_ENUMS, // this must be last
      INVALID_ENUM
    };
    int read_enum_parameter(const char* option_name, const char* default_value,
      const ConfigFile& config);
  public:
    static const char* get_name_of_enum(int in);
  public:
    /*! constructor: initialize physical parameters with values */
    Collective(int argc, char **argv);
    /*! read input file */
    void ReadInput(string inputfile);
    /*! read the restart input file from HDF5 */
    int ReadRestart(string inputfile);

    void trim_conserved_quantities_file(const std::string& filename, int restart_cycle);

    void read_field_restart(const VCtopology3D* vct,const Grid* grid, arr3_double Bxn, arr3_double Byn, arr3_double Bzn,
                            arr3_double Bxc, arr3_double Byc, arr3_double Bzc,
    						arr3_double Ex, arr3_double Ey, arr3_double Ez,
                            arr3_double rhoc_avg, arr3_double divE_avg, int ns)const;

    void read_particles_restart(const VCtopology3D* vct,int species_number,vector_double& u,vector_double& v,vector_double& w,
    							vector_double& q,vector_double& x,vector_double& y,vector_double& z,vector_double& t)const;

    void init_derived_parameters();
    /*! Print physical parameters */
    void Print();
    /*! save setting in a file */
    void save();

    // accessors
    int getDim()                        const{ return (dim); }
    double getLx()                      const{ return (Lx); }
    double getLy()                      const{ return (Ly); }
    double getLz()                      const{ return (Lz); }
    double getx_center()                const{ return (x_center); }
    double gety_center()                const{ return (y_center); }
    double getz_center()                const{ return (z_center); }
    double getL_square()                const{ return (L_square); }
    int getNxc()                        const{ return (nxc); }
    int getNyc()                        const{ return (nyc); }
    int getNzc()                        const{ return (nzc); }
    int getXLEN()                       const{ return (XLEN); }
    int getYLEN()                       const{ return (YLEN); }
    int getZLEN()                       const{ return (ZLEN); }
    bool getPERIODICX()                 const{ return (PERIODICX); }
    bool getPERIODICY()                 const{ return (PERIODICY); }
    bool getPERIODICZ()                 const{ return (PERIODICZ); }
    bool getPERIODICX_P()               const{ return (PERIODICX_P); }
    bool getPERIODICY_P()               const{ return (PERIODICY_P); }
    bool getPERIODICZ_P()               const{ return (PERIODICZ_P); }
    double getDx()                      const{ return (dx); }
    double getDy()                      const{ return (dy); }
    double getDz()                      const{ return (dz); }
    double getC()                       const{ return (c); }
    double getDt()                      const{ return (dt); }
    double getTh()                      const{ return (th); }    
    double getSmooth()                  const{ return (Smooth); }
    int getNcycles()                    const{ return (ncycles); }

    int getNs()                         const{ return (ns); }
    int getNsTestPart()                 const{ return (nstestpart); }
    int getNpcel(int nspecies)          const{ return (npcel[nspecies]); }
    int getNpcelx(int nspecies)         const{ return (npcelx[nspecies]); }
    int getNpcely(int nspecies)         const{ return (npcely[nspecies]); }
    int getNpcelz(int nspecies)         const{ return (npcelz[nspecies]); }

    double getNpMaxNpRatio()            const{ return (NpMaxNpRatio); }
    double getQOM(int nspecies)         const{ return (qom[nspecies]); }
    double getRHOinit(int nspecies)     const{ return (rhoINIT[nspecies]); }
    double getRHOinject(int nspecies)   const{ return(rhoINJECT[nspecies]); }
    double getUth(int nspecies)         const{ return (uth[nspecies]); }
    double getVth(int nspecies)         const{ return (vth[nspecies]); }
    double getWth(int nspecies)         const{ return (wth[nspecies]); }
    double getU0(int nspecies)          const{ return (u0[nspecies]); }
    double getV0(int nspecies)          const{ return (v0[nspecies]); }
    double getW0(int nspecies)          const{ return (w0[nspecies]); }

    double getPitchAngle(int nspecies)  const{ return (pitch_angle[nspecies]); }
    double getEnergy(int nspecies)      const{ return (energy[nspecies]); }

    //? Nonperiodic boundaries
    int getBcPfaceXright()              const{ return (bcPfaceXright); }
    int getBcPfaceXleft()               const{ return (bcPfaceXleft); }
    int getBcPfaceYright()              const{ return (bcPfaceYright); }
    int getBcPfaceYleft()               const{ return (bcPfaceYleft); }
    int getBcPfaceZright()              const{ return (bcPfaceZright); }
    int getBcPfaceZleft()               const{ return (bcPfaceZleft); }
    int getBcPHIfaceXright()            const{ return (bcPHIfaceXright); }
    int getBcPHIfaceXleft()             const{ return (bcPHIfaceXleft); }
    int getBcPHIfaceYright()            const{ return (bcPHIfaceYright); }
    int getBcPHIfaceYleft()             const{ return (bcPHIfaceYleft); }
    int getBcPHIfaceZright()            const{ return (bcPHIfaceZright); }
    int getBcPHIfaceZleft()             const{ return (bcPHIfaceZleft); }
    int getBcEMfaceXright()             const{ return (bcEMfaceXright); }
    int getBcEMfaceXleft()              const{ return (bcEMfaceXleft); }
    int getBcEMfaceYright()             const{ return (bcEMfaceYright); }
    int getBcEMfaceYleft()              const{ return (bcEMfaceYleft); }
    int getBcEMfaceZright()             const{ return (bcEMfaceZright); }
    int getBcEMfaceZleft()              const{ return (bcEMfaceZleft); }

    double getB0x()                     const{ return (B0x); }
    double getB0y()                     const{ return (B0y); }
    double getB0z()                     const{ return (B0z); }
    double getB1x()                     const{ return (B1x); }
    double getB1y()                     const{ return (B1y); }
    double getB1z()                     const{ return (B1z); }

    int getRestart_status()             const{ return (restart_status); }
    string getSaveDirName()             const{ return (SaveDirName); }
    string getRestartDirName()          const{ return (RestartDirName); }
    string getinputfile()               const{ return (inputfile); }
    string getCase()                    const{ return (Case); }
    string getSimName()                 const{ return (SimName); }
    string getWriteMethod()             const{ return (wmethod); }
    string getFieldOutputTag()          const{ return FieldOutputTag;}
    string getMomentsOutputTag()        const{ return MomentsOutputTag;}
    string getPclOutputTag()            const{ return ParticlesOutputTag;}
    string getPclDSOutputTag()          const{ return ParticlesDownsampleOutputTag;}
    string get_output_data_precision()  const{ return output_data_precision; }

    double getVinj()                    const{ return (Vinj); }
    double getGMREStol()                const{ return (GMREStol); }
    int getLast_cycle()                 const{ return (last_cycle); }
    int getNiterMover()                 const{ return (NiterMover); }
    int getFieldOutputCycle()           const{ return (FieldOutputCycle); }
    int getParticlesOutputCycle()       const{ return (ParticlesOutputCycle); }
    int getTestParticlesOutputCycle()   const{ return (TestParticlesOutputCycle); }
    int getRestartOutputCycle()         const{ return (RestartOutputCycle); }
    int getDiagnosticsOutputCycle()     const{ return (DiagnosticsOutputCycle); }
    int getParticlesDownsampleOutputCycle()    const{ return (ParticlesDownsampleOutputCycle); }
    int getParticlesDownsampleFactor()  const{ return (ParticlesDownsampleFactor); }
    
    bool getCallFinalize()              const{ return (CallFinalize); }
    bool particle_output_is_off()       const;
    bool DS_particle_output_is_off()    const;
    bool testparticle_output_is_off()   const;
    bool field_output_is_off()          const;
    bool getSaveHeatFluxTensor()        const { return (SaveHeatFluxTensor); }

    bool getAddExternalCurlB()          const { return AddExternalCurlB; }
    bool getAddExternalCurlE()          const { return AddExternalCurlE; }
    int getZeroCurrent()                      { return zeroCurrent; }

    int getSmoothCycle()                const { return SmoothCycle; }
    int getNumSmoothings()              const { return num_smoothings; }
    int getCurrentCycle()               const { return CurrentCycle; }
    void setCurrentCycle(int cycle)           { CurrentCycle = cycle; }

    double getPoissonMAdiv()                  { return PoissonMAdiv; }
    double getPoissonMAres()                  { return PoissonMAres; }
    double getPoissonMArho()                  { return PoissonMArho; }

    //* Particle distribution parameters
    int getParticleDistOutputCycle()          { return ParticleDistOutputCycle; }
    int getParticleDistBins()                 { return ParticleDistBins; }
    double getParticleDistMinVelocity()       { return ParticleDistMinVelocity; }
    double getParticleDistMaxVelocity()       { return ParticleDistMaxVelocity; }

    bool getRelativistic()              const { return Relativistic; }
    string getRelativisticPusher()      const { return Relativistic_pusher; }   

    int getNparam()                     const { return (nparam); }
    double getInputParam(int ip)        const { return (input_param[ip]); }
    
    /*! Boundary condition selection for BCFace for the electric field components */
    int bcEx[6], bcEy[6], bcEz[6];
    /*! Boundary condition selection for BCFace for the magnetic field components */
    int bcBx[6], bcBy[6], bcBz[6];
    string getParaviewScriptPath()      const { return ParaviewScriptPath; }

  private:
    /*! inputfile */
    string inputfile;
    string ParaviewScriptPath;

    double c; double fourpi; double dt; double th; //! second-order for th=1/2; stable for 1/2 <= th <= 1

    int ImplSusceptMode; // "initial" (default), "explPredict", "implPredict"
    
    //* Smoothing parameters
    double Smooth; int num_smoothings; int SmoothCycle;

    int CurrentCycle;
    int zeroCurrent;

    /*! number of time cycles */
    int ncycles;
    /*! physical space dimensions */
    int dim;
    /*! simulation box length - X direction */
    double Lx;
    /*! simulation box length - Y direction */
    double Ly;
    /*! simulation box length - Z direction */
    double Lz;
    /*! object center - X direction */
    double x_center;
    /*! object center - Y direction */
    double y_center;
    /*! object center - Z direction */
    double z_center;
    /*! object size - assuming a cubic box */
    double L_square;
    // number of cells per direction of problem domain
    int nxc;
    int nyc;
    int nzc;
    /*! grid spacing - X direction */
    double dx;
    /*! grid spacing - Y direction */
    double dy;
    /*! grid spacing - Z direction */
    double dz;
    /*! number of MPI subdomains in each direction */
    int XLEN;
    int YLEN;
    int ZLEN;
    /*! periodicity in each direction */
    bool PERIODICX;
    bool PERIODICY;
    bool PERIODICZ;
    /*! Particle periodicity in each direction */
    bool PERIODICX_P;
    bool PERIODICY_P;
    bool PERIODICZ_P;

    /*! number of species */
    int ns;
    /*! number of test particle species */
    int nstestpart;
    /*! number of particles per cell */
    std::unique_ptr<int[]> npcel;
    /*! number of particles per cell - X direction */
    std::unique_ptr<int[]> npcelx;
    /*! number of particles per cell - Y direction */
    std::unique_ptr<int[]> npcely;
    /*! number of particles per cell - Z direction */
    std::unique_ptr<int[]> npcelz;
    // either make these of longid type or do not declare them.
    //int *np; /*! number of particles array for different species */
    //int *npMax; /*! maximum number of particles array for different species */
    /*! max number of particles */
    double NpMaxNpRatio;
    /*! charge to mass ratio array for different species */
    std::unique_ptr<double[]> qom;
    /*! charge to mass ratio array for different species */
    std::unique_ptr<double[]> rhoINIT;
    /*! density of injection */
    std::unique_ptr<double[]> rhoINJECT;
    /*! thermal velocity - Direction X */
    std::unique_ptr<double[]> uth;
    /*! thermal velocity - Direction Y */
    std::unique_ptr<double[]> vth;
    /*! thermal velocity - Direction Z */
    std::unique_ptr<double[]> wth;
    /*! Drift velocity - Direction X */
    std::unique_ptr<double[]> u0;
    /*! Drift velocity - Direction Y */
    std::unique_ptr<double[]> v0;
    /*! Drift velocity - Direction Z */
    std::unique_ptr<double[]> w0;

    /*! Pitch Angle for Test Particles */
    std::unique_ptr<double[]> pitch_angle;
    /*! Energy for Test Particles */
    std::unique_ptr<double[]> energy;

    /*! Case type */
    string Case;
    /*! Output writing method */
    string wmethod;
    /*! Simulation name */
    string SimName;

    /*! TrackParticleID */
    //bool *TrackParticleID;
    /*! SaveDirName */
    string SaveDirName;
    /*! RestartDirName */
    string RestartDirName;
    /*! restart_status 0 --> no restart; 1--> restart, create new; 2--> restart, append; */
    int restart_status;
    /*! last cycle */
    int last_cycle;

    /*! Boundary condition on particles 0 = exit 1 = perfect mirror 2 = riemission */
    /*! Boundary Condition Particles: FaceXright */
    int bcPfaceXright;
    /*! Boundary Condition Particles: FaceXleft */
    int bcPfaceXleft;
    /*! Boundary Condition Particles: FaceYright */
    int bcPfaceYright;
    /*! Boundary Condition Particles: FaceYleft */
    int bcPfaceYleft;
    /*! Boundary Condition Particles: FaceYright */
    int bcPfaceZright;
    /*! Boundary Condition Particles: FaceYleft */
    int bcPfaceZleft;


    /*! Field Boundary Condition 0 = Dirichlet Boundary Condition: specifies the valueto take pn the boundary of the domain 1 = Neumann Boundary Condition: specifies the value of derivative to take on the boundary of the domain 2 = Periodic Condition */
    /*! Boundary Condition Electrostatic Potential: FaceXright */
    int bcPHIfaceXright;
    /*! Boundary Condition Electrostatic Potential:FaceXleft */
    int bcPHIfaceXleft;
    /*! Boundary Condition Electrostatic Potential:FaceYright */
    int bcPHIfaceYright;
    /*! Boundary Condition Electrostatic Potential:FaceYleft */
    int bcPHIfaceYleft;
    /*! Boundary Condition Electrostatic Potential:FaceZright */
    int bcPHIfaceZright;
    /*! Boundary Condition Electrostatic Potential:FaceZleft */
    int bcPHIfaceZleft;

    /*! Boundary Condition EM Field: FaceXright */
    int bcEMfaceXright;
    /*! Boundary Condition EM Field: FaceXleft */
    int bcEMfaceXleft;
    /*! Boundary Condition EM Field: FaceYright */
    int bcEMfaceYright;
    /*! Boundary Condition EM Field: FaceYleft */
    int bcEMfaceYleft;
    /*! Boundary Condition EM Field: FaceZright */
    int bcEMfaceZright;
    /*! Boundary Condition EM Field: FaceZleft */
    int bcEMfaceZleft;

    /*! GEM Challenge parameters */

    /* Amplitude of the field */
    double B0x;
    double B0y;
    double B0z;
    double B1x;
    double B1y;
    double B1z;

    //* RESTART simulations
    bool RESTART1;
    
    //* Restart cycle
    int RestartOutputCycle;

    //* Velocity of the injection from the wall
    double Vinj;

    //* Tolerance for GMRES solver
    double GMREStol;

    //* mover predictor correction iteration (not needed for ECSIM)
    int NiterMover;

    //* Output for field
    int FieldOutputCycle;
    string FieldOutputTag;
    string MomentsOutputTag;

    //* Output for particles
    int ParticlesOutputCycle;
    string ParticlesOutputTag;
    string ParticlesDownsampleOutputTag;

    //* Downsamplimg of particles
    int ParticlesDownsampleOutputCycle; int ParticlesDownsampleFactor;

    //* Output for test particles
    int TestParticlesOutputCycle;
    
    //* Test particles are flushed to disk every testPartFlushCycle
    int testPartFlushCycle;
    
    //* Output for diagnostics
    int DiagnosticsOutputCycle;
    
    /*! Call Finalize() at end of program execution (true by default) */
    bool CallFinalize;

    //* Add External CurlB and External CurlE
    bool AddExternalCurlB;
    bool AddExternalCurlE;

    //* RelSIM
    bool Relativistic;

    //* Relativistic particle pusher
    string Relativistic_pusher;

    //* Write heat flux tensor to files
    bool SaveHeatFluxTensor;

    //* Moving average value for rho density (Poisson correction)
    double PoissonMArho, PoissonMAdiv, PoissonMAres;

    //* Custom input parameters
    double *input_param; int nparam;

    //* Output data precision
    string output_data_precision;

    //* Particle distribution parameters
    int ParticleDistOutputCycle; int ParticleDistBins;
    double ParticleDistMinVelocity; double ParticleDistMaxVelocity;
};
typedef Collective CollectiveIO;

#endif
