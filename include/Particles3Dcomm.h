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
  Particles3Dcommcomm.h  -  Class for particles of the same species, in a 2D space and 3component velocity with communications methods
  -------------------
developers: Stefano Markidis, Giovanni Lapenta
 ********************************************************************************************/

#ifndef Part3DCOMM_H
#define Part3DCOMM_H

#include "ipicfwd.h"
#include "Alloc.h"
#include "Particle.h" // for ParticleType
// unfortunately this includes mpi.h, which includes 35000 lines:
#include "BlockCommunicator.h"
#include "aligned_vector.h"
#include "Larray.h"
#include "IDgenerator.h"
#include "Neighbouring_Nodes.h"

using namespace std;

namespace BCparticles
{
    enum Enum
    {
        EXIT            = 0,
        PERFECT_MIRROR  = 1,
        REEMISSION      = 2,
        OPENBCOut       = 3,
        OPENBCIn        = 4
    };
}

/**
 * 
 * class for particles of the same species with communications methods
 * @date Fri Jun 4 2007
 * @author Stefano Markidis, Giovanni Lapenta
 * @version 2.0
 *
 */
class Particles3Dcomm // :public Particles
{
public:
    //! Constructor !//
    Particles3Dcomm(int species, CollectiveIO * col, VirtualTopology3D * vct, Grid * grid);
  
    //! Destructor !//
    ~Particles3Dcomm();

  /** interpolation method GRID->PARTICLE order 1: CIC */
  // This does not belong in this class and is no longer in use.
//   void interpP2G(Field * EMf);

public: 
    //* Apply boundary conditions to all particles at the end of a list of particles starting with index start;
    //* These are virtual so user can override these to provide arbitrary custom boundary conditions
    virtual void apply_Xleft_BC(vector_SpeciesParticle& pcls, int start=0);
    virtual void apply_Yleft_BC(vector_SpeciesParticle& pcls, int start=0);
    virtual void apply_Zleft_BC(vector_SpeciesParticle& pcls, int start=0);
    virtual void apply_Xrght_BC(vector_SpeciesParticle& pcls, int start=0);
    virtual void apply_Yrght_BC(vector_SpeciesParticle& pcls, int start=0);
    virtual void apply_Zrght_BC(vector_SpeciesParticle& pcls, int start=0);

    // void computeMoments(Field * EMf);

private:
    void apply_periodic_BC_global(vector_SpeciesParticle& pcl_list, int pstart);
    bool test_pcls_are_in_nonperiodic_domain(const vector_SpeciesParticle& pcls)const;
    bool test_pcls_are_in_domain(const vector_SpeciesParticle& pcls)const;
    bool test_outside_domain(const SpeciesParticle& pcl)const;
    bool test_outside_nonperiodic_domain(const SpeciesParticle& pcl)const;
    bool test_Xleft_of_domain(const SpeciesParticle& pcl){ return pcl.get_x() < 0.; }
    bool test_Xrght_of_domain(const SpeciesParticle& pcl){ return pcl.get_x() > Lx; }
    bool test_Yleft_of_domain(const SpeciesParticle& pcl){ return pcl.get_y() < 0.; }
    bool test_Yrght_of_domain(const SpeciesParticle& pcl){ return pcl.get_y() > Ly; }
    bool test_Zleft_of_domain(const SpeciesParticle& pcl){ return pcl.get_z() < 0.; }
    bool test_Zrght_of_domain(const SpeciesParticle& pcl){ return pcl.get_z() > Lz; }
    void apply_nonperiodic_BCs_global(vector_SpeciesParticle&, int pstart);
    bool test_all_pcls_are_in_subdomain();
    void apply_BCs_globally(vector_SpeciesParticle& pcl_list);
    void apply_BCs_locally(vector_SpeciesParticle& pcl_list, int direction, bool apply_shift, bool do_apply_BCs);

private: // communicate particles between processes
    void flush_send();
    bool send_pcl_to_appropriate_buffer(SpeciesParticle& pcl, int count[6]);
    int handle_received_particles(int pclCommMode=0);

public:
    int separate_and_send_particles();
    void recommunicate_particles_until_done(int min_num_iterations=3);
    void communicate_particles();
    void pad_capacities();

private:
    void resize_AoS(int nop);
    void resize_SoA(int nop);
    void copyParticlesToAoS();
    void copyParticlesToSoA();

public:
    void convertParticlesToSynched();
    void convertParticlesToAoS();
    void convertParticlesToSoA();
    bool particlesAreSoA()const;

    //* sort particles for vectorized push (needs to be parallelized) *//
    void sort_particles_serial();
    void sort_particles_serial_AoS();

    //* Particle creation methods
    void reserve_remaining_particle_IDs()
    {
        // reserve remaining particle IDs starting from getNOP()
        pclIDgenerator.reserve_particles_in_range(getNOP());
    }

    //* Create new particle
    void create_new_particle(double u, double v, double w, double q,
                             double x, double y, double z)
    {
        const double t = pclIDgenerator.generateID();
        _pcls.push_back(SpeciesParticle(u,v,w,q,x,y,z,t));
    }

    //* Add particle to the list
    void add_new_particle(double u, double v, double w, double q,
                          double x, double y, double z, double t)
    {
        _pcls.push_back(SpeciesParticle(u,v,w,q,x,y,z,t));
    }

    void delete_particle(int pidx)
    {
        _pcls[pidx]=_pcls.back();
        _pcls.pop_back();
    }

    // inline get accessors
    double get_dx(){return dx;}
    double get_dy(){return dy;}
    double get_dz(){return dz;}
    double get_invdx(){return inv_dx;}
    double get_invdy(){return inv_dy;}
    double get_invdz(){return inv_dz;}
    double get_xstart(){return xstart;}
    double get_ystart(){return ystart;}
    double get_zstart(){return zstart;}

    ParticleType::Type get_particleType()const { return particleType; }
    const SpeciesParticle& get_pcl(int pidx)const{ return _pcls[pidx]; }
    const vector_SpeciesParticle& get_pcl_list()const{ return _pcls; }
    const SpeciesParticle* get_pclptr(int id)const{ return &(_pcls[id]); }
    const double *getUall()  const { assert(particlesAreSoA()); return &u[0]; }
    const double *getVall()  const { assert(particlesAreSoA()); return &v[0]; }
    const double *getWall()  const { assert(particlesAreSoA()); return &w[0]; }
    const double *getXall()  const { assert(particlesAreSoA()); return &x[0]; }
    const double *getYall()  const { assert(particlesAreSoA()); return &y[0]; }
    const double *getZall()  const { assert(particlesAreSoA()); return &z[0]; }
    const double *getQall()  const { assert(particlesAreSoA()); return &q[0]; }
    const double *getParticleIDall() const{assert(particlesAreSoA());return &t[0];  }


    //* Downsampled particles 
    double *getU_DS()
    {
        const long long required_size = (getNOP() + ParticlesDownsampleFactor - 1) / ParticlesDownsampleFactor;

        if (u_ds != nullptr)
            delete[] u_ds;

        u_ds = new double[required_size];

        long long counter = 0;
        for (long long ip = 0; ip < getNOP(); ip += ParticlesDownsampleFactor) 
            u_ds[counter++] = u[ip];

        return u_ds; 
    }
    
    double *getV_DS()
    { 
        const long long required_size = (getNOP() + ParticlesDownsampleFactor - 1) / ParticlesDownsampleFactor;

        if (v_ds != nullptr)
            delete[] v_ds;

        v_ds = new double[required_size];

        long long counter = 0;
        for (long long ip = 0; ip < getNOP(); ip += ParticlesDownsampleFactor) 
            v_ds[counter++] = v[ip];

        return v_ds;
    }

    double *getW_DS()
    {
        const long long required_size = (getNOP() + ParticlesDownsampleFactor - 1) / ParticlesDownsampleFactor;

        if (w_ds != nullptr)
            delete[] w_ds;

        w_ds = new double[required_size];

        long long counter = 0;
        for (long long ip = 0; ip < getNOP(); ip += ParticlesDownsampleFactor) 
            w_ds[counter++] = w[ip];

        return w_ds;
    }

    double *getQ_DS()
    {
        const long long required_size = (getNOP() + ParticlesDownsampleFactor - 1) / ParticlesDownsampleFactor;

        if (q_ds != nullptr)
            delete[] q_ds;

        q_ds = new double[required_size];

        long long counter = 0;
        for (long long ip = 0; ip < getNOP(); ip += ParticlesDownsampleFactor) 
        {
            q_ds[counter] = q[ip];
            counter++;
        }

        return q_ds; 
    }

    double *getX_DS()
    {
        const long long required_size = (getNOP() + ParticlesDownsampleFactor - 1) / ParticlesDownsampleFactor;

        if (x_ds != nullptr)
            delete[] x_ds;

        x_ds = new double[required_size];

        long long counter = 0;
        for (long long ip = 0; ip < getNOP(); ip += ParticlesDownsampleFactor) 
        {
            x_ds[counter] = x[ip];
            counter++;
        }

        return x_ds; 
    }
    
    double *getY_DS()
    {
        const long long required_size = (getNOP() + ParticlesDownsampleFactor - 1) / ParticlesDownsampleFactor;

        if (y_ds != nullptr)
            delete[] y_ds;

        y_ds = new double[required_size];

        long long counter = 0;
        for (long long ip = 0; ip < getNOP(); ip += ParticlesDownsampleFactor) 
        {
            y_ds[counter] = y[ip];
            counter++;
        }

        return y_ds; 
    }

    double *getZ_DS()
    {
        const long long required_size = (getNOP() + ParticlesDownsampleFactor - 1) / ParticlesDownsampleFactor;

        if (z_ds != nullptr)
            delete[] z_ds;

        z_ds = new double[required_size];

        long long counter = 0;
        for (long long ip = 0; ip < getNOP(); ip += ParticlesDownsampleFactor) 
        {
            z_ds[counter] = z[ip];
            counter++;
        }

        return z_ds; 
    }
  
    //* accessors for particle with index indexPart
    int getNOP() const { return _pcls.size(); }

    //* Get number of downsampled particles
    long long get_NOP_DS() const 
    {
        long long nop_ds = static_cast<long long>(ceil(static_cast<double>(getNOP())/ParticlesDownsampleFactor));
        return (nop_ds);
    }

    // set particle components
    void setU(int i, double in){_pcls[i].set_u(in);}
    void setV(int i, double in){_pcls[i].set_v(in);}
    void setW(int i, double in){_pcls[i].set_w(in);}
    void setQ(int i, double in){_pcls[i].set_q(in);}
    void setX(int i, double in){_pcls[i].set_x(in);}
    void setY(int i, double in){_pcls[i].set_y(in);}
    void setZ(int i, double in){_pcls[i].set_z(in);}
    void setT(int i, double in){_pcls[i].set_t(in);}
    // fetch particle components
    double& fetchU(int i){return _pcls[i].fetch_u();}
    double& fetchV(int i){return _pcls[i].fetch_v();}
    double& fetchW(int i){return _pcls[i].fetch_w();}
    double& fetchQ(int i){return _pcls[i].fetch_q();}
    double& fetchX(int i){return _pcls[i].fetch_x();}
    double& fetchY(int i){return _pcls[i].fetch_y();}
    double& fetchZ(int i){return _pcls[i].fetch_z();}
    double& fetchT(int i){return _pcls[i].fetch_t();}
    // get particle components
    double getU(int i)const{return _pcls[i].get_u();}
    double getV(int i)const{return _pcls[i].get_v();}
    double getW(int i)const{return _pcls[i].get_w();}
    double getQ(int i)const{return _pcls[i].get_q();}
    double getX(int i)const{return _pcls[i].get_x();}
    double getY(int i)const{return _pcls[i].get_y();}
    double getZ(int i)const{return _pcls[i].get_z();}
    double getT(int i)const{return _pcls[i].get_t();}

    double get_momentum();
    double get_kinetic_energy();
    
    int get_num_particles();
    double get_total_charge();

    double getMaxVelocity();
    double getMinVelocity();
    double *getVelocityDistribution(int nBins, double minVel, double maxVel);

    void Print() const;
    void PrintNp() const;
      
public:

    int get_species_num()const { return ns; }
    int get_numpcls_in_bucket(int cx, int cy, int cz)const { return (*numpcls_in_bucket)[cx][cy][cz]; }
    int get_bucket_offset(int cx, int cy, int cz)const { return (*bucket_offset)[cx][cy][cz]; }

protected:
  
    //* pointers to topology and grid information (should be const)
    const Collective * col;
    const VirtualTopology3D * vct;
    const Grid * grid;

    //* Number of species
    int ns;

    //* Number of particles per cell (total, X, Y, Z)
    int npcel, npcelx, npcely, npcelz;

    //* Charge to mass ratio
    double qom;
    
    //* Thermal velocity (X, Y, Z)
    double uth, vth, wth;

    //* Bulk/Drift velocity (X, Y, Z)
    double u0, v0, w0;

    //* Initial charge density
    double rhoINIT;

    // used to generate unique particle IDs
    doubleIDgenerator pclIDgenerator;

    bool SaveHeatFluxTensor;

    ParticleType::Type particleType;

    // AoS representation
    vector_SpeciesParticle _pcls;

    //? Particles' data (SoA representation)
    vector_double u;
    vector_double v;
    vector_double w;
    vector_double q;
    vector_double x;
    vector_double y;
    vector_double z;
    vector_double t;            // subcycle time

    //? Downsampled particles' data
    double* u_ds;
    double* v_ds;
    double* w_ds;
    double* q_ds;
    double* x_ds;
    double* y_ds;
    double* z_ds;

    // indicates whether this class is for tracking particles
    bool TrackParticleID;
    bool isTestParticle;
    double pitch_angle;
    double energy;

    //Larray<SpeciesParticle> _pclstmp;
    vector_SpeciesParticle _pclstmp;

    // references for buckets for serial sort.
    array3_int* numpcls_in_bucket;
    array3_int* numpcls_in_bucket_now; // accumulator used during sorting
    //array3_int* bucket_size; // maximum number of particles in bucket
    array3_int* bucket_offset;

    /** rank of processor in which particle is created (for ID) */
    int BirthRank[2];
    /** number of variables to be stored in buffer for communication for each particle  */
    int nVar;
    /** time step */
    double dt;

    // Copies of grid data (should just put pointer to Grid in this class)
    //* Simulation domain length
    double xstart, xend, ystart, yend, zstart, zend, invVOL;
    
    //* Simulation box length
    double Lx, Ly, Lz, dx, dy, dz;

    //* Number of grid nodes and cells
    int nxn, nyn, nzn, nxc, nyc, nzc;

    // convenience values from grid
    double inv_dx, inv_dy, inv_dz;

    // Communication variables
    /** buffers for communication */
    MPI_Comm mpi_comm;

    // send buffers
    BlockCommunicator<SpeciesParticle> sendXleft;
    BlockCommunicator<SpeciesParticle> sendXrght;
    BlockCommunicator<SpeciesParticle> sendYleft;
    BlockCommunicator<SpeciesParticle> sendYrght;
    BlockCommunicator<SpeciesParticle> sendZleft;
    BlockCommunicator<SpeciesParticle> sendZrght;

    // recv buffers
    BlockCommunicator<SpeciesParticle> recvXleft;
    BlockCommunicator<SpeciesParticle> recvXrght;
    BlockCommunicator<SpeciesParticle> recvYleft;
    BlockCommunicator<SpeciesParticle> recvYrght;
    BlockCommunicator<SpeciesParticle> recvZleft;
    BlockCommunicator<SpeciesParticle> recvZrght;

    /** bool for communication verbose */
    bool cVERBOSE;
    /** Boundary condition on particles:
             <ul>
            <li>0 = exit</li>
            <li>1 = perfect mirror</li>
            <li>2 = riemission</li>
            <li>3 = periodic condition </li>
            </ul>
    */
    //* Boundary Condition for Particles
    int bcPfaceXright;
    int bcPfaceXleft;
    int bcPfaceYright;
    int bcPfaceYleft;
    int bcPfaceZright;
    int bcPfaceZleft;

    //* Speed of light in vacuum
    double c;

    //* restart variable for loading particles from restart file
    int restart;

    /** Number of iteration of the mover*/
    int NiterMover;
    
    //* Velocity of injection of particles
    double Vinj;

    //* Removed charge from species
    double Q_removed;

    //* density of the injection of the particles
    double Ninj;

    //* Object of class to handle which nodes have to be computed when the mass matrix is calculated
    NeighbouringNodes NeNo;

    //* Limits to apply to particle velocity
    double umin, umax, vmin, vmax, wmin, wmax;

    //* RelSIM
    bool Relativistic;

    //* Relativistic particle pusher
    string Relativistic_pusher;

    //* Custom input parameters
    double *input_param; int nparam;

    //* Downsampling factor
    int ParticlesDownsampleFactor;

};

// find the particles with particular IDs and print them
void print_pcls(vector_SpeciesParticle& pcls, int ns, longid* id_list, int num_ids);

typedef Particles3Dcomm Particles;

#endif
