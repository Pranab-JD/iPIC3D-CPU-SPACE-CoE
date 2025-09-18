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
  Particles3Dcomm.cpp  -  Class for particles of the same species, in a 2D space and 3component velocity
  -------------------
developers: Stefano Markidis, Giovanni Lapenta.
 ********************************************************************************************/

#include <mpi.h>
#include <iostream>
#include <math.h>
#include <limits.h>
#include "asserts.h"
#include <algorithm> // for swap, std::max
#include "VCtopology3D.h"
#include "Collective.h"
#include "Alloc.h"
#include "Grid3DCU.h"
#include "Field.h"
#include "MPIdata.h"
#include "ompdefs.h"
#include "ipicmath.h"
#include "ipicdefs.h"
#include "mic_basics.h"
#include "parallel.h"

#include "Particle.h"
#include "Particles3Dcomm.h"
#include "Parameters.h"

#include "ipichdf5.h"
#include <vector>
//#include <complex>
#include "debug.h"
#include "TimeTasks.h"
#include "../LeXInt_Timer.hpp"

using std::cout;
using std::endl;

/**
 * 
 * Class for particles of the same species, in a 2D space and 3component velocity
 * @date Fri Jun 4 2007
 * @author Stefano Markidis, Giovanni Lapenta
 * @version 2.0
 *
 */

static bool print_pcl_comm_counts = false;

static void print_pcl(SpeciesParticle& pcl, int ns)
{
    dprintf("--- pcl spec %d ---", ns);
    dprintf("u = %+6.4f", pcl.get_u());
    dprintf("v = %+6.4f", pcl.get_v());
    dprintf("w = %+6.4f", pcl.get_w());
    dprintf("q = %+6.4f", pcl.get_q());
    dprintf("x = %+6.4f", pcl.get_x());
    dprintf("y = %+6.4f", pcl.get_y());
    dprintf("z = %+6.4f", pcl.get_z());
    dprintf("t = %5.0f", pcl.get_t());
}

static void print_pcls(vector_SpeciesParticle& pcls, int start, int ns)
{
    for(int pidx = start; pidx < pcls.size(); pidx++)
    {
        dprintf("--- particle %d.%d ---", ns,pidx);
        dprintf("u[%d] = %+6.4f", pidx, pcls[pidx].get_u());
        dprintf("v[%d] = %+6.4f", pidx, pcls[pidx].get_v());
        dprintf("w[%d] = %+6.4f", pidx, pcls[pidx].get_w());
        dprintf("q[%d] = %+6.4f", pidx, pcls[pidx].get_q());
        dprintf("x[%d] = %+6.4f", pidx, pcls[pidx].get_x());
        dprintf("y[%d] = %+6.4f", pidx, pcls[pidx].get_y());
        dprintf("z[%d] = %+6.4f", pidx, pcls[pidx].get_z());
        dprintf("t[%d] = %5.0f", pidx, pcls[pidx].get_t());
    }
}

void print_pcls(vector_SpeciesParticle& pcls, int ns, longid* id_list, int num_ids)
{
    dprintf("=== species %d, with %d pcls ===", ns, pcls.size());

        for(int pidx=0; pidx<pcls.size();pidx++)
            for(int i=0;i<num_ids;i++)
                if(pcls[pidx].get_ID()==id_list[i])
                {
                    dprintf("--- particle %d.%d ---", ns,pidx);
                    dprintf("u[%d] = %+6.4f", pidx, pcls[pidx].get_u());
                    dprintf("v[%d] = %+6.4f", pidx, pcls[pidx].get_v());
                    dprintf("w[%d] = %+6.4f", pidx, pcls[pidx].get_w());
                    dprintf("q[%d] = %+6.4f", pidx, pcls[pidx].get_q());
                    dprintf("x[%d] = %+6.4f", pidx, pcls[pidx].get_x());
                    dprintf("y[%d] = %+6.4f", pidx, pcls[pidx].get_y());
                    dprintf("z[%d] = %+6.4f", pidx, pcls[pidx].get_z());
                    dprintf("t[%d] = %5.0f", pidx, pcls[pidx].get_t());
                }
}

//! Destructor: deallocate particles !//
Particles3Dcomm::~Particles3Dcomm() 
{
    MPI_Comm_free(&mpi_comm);
    delete numpcls_in_bucket;
    delete numpcls_in_bucket_now;
    delete bucket_offset;

    //* Downsampled particles (not used for computations; only written to files)
    // if (u_ds) delete[] u_ds;
    // if (v_ds) delete[] v_ds;
    // if (w_ds) delete[] w_ds;
    // if (x_ds) delete[] q_ds;
    // if (y_ds) delete[] x_ds;
    // if (z_ds) delete[] y_ds;
    // if (q_ds) delete[] z_ds; 
}

//! Constructor for a single species !//
//* This was formerly Particles3Dcomm::allocate()
Particles3Dcomm::Particles3Dcomm(int species_number, CollectiveIO * col_, VirtualTopology3D * vct_, Grid * grid_):
                                ns(species_number),
                                col(col_),
                                vct(vct_),
                                grid(grid_),
                                pclIDgenerator(),
                                particleType(ParticleType::AoS)
{
    //* communicators for particles
    MPI_Comm_dup(vct->getParticleComm(), &mpi_comm);
    
    //* define connections
    using namespace Direction;

    sendXleft.init(Connection::null2self(vct->getXleft_neighbor_P(),XDN,XDN,mpi_comm));
    sendXrght.init(Connection::null2self(vct->getXright_neighbor_P(),XUP,XUP,mpi_comm));
    recvXleft.init(Connection::null2self(vct->getXleft_neighbor_P(),XUP,XDN,mpi_comm));
    recvXrght.init(Connection::null2self(vct->getXright_neighbor_P(),XDN,XUP,mpi_comm));

    sendYleft.init(Connection::null2self(vct->getYleft_neighbor_P(),YDN,YDN,mpi_comm));
    sendYrght.init(Connection::null2self(vct->getYright_neighbor_P(),YUP,YUP,mpi_comm));
    recvYleft.init(Connection::null2self(vct->getYleft_neighbor_P(),YUP,YDN,mpi_comm));
    recvYrght.init(Connection::null2self(vct->getYright_neighbor_P(),YDN,YUP,mpi_comm));

    sendZleft.init(Connection::null2self(vct->getZleft_neighbor_P(),ZDN,ZDN,mpi_comm));
    sendZrght.init(Connection::null2self(vct->getZright_neighbor_P(),ZUP,ZUP,mpi_comm));
    recvZleft.init(Connection::null2self(vct->getZleft_neighbor_P(),ZUP,ZDN,mpi_comm));
    recvZrght.init(Connection::null2self(vct->getZright_neighbor_P(),ZDN,ZUP,mpi_comm));

    recvXleft.post_recvs();
    recvXrght.post_recvs();
    recvYleft.post_recvs();
    recvYrght.post_recvs();
    recvZleft.post_recvs();
    recvZrght.post_recvs();

    // info from collectiveIO
    isTestParticle = (get_species_num()>=col->getNs());
    npcel  = col->getNpcel(get_species_num());
    npcelx = col->getNpcelx(get_species_num());
    npcely = col->getNpcely(get_species_num());
    npcelz = col->getNpcelz(get_species_num());
    qom    = col->getQOM(get_species_num());
    SaveHeatFluxTensor = col->getSaveHeatFluxTensor();
    ParticlesDownsampleFactor = col->getParticlesDownsampleFactor();

    if(!isTestParticle)
    {
        uth = col->getUth(get_species_num());
        vth = col->getVth(get_species_num());
        wth = col->getWth(get_species_num());
        u0 = col->getU0(get_species_num());
        v0 = col->getV0(get_species_num());
        w0 = col->getW0(get_species_num());
        Ninj = col->getRHOinject(get_species_num());
    }
    else
    {
        pitch_angle = col->getPitchAngle(get_species_num()-col->getNs());
        energy = col->getEnergy(get_species_num()-col->getNs());
    }

    dt = col->getDt();
    Lx = col->getLx();
    Ly = col->getLy();
    Lz = col->getLz();
    dx = grid->getDX();
    dy = grid->getDY();
    dz = grid->getDZ();

    c = col->getC();
    NiterMover = col->getNiterMover();            // info for mover
    Vinj = col->getVinj();                        // velocity of the injection from the wall

    // boundary condition for particles
    bcPfaceXright = col->getBcPfaceXright();
    bcPfaceXleft = col->getBcPfaceXleft();
    bcPfaceYright = col->getBcPfaceYright();
    bcPfaceYleft = col->getBcPfaceYleft();
    bcPfaceZright = col->getBcPfaceZright();
    bcPfaceZleft = col->getBcPfaceZleft();

    // info from Grid
    xstart = grid->getXstart();
    xend = grid->getXend();
    ystart = grid->getYstart();
    yend = grid->getYend();
    zstart = grid->getZstart();
    zend = grid->getZend();

    dx = grid->getDX();
    dy = grid->getDY();
    dz = grid->getDZ();
    inv_dx = 1.0/dx;
    inv_dy = 1.0/dy;
    inv_dz = 1.0/dz;

    nxn = grid->getNXN();
    nyn = grid->getNYN();
    nzn = grid->getNZN();
    nxc = grid->getNXC();
    nyc = grid->getNYC();
    nzc = grid->getNZC();
    assert_eq(nxc,nxn-1);
    assert_eq(nyc,nyn-1);
    assert_eq(nzc,nzn-1);
    invVOL = grid->getInvVOL();

    //* Custom input parameters
    nparam = col->getNparam();
    if (nparam > 0) 
    {
        input_param = new double[nparam];
        
        for (int ip=0; ip<nparam; ip++) 
            input_param[ip] = col->getInputParam(ip);
    }

    // info from VirtualTopology3D
    cVERBOSE = vct->getcVERBOSE();

    Relativistic = col->getRelativistic();
    Relativistic_pusher = col->getRelativisticPusher();

    //? Preallocate space in arrays ?//

    // determine number of particles to preallocate for this process.
    // determine number of cells in this process
    // we calculate in double precision to guard against overflow

    double dNp = double(grid->get_num_cells_rr())*col->getNpcel(species_number);
    double dNpmax = dNp * col->getNpMaxNpRatio();
    // ensure that particle index will not overflow 32-bit representation as long as dmaxnop is respected.
    assert_le(dNpmax,double(INT_MAX));
    const int nop = dNp;
    // initialize particle ID generator based on number of particles that will initially be produced.
    pclIDgenerator.reserve_num_particles(nop);

    // initialize each process with capacity for some extra particles
    const int initial_capacity = roundup_to_multiple(nop*1.2, DVECWIDTH);
    const int downsampled_capacity = roundup_to_multiple(nop/ParticlesDownsampleFactor*1.2, DVECWIDTH);

    //* SoA particle representation
    u.reserve(initial_capacity);
    v.reserve(initial_capacity);
    w.reserve(initial_capacity);
    q.reserve(initial_capacity);
    x.reserve(initial_capacity);
    y.reserve(initial_capacity);
    z.reserve(initial_capacity);
    t.reserve(initial_capacity);        // subcycle time

    if (ParticlesDownsampleFactor < 1) 
    {
        cout << "ERROR! ParticlesDownsampleFactor must be greater than or equal to 1!" << endl;
        cout << "===== SIMULATIONS ABORTED =====" << endl << endl;
        abort();
    }

    // AoS particle representation
    _pcls.reserve(initial_capacity);
    particleType = ParticleType::AoS; // canonical representation

    // allocate arrays for sorting particles
    numpcls_in_bucket = new array3_int(nxc, nyc, nzc);
    numpcls_in_bucket_now = new array3_int(nxc, nyc, nzc);
    bucket_offset = new array3_int(nxc, nyc, nzc);

    assert_eq(sizeof(SpeciesParticle), 64);

    //? if RESTART is true, initialize the particle in allocate method
    int restart_status = col->getRestart_status();
    if (restart_status == 1 || restart_status == 2)
    {
        #ifdef NO_HDF5
            eprintf("restart is supported only if compiling with HDF5");
        #else
            int species_number = get_species_num();
            // prepare arrays to receive particles
            particleType = ParticleType::SoA;
            col->read_particles_restart(vct, species_number,u, v, w, q, x, y, z, t);
            convertParticlesToAoS();
        #endif

        if (vct->getCartesian_rank() == 0)
            cout << "SUCCESSFULLY READ PARTICLE DATA FROM HDF5 FILES FOR RESTART" << endl;
    }

    //* set_velocity_caps()
    umax = 0.95*col->getLx()/col->getDt();
    vmax = 0.95*col->getLy()/col->getDt();
    wmax = 0.95*col->getLz()/col->getDt();
    umin = -umax;
    vmin = -vmax;
    wmin = -wmax;

    if(false && is_output_thread())
        printf("species %d velocity cap: umax=%g,vmax=%g,wmax=%g\n", ns, umax,vmax,wmax);
}

// pad capacities so that aligned vectorization does not result in an array overrun.
// This should usually be cheap (a no-op)
void Particles3Dcomm::pad_capacities()
{
    #pragma omp master
    {
        _pcls.reserve(roundup_to_multiple(_pcls.size(),DVECWIDTH));
        u.reserve(roundup_to_multiple(u.size(),DVECWIDTH));
        v.reserve(roundup_to_multiple(v.size(),DVECWIDTH));
        w.reserve(roundup_to_multiple(w.size(),DVECWIDTH));
        q.reserve(roundup_to_multiple(q.size(),DVECWIDTH));
        x.reserve(roundup_to_multiple(x.size(),DVECWIDTH));
        y.reserve(roundup_to_multiple(y.size(),DVECWIDTH));
        z.reserve(roundup_to_multiple(z.size(),DVECWIDTH));
        t.reserve(roundup_to_multiple(t.size(),DVECWIDTH));
    }
}

void Particles3Dcomm::resize_AoS(int nop)
{
    #pragma omp master
    {
        const int padded_nop = roundup_to_multiple(nop,DVECWIDTH);
        _pcls.reserve(padded_nop);
        _pcls.resize(nop);
    }
}

void Particles3Dcomm::resize_SoA(int nop)
{
    #pragma omp master
    {
        // allocate space for particles including padding
        const int padded_nop = roundup_to_multiple(nop,DVECWIDTH);

        u.reserve(padded_nop);
        v.reserve(padded_nop);
        w.reserve(padded_nop);
        q.reserve(padded_nop);
        x.reserve(padded_nop);
        y.reserve(padded_nop);
        z.reserve(padded_nop);
        t.reserve(padded_nop);

        // define size of particle data
        u.resize(nop);
        v.resize(nop);
        w.resize(nop);
        q.resize(nop);
        x.resize(nop);
        y.resize(nop);
        z.resize(nop);
        t.resize(nop);
        //if(is_output_thread()) dprintf("done resizing to hold %d", nop);
    }
}

//? returns true if particle was sent
inline bool Particles3Dcomm::send_pcl_to_appropriate_buffer(SpeciesParticle& pcl, int count[6])
{
    int was_sent = true;
    
    // put particle in appropriate communication buffer if exiting
    if(pcl.get_x() < xstart)
    {
        sendXleft.send(pcl);
        count[0]++;
    }
    else if(pcl.get_x() > xend)
    {
        sendXrght.send(pcl);
        count[1]++;
    }
    else if(pcl.get_y() < ystart)
    {
        sendYleft.send(pcl);
        count[2]++;
    }
    else if(pcl.get_y() > yend)
    {
        sendYrght.send(pcl);
        count[3]++;
    }
    else if(pcl.get_z() < zstart)
    {
        sendZleft.send(pcl);
        count[4]++;
    }
    else if(pcl.get_z() > zend)
    {
        sendZrght.send(pcl);
        count[5]++;
    }
    else was_sent = false;

    return was_sent;
}

// flush sending particles.
void Particles3Dcomm::flush_send()
{
    sendXleft.send_complete();
    sendXrght.send_complete();
    sendYleft.send_complete();
    sendYrght.send_complete();
    sendZleft.send_complete();
    sendZrght.send_complete();
}

void Particles3Dcomm::apply_periodic_BC_global(vector_SpeciesParticle& pcl_list, int pstart)
{
    const double Lxinv = 1/Lx;
    const double Lyinv = 1/Ly;
    const double Lzinv = 1/Lz;
    // apply shift to all periodic directions
    for(int pidx=pstart;pidx<pcl_list.size();pidx++)
    {
        SpeciesParticle& pcl = pcl_list[pidx];
        if(vct->getPERIODICX_P())
        {
            double& x = pcl.fetch_x();
            x = modulo(x, Lx, Lxinv);
        }
        if(vct->getPERIODICY_P())
        {
            double& y = pcl.fetch_y();
            y = modulo(y, Ly, Lyinv);
        }
        if(vct->getPERIODICZ_P())
        {
            double& z = pcl.fetch_z();
            z = modulo(z, Lz, Lzinv);
        }
    }
}

// routines for sorting list of particles
//
// sort_pcls: macro to put all particles that satisfy a condition
// at the end of an array of given size.  It has been written so
// that it could be replaced with a generic routine that takes
// the "condition" method as a callback function (and if the
// optimizer is good, the performance in this case would actually
// be just as good, so maybe we should just make this change).
//
// pcls: SpeciesParticle* or vector_SpeciesParticle
// size_in: number of elements in pcls list
// start_in: starting index of list to be sorted
// start_out: returns starting index of particles
//    for which the condition is true.
// condition: a (probably inline) function of SpeciesParticle
//   that returns true if the particle should go at the end of the list
//
#define sort_pcls(pcls, start_in, start_out, condition) \
{ \
  int start = (start_in); \
  assert(0<=start); \
  start_out = pcls.size(); \
  /* pidx traverses the array */ \
  for(int pidx=pcls.size()-1;pidx>=start;pidx--) \
  { \
    assert(pidx<start_out); \
    /* if condition is true, put the particle at the end of the list */ \
    if(condition(pcls[pidx])) \
    { \
      --start_out; \
      SpeciesParticle tmp_pcl = pcls[pidx]; \
      pcls[pidx] = pcls[start_out]; \
      pcls[start_out] = tmp_pcl; \
    } \
  } \
}

// condition methods to use in sorting particles
inline bool Particles3Dcomm::test_outside_domain(const SpeciesParticle& pcl)const
{
  // This could be vectorized
  bool is_outside_domain=(
       pcl.get_x() < 0. || pcl.get_y() < 0. || pcl.get_z() < 0.
    || pcl.get_x() > Lx || pcl.get_y() > Ly || pcl.get_z() > Lz );
  return is_outside_domain;
}
inline bool Particles3Dcomm::test_outside_nonperiodic_domain(const SpeciesParticle& pcl)const
{
  // This could be vectorized
  bool is_outside_nonperiodic_domain =
     (!vct->getPERIODICX_P() && (pcl.get_x() < 0. || pcl.get_x() > Lx)) ||
     (!vct->getPERIODICY_P() && (pcl.get_y() < 0. || pcl.get_y() > Ly)) ||
     (!vct->getPERIODICZ_P() && (pcl.get_z() < 0. || pcl.get_z() > Lz));
  return is_outside_nonperiodic_domain;
}

// apply user-supplied boundary conditions
//
void Particles3Dcomm::apply_nonperiodic_BCs_global(
  vector_SpeciesParticle& pcl_list, int pstart)
{
  int lstart;
  int lsize;
  if(!vct->getPERIODICX_P())
  {
    // separate out particles that need Xleft boundary conditions applied
    sort_pcls(pcl_list, pstart, lstart, test_Xleft_of_domain);
    // apply boundary conditions
    apply_Xleft_BC(pcl_list, lstart);
    // separate out particles that need Xrght boundary conditions applied
    sort_pcls(pcl_list, pstart, lstart, test_Xrght_of_domain);
    // apply boundary conditions
    apply_Xrght_BC(pcl_list, lstart);
  }
  if(!vct->getPERIODICY_P())
  {
    // separate out particles that need Yleft boundary conditions applied
    sort_pcls(pcl_list, pstart, lstart, test_Yleft_of_domain);
    // apply boundary conditions
    apply_Yleft_BC(pcl_list, lstart);
    // separate out particles that need Yrght boundary conditions applied
    sort_pcls(pcl_list, pstart, lstart, test_Yrght_of_domain);
    // apply boundary conditions
    apply_Yrght_BC(pcl_list, lstart);
  }
  if(!vct->getPERIODICZ_P())
  {
    // separate out particles that need Zleft boundary conditions applied
    sort_pcls(pcl_list, pstart, lstart, test_Zleft_of_domain);
    // apply boundary conditions
    apply_Zleft_BC(pcl_list, lstart);
    // separate out particles that need Zrght boundary conditions applied
    sort_pcls(pcl_list, pstart, lstart, test_Zrght_of_domain);
    // apply boundary conditions
    apply_Zrght_BC(pcl_list, lstart);
  }
}

bool Particles3Dcomm::test_pcls_are_in_nonperiodic_domain(const vector_SpeciesParticle& pcls)const
{
  const int size = pcls.size();
  for(int pidx=0;pidx<size;pidx++)
  {
    const SpeciesParticle& pcl = pcls[pidx];
    // should vectorize these comparisons
    bool not_in_domain = test_outside_nonperiodic_domain(pcl);
    if(__builtin_expect(not_in_domain, false)) return false;
  }
  return true; // all pcls are in domain
}
bool Particles3Dcomm::test_pcls_are_in_domain(const vector_SpeciesParticle& pcls)const
{
  const int size = pcls.size();
  for(int pidx=0;pidx<size;pidx++)
  {
    const SpeciesParticle& pcl = pcls[pidx];
    // should vectorize these comparisons
    bool not_in_domain = test_outside_domain(pcl);
    if(__builtin_expect(not_in_domain, false)) return false;
  }
  return true; // all pcls are in domain
}
bool Particles3Dcomm::test_all_pcls_are_in_subdomain()
{
  const int size = _pcls.size();
  for(int pidx=0;pidx<size;pidx++)
  {
    SpeciesParticle& pcl = _pcls[pidx];
    // should vectorize these comparisons
    bool not_in_subdomain = 
         pcl.get_x() < xstart || pcl.get_y() < ystart || pcl.get_z() < zstart
      || pcl.get_x() > xend   || pcl.get_y() > yend   || pcl.get_z() > yend;
    if(__builtin_expect(not_in_subdomain, false)) return false;
  }
  return true; // all pcls are in processor subdomain
}

// If do_apply_periodic_BC_global is false, then we may need to
// communicate particles as many as 2*(XLEN+YLEN+ZLEN) times
// after the call to handle_received_particles(true), because it
// is conceivable that a particle will be communicated the full
// length of a periodic domain, then have its position remapped
// to the proper position, and then traverse the full domain
// again.
//
// If do_apply_periodic_BC_global is true, then we can
// guarantee that all particles will be communicated within
// at most (XLEN+YLEN+ZLEN) communications, but on the other
// hand, a particle that would have been communicated in only
// two iterations by being wrapped around a periodic boundary
// will instead be communicated almost the full length of that
// dimension.
static bool do_apply_periodic_BC_global = false;
// apply boundary conditions to a list of particles globally
// (i.e. without regard to their current location in memory)
void Particles3Dcomm::apply_BCs_globally(vector_SpeciesParticle& pcl_list)
{
  // apply boundary conditions to every
  // particle until every particle lies in the domain.
  //
  //const int pend = pcl_list.size();
  const double Lxinv = 1/Lx;
  const double Lyinv = 1/Ly;
  const double Lzinv = 1/Lz;

  // put the particles outside the domain at the end of the list
  //
  // index of first particle that is unfinished
  int pstart = 0;
  // sort particles outside of the domain to the end of the list
  sort_pcls(pcl_list, 0, pstart, test_outside_domain);

  for(int i=0; pstart < pcl_list.size(); i++)
  {
    if(do_apply_periodic_BC_global)
    {
      apply_periodic_BC_global(pcl_list, pstart);
      // apply user-supplied boundary conditions
      apply_nonperiodic_BCs_global(pcl_list, pstart);
      // put particles outside of the domain at the end of the list
      sort_pcls(pcl_list, pstart, pstart, test_outside_domain);
    }
    else
    {
      apply_nonperiodic_BCs_global(pcl_list, pstart);
      sort_pcls(pcl_list, pstart, pstart, test_outside_nonperiodic_domain);
    }

    // if this fails, something has surely gone wrong
    // (e.g. we have a runaway particle).
    if(i>=100)
    {
      print_pcls(pcl_list,pstart, ns);
      dprint(pstart);
      dprint(pcl_list.size());
      eprintf("something went wrong.")
    }
  }
  if(do_apply_periodic_BC_global)
  {
    assert(test_pcls_are_in_domain(pcl_list));
  }
  else
  {
    assert(test_pcls_are_in_nonperiodic_domain(pcl_list));
  }
  // compute how many communications will be needed to communicate
  // the particles in the list to their appropriate locations
  //if(do_apply_periodic_BC_global)
  //{
  //  for(int pidx=0; pidx < pcl_list.size(); pidx++)
  //  {
  //    SpeciesParticle& pcl = pcl_list[pidx];
  //    // compute the processor subdomain coordinates of this particle
  //    ...
  //    // compute the distance from the current processor subdomain coordinates
  //    ...
  //  }
  //}
  //else
  //{
  //  for(int pidx=0; pidx < pcl_list.size(); pidx++)
  //  {
  //    SpeciesParticle& pcl = pcl_list[pidx];
  //    ...
  //  }
  //}
}

// direction: direction that list of particles is coming from
void Particles3Dcomm::apply_BCs_locally(vector_SpeciesParticle& pcl_list,
  int direction, bool apply_shift, bool do_apply_BCs)
{
  using namespace Direction;
  // if appropriate apply periodicity shift for this block
  //
  // could change from modulo to simple shift.
  //
  const double Lxinv = 1/Lx;
  const double Lyinv = 1/Ly;
  const double Lzinv = 1/Lz;
  if(apply_shift)
  {
    switch(direction)
    {
      default:
        invalid_value_error(direction);
      case XDN:
      case XUP:
        for(int pidx=0;pidx<pcl_list.size();pidx++)
        {
          SpeciesParticle& pcl = pcl_list[pidx];
          double& x = pcl.fetch_x();
          //const double x_old = x;
          x = modulo(x, Lx, Lxinv);
          //dprintf("recved pcl#%g: remapped x=%g outside [0,%g] to x=%g", pcl.get_t(), x_old, Lx, x);
          // if(direction==XDN) x -= Lx; else x += Lx;
        }
        break;
      case YDN:
      case YUP:
        for(int pidx=0;pidx<pcl_list.size();pidx++)
        {
          double& y = pcl_list[pidx].fetch_y();
          y = modulo(y, Ly, Lyinv);
          // if(direction==YDN) y -= Ly; else y += Ly;
        }
        break;
      case ZDN:
      case ZUP:
        for(int pidx=0;pidx<pcl_list.size();pidx++)
        {
          double& z = pcl_list[pidx].fetch_z();
          z = modulo(z, Lz, Lzinv);
          // if(direction==ZDN) z -= Lz; else z += Lz;
        }
        break;
    }
  }
  // if appropriate then apply boundary conditions to this block
  else if(do_apply_BCs)
  {
    int size = pcl_list.size();
    switch(direction)
    {
      default:
        invalid_value_error(direction);
      case XDN: assert(vct->noXleftNeighbor_P());
        apply_Xleft_BC(pcl_list);
        break;
      case XUP: assert(vct->noXrghtNeighbor_P());
        apply_Xrght_BC(pcl_list);
        break;
      case YDN: assert(vct->noYleftNeighbor_P());
        apply_Yleft_BC(pcl_list);
        break;
      case YUP: assert(vct->noYrghtNeighbor_P());
        apply_Yrght_BC(pcl_list);
        break;
      case ZDN: assert(vct->noZleftNeighbor_P());
        apply_Zleft_BC(pcl_list);
        break;
      case ZUP: assert(vct->noZrghtNeighbor_P());
        apply_Zrght_BC(pcl_list);
        break;
    }
    pcl_list.resize(size);
  }
}

namespace PclCommMode
{
  enum Enum
  {
    do_apply_BCs_globally=1,
    print_sent_pcls=2,
  };
}
// receive, sort, and, as appropriate, resend incoming particles
//
// assumes that flush_send() has been called
//
// returns number of particles that were resent
//
int Particles3Dcomm::handle_received_particles(int pclCommMode)
{
  using namespace PclCommMode;
  // we expect to receive at least one block from every
  // communicator, so make sure that all receive buffers are
  // clear and waiting
  //
  recvXleft.recv_start(); recvXrght.recv_start();
  recvYleft.recv_start(); recvYrght.recv_start();
  recvZleft.recv_start(); recvZrght.recv_start();

  // make sure that current block in each sender is ready for sending
  //
  sendXleft.send_start(); sendXrght.send_start();
  sendYleft.send_start(); sendYrght.send_start();
  sendZleft.send_start(); sendZrght.send_start();

  const int num_recv_buffers = 6;

  int recv_count[6]={0,0,0,0,0,0};
  int send_count[6]={0,0,0,0,0,0};
  int num_pcls_recved = 0;
  int num_pcls_resent = 0;
  //const int direction_map[6]={
  //  Connection::XDN,Connection::XUP,
  //  Connection::YDN,Connection::YUP,
  //  Connection::ZDN,Connection::ZUP};
  // receive incoming particles, 
  // immediately resending any exiting particles
  //
  MPI_Request recv_requests[num_recv_buffers] = 
  {
    recvXleft.get_curr_request(), recvXrght.get_curr_request(),
    recvYleft.get_curr_request(), recvYrght.get_curr_request(),
    recvZleft.get_curr_request(), recvZrght.get_curr_request()
  };
  BlockCommunicator<SpeciesParticle>* recvBuffArr[num_recv_buffers] =
  {
    &recvXleft, &recvXrght,
    &recvYleft, &recvYrght,
    &recvZleft, &recvZrght
  };

  assert(!recvXleft.comm_finished());
  assert(!recvXrght.comm_finished());
  assert(!recvYleft.comm_finished());
  assert(!recvYrght.comm_finished());
  assert(!recvZleft.comm_finished());
  assert(!recvZrght.comm_finished());

  // determine the periodicity shift for each incoming buffer
  const bool apply_shift[num_recv_buffers] =
  {
    vct->isPeriodicXlower_P(), vct->isPeriodicXupper_P(),
    vct->isPeriodicYlower_P(), vct->isPeriodicYupper_P(),
    vct->isPeriodicZlower_P(), vct->isPeriodicZupper_P()
  };
  const bool do_apply_BCs[num_recv_buffers] =
  {
    vct->noXleftNeighbor_P(), vct->noXrghtNeighbor_P(),
    vct->noYleftNeighbor_P(), vct->noYrghtNeighbor_P(),
    vct->noZleftNeighbor_P(), vct->noZrghtNeighbor_P()
  };
  const int direction[num_recv_buffers] =
  {
    Direction::XDN, Direction::XUP,
    Direction::YDN, Direction::YUP,
    Direction::ZDN, Direction::ZUP
  };
  // The documentation in the input file says that boundary conditions
  // are simply ignored in the periodic case, so I omit this check.
  //for(int i=0;i<6;i++)assert(!(apply_shift[i]&&do_apply_BCs[i]));
  // while there are still incoming particles
  // put them in the appropriate buffer
  //
  while(!(
    recvXleft.comm_finished() && recvXrght.comm_finished() &&
    recvYleft.comm_finished() && recvYrght.comm_finished() &&
    recvZleft.comm_finished() && recvZrght.comm_finished()))
  {
    int recv_index;
    MPI_Status recv_status;
    MPI_Waitany(num_recv_buffers,recv_requests,&recv_index,&recv_status);
    if(recv_index==MPI_UNDEFINED)
      eprintf("recv_requests contains no active handles");
    assert_ge(recv_index,0);
    assert_lt(recv_index,num_recv_buffers);
    //
    // grab the received block of particles and process it
    //
    BlockCommunicator<SpeciesParticle>* recvBuff = recvBuffArr[recv_index];
    Block<SpeciesParticle>& recv_block
      = recvBuff->fetch_received_block(recv_status);
    vector_SpeciesParticle& pcl_list = recv_block.fetch_block();

    if(pclCommMode&do_apply_BCs_globally)
    {
      apply_BCs_globally(pcl_list);
    }
    else
    {
      apply_BCs_locally(pcl_list, direction[recv_index],
        apply_shift[recv_index], do_apply_BCs[recv_index]);
    }

    recv_count[recv_index]+=recv_block.size();
    num_pcls_recved += recv_block.size();
    // process each particle in the received block.
    {
      for(int pidx=0;pidx<recv_block.size();pidx++)
      {
        SpeciesParticle& pcl = recv_block[pidx];
        bool was_sent = send_pcl_to_appropriate_buffer(pcl, send_count);

        if(__builtin_expect(was_sent,false))
        {
          num_pcls_resent++;
          if(pclCommMode&print_sent_pcls)
          {
            print_pcl(pcl,ns);
          }
        }
        else
        {
          // The particle belongs here, so put it in the
          // appropriate place. For now, all particles are in a
          // single list, so we append to the list.
          _pcls.push_back(pcl);
        }
      }
    }
    // release the block and update the receive request
    recvBuff->release_received_block();
    recv_requests[recv_index] = recvBuff->get_curr_request();
  }

  if(print_pcl_comm_counts)
  {
    dprintf("spec %d recved_count: %d+%d+%d+%d+%d+%d=%d", ns,
      recv_count[0], recv_count[1], recv_count[2],
      recv_count[3], recv_count[4], recv_count[5], num_pcls_recved);
    dprintf("spec %d resent_count: %d+%d+%d+%d+%d+%d=%d", ns,
      send_count[0], send_count[1], send_count[2],
      send_count[3], send_count[4], send_count[5], num_pcls_resent);
  }
  // return the number of particles that were resent

  return num_pcls_resent;
}

// these methods should be made virtual
// so that the user can override boundary conditions.
//
void Particles3Dcomm::apply_Xleft_BC(vector_SpeciesParticle& pcls, int start)
{
  int size = pcls.size();
  assert_le(0,start);
  switch(bcPfaceXleft)
  {
    default:
      unsupported_value_error(bcPfaceXleft);
    case BCparticles::PERFECT_MIRROR:
      for(int p=start;p<size;p++)
      {
        pcls[p].fetch_x() *= -1;
        pcls[p].fetch_u() *= -1;
      }
      break;
    case BCparticles::REEMISSION:
      // in this case it might be faster to convert to and
      // from SoA format, if calls to rand() can vectorize.
      for(int p=start;p<size;p++)
      {
        SpeciesParticle& pcl = pcls[p];
        pcl.fetch_x() *= -1;
        double u[3];
        sample_maxwellian(u[0],u[1],u[2], uth,vth,wth);
        u[0] = fabs(u[0]);
        pcl.set_u(u);
      }
      break;
    case BCparticles::EXIT:
      // clear the remainder of the list
      pcls.resize(start);
      break;
    case BCparticles::OPENBCIn:
    	break;
    case BCparticles::OPENBCOut:
    	break;
  }
}
void Particles3Dcomm::apply_Yleft_BC(vector_SpeciesParticle& pcls, int start)
{
  int size = pcls.size();
  assert_le(0,start);
  switch(bcPfaceYleft)
  {
    default:
      unsupported_value_error(bcPfaceYleft);
    case BCparticles::PERFECT_MIRROR:
      for(int p=start;p<size;p++)
      {
        SpeciesParticle& pcl = pcls[p];
        const double y_old = pcl.fetch_y();
        const double v_old = pcl.fetch_v();
        pcl.fetch_y() *= -1;
        pcl.fetch_v() *= -1;
        const double y_new = pcl.fetch_y();
        const double v_new = pcl.fetch_v();
        //dprintf("mirrored pcl#%g: y: %g->%g, v: %g->%g",
        //  pcl.get_t(), y_old, y_new, v_old, v_new);
      }
      break;
    case BCparticles::REEMISSION:
      // in this case it might be faster to convert to and
      // from SoA format, if calls to rand() can vectorize.
      for(int p=start;p<size;p++)
      {
        SpeciesParticle& pcl = pcls[p];
        pcl.fetch_y() *= -1;
        double u[3];
        sample_maxwellian(u[0],u[1],u[2], uth,vth,wth);
        u[1] = fabs(u[1]);
        pcl.set_u(u);
      }
      break;
    case BCparticles::EXIT:
      pcls.resize(start);
      break;
    case BCparticles::OPENBCIn:
    	break;
    case BCparticles::OPENBCOut:
    	break;
  }
}
void Particles3Dcomm::apply_Zleft_BC(vector_SpeciesParticle& pcls, int start)
{
  int size = pcls.size();
  assert_le(0,start);
  switch(bcPfaceZleft)
  {
    default:
      unsupported_value_error(bcPfaceZleft);
    case BCparticles::PERFECT_MIRROR:
      for(int p=start;p<size;p++)
      {
        pcls[p].fetch_z() *= -1;
        pcls[p].fetch_w() *= -1;
      }
      break;
    case BCparticles::REEMISSION:
      // in this case it might be faster to convert to and
      // from SoA format, if calls to rand() can vectorize.
      for(int p=start;p<size;p++)
      {
        SpeciesParticle& pcl = pcls[p];
        pcl.fetch_z() *= -1;
        double u[3];
        sample_maxwellian(u[0],u[1],u[2], uth,vth,wth);
        u[2] = fabs(u[2]);
        pcl.set_u(u);
      }
      break;
    case BCparticles::EXIT:
      pcls.resize(start);
      break;
    case BCparticles::OPENBCIn:
    	break;
    case BCparticles::OPENBCOut:
    	break;
  }
}
void Particles3Dcomm::apply_Xrght_BC(vector_SpeciesParticle& pcls, int start)
{
  int size = pcls.size();
  assert_le(0,start);
  switch(bcPfaceXright)
  {
    default:
      unsupported_value_error(bcPfaceXright);
    case BCparticles::PERFECT_MIRROR:
      for(int p=start;p<size;p++)
      {
        double& x = pcls[p].fetch_x();
        x = 2*Lx - x;
        pcls[p].fetch_u() *= -1;
      }
      break;
    case BCparticles::REEMISSION:
      // in this case it might be faster to convert to and
      // from SoA format, if calls to rand() can vectorize.
      for(int p=start;p<size;p++)
      {
        SpeciesParticle& pcl = pcls[p];
        double& x = pcl.fetch_x();
        x = 2*Lx - x;
        double u[3];
        sample_maxwellian(u[0],u[1],u[2], uth,vth,wth);
        u[0] = -fabs(u[0]);
        pcl.set_u(u);
      }
      break;
    case BCparticles::EXIT:
      pcls.resize(start);
      break;
    case BCparticles::OPENBCIn:
    	break;
    case BCparticles::OPENBCOut:
    	break;
  }
}
void Particles3Dcomm::apply_Yrght_BC(vector_SpeciesParticle& pcls, int start)
{
  int size = pcls.size();
  assert_le(0,start);
  switch(bcPfaceYright)
  {
    default:
      unsupported_value_error(bcPfaceYright);
    case BCparticles::PERFECT_MIRROR:
      for(int p=start;p<size;p++)
      {
        SpeciesParticle& pcl = pcls[p];
        double& y = pcl.fetch_y();
        const double y_old = y;
        const double v_old = pcl.fetch_v();
        y = 2*Ly - y;
        pcl.fetch_v() *= -1;
        const double y_new = pcl.fetch_y();
        const double v_new = pcl.fetch_v();
        //dprintf("mirrored pcl#%g: y: %g->%g, v: %g->%g",
        //  pcl.get_t(), y_old, y_new, v_old, v_new);
      }
      break;
    case BCparticles::REEMISSION:
      for(int p=start;p<size;p++)
      {
        SpeciesParticle& pcl = pcls[p];
        double& y = pcl.fetch_y();
        y = 2*Ly - y;
        double u[3];
        sample_maxwellian(u[0],u[1],u[2], uth,vth,wth);
        v[0] = -fabs(v[0]);
        pcl.set_u(u);
      }
      break;
    case BCparticles::EXIT:
      pcls.resize(start);
      break;
    case BCparticles::OPENBCIn:
    	break;
    case BCparticles::OPENBCOut:
    	break;
  }
}
void Particles3Dcomm::apply_Zrght_BC(vector_SpeciesParticle& pcls, int start)
{
  int size = pcls.size();
  assert_le(0,start);
  switch(bcPfaceZright)
  {
    default:
      unsupported_value_error(bcPfaceZright);
    case BCparticles::PERFECT_MIRROR:
      for(int p=start;p<size;p++)
      {
        double& z = pcls[p].fetch_z();
        z = 2*Lz - z;
        pcls[p].fetch_w() *= -1;
      }
      break;
    case BCparticles::REEMISSION:
      for(int p=start;p<size;p++)
      {
        SpeciesParticle& pcl = pcls[p];
        double& z = pcl.fetch_z();
        z = 2*Lz - z;
        double u[3];
        sample_maxwellian(u[0],u[1],u[2], uth,vth,wth);
        w[0] = -fabs(w[0]);
        pcl.set_u(u);
      }
      break;
    case BCparticles::EXIT:
      pcls.resize(start);
      break;
    case BCparticles::OPENBCIn:
    	break;
    case BCparticles::OPENBCOut:
    	break;
  }
}

// return number of particles sent
int Particles3Dcomm::separate_and_send_particles()
{
  // why does it happen that multiple particles have an ID of 0?
  const int num_ids = 1;
  longid id_list[num_ids] = {0};

  //timeTasks_set_communicating();

  convertParticlesToAoS();

  // activate receiving
  //
  recvXleft.recv_start(); recvXrght.recv_start();
  recvYleft.recv_start(); recvYrght.recv_start();
  recvZleft.recv_start(); recvZrght.recv_start();

  // make sure that current block in each sender is ready for sending
  //
  sendXleft.send_start(); sendXrght.send_start();
  sendYleft.send_start(); sendYrght.send_start();
  sendZleft.send_start(); sendZrght.send_start();

  int send_count[6]={0,0,0,0,0,0};
  const int num_pcls_initially = _pcls.size();
  int np_current = 0;
  while(np_current < _pcls.size())
  {
    SpeciesParticle& pcl = _pcls[np_current];
    // if the particle is exiting, put it in the appropriate send bin;
    // this could be done at conclusion of push after particles are
    // converted to AoS format in order to overlap communication
    // with computation.
    bool was_sent = send_pcl_to_appropriate_buffer(pcl,send_count);

    // fill in hole; for the sake of data pipelining could change
    // this to make a list of holes and then go back and fill
    // them in, but will builtin_expect also allow efficient
    // pipelining?  Or does the compiler generate instructions
    // that automatically adjust pipelining based on
    // accumulated statistical branching behavior?
    //
    // optimizer should assume that most particles are not sent
    if(__builtin_expect(was_sent,false))
    {
      //dprintf("sent particle %d", np_current);
      delete_particle(np_current);
    }
    else
    {
      np_current++;
    }
  }
  assert_eq(_pcls.size(),np_current);
  const int num_pcls_unsent = getNOP();
  const int num_pcls_sent = num_pcls_initially - num_pcls_unsent;
  if(print_pcl_comm_counts)
  {
    dprintf("spec %d send_count: %d+%d+%d+%d+%d+%d=%d",ns,
      send_count[0], send_count[1], send_count[2],
      send_count[3], send_count[4], send_count[5],num_pcls_sent);
  }
  return num_pcls_sent;
}

// communicate particles and apply boundary conditions
// until every particle is in the process of its subdomain.
//
// At the end of this method, the position of every particle in
// this process must lie in this process's proper subdomain,
// for two reasons: (1) sumMoments assumes that all particles
// lie in the proper subdomain of the process, and if this
// assumption is violated memory corruption can result, and
// (2) the extrapolation algorithm used by the mover is unstable,
// which could cause the velocity of particles not in the proper
// subdomain to blow up.
//
// The algorithm proceeds in three steps:
// (1) for min_num_iterations, receive particles from neighbor
// processes, apply boundary conditions locally (i.e. only if
// this is a boundary process), and resend any particles that
// still do not belong in this subdomain,
// (2) apply boundary conditions globally (i.e. independent of
// whether this is a boundary process), and resend any particles
// not in this subdomain, and
// (3) communicate particles (as many as XLEN+YLEN+ZLEN times)
// until every particle is in its appropriate subdomain.
//
// This communication algorithm is perhaps more sophisticated
// than is justified by the mover, which does not properly
// resolve fast-moving particles.
//
// min_num_iterations: number of iterations that this process
//   applies boundary conditions locally.
//   This method then applies boundary conditions globally to each
//   incoming particle until the particle resides in the domain.
//   forces
//   
void Particles3Dcomm::recommunicate_particles_until_done(int min_num_iterations)
{
  //timeTasks_set_communicating(); // communicating until end of scope
  assert_gt(min_num_iterations,0);
  // most likely exactly three particle communications
  // will be needed, one for each dimension of space,
  // so we begin with three receives and thereafter
  // before each receive we do an all-reduce to check
  // if more particles actually need to be received.
  //
  // alternatively, we could monitor how many iterations
  // are needed and adjust the initial number of iterations
  // accordingly.  The goals to balance would be
  // * to minimize the number of all-reduce calls and
  // * to minimize unnecessary sends,
  // with the overall goal of minimizing time spent in communication
  //
  long long num_pcls_sent;
  for(int i=0;i<min_num_iterations;i++)
  {
    flush_send(); // flush sending of particles
    num_pcls_sent = handle_received_particles();
    //dprintf("spec %d #pcls sent = %d iterations %d", ns, num_pcls_sent, i);
  }

  // apply boundary conditions to incoming particles
  // until they are in the domain
  flush_send(); // flush sending of particles
  num_pcls_sent = handle_received_particles(PclCommMode::do_apply_BCs_globally);

  //if(do_apply_periodic_BC_global)
  //  assert(test_pcls_are_in_domain(_pcls));
  //else
  //  assert(test_pcls_are_in_nonperiodic_domain(_pcls));

  // compute how many times particles must be communicated
  // globally before every particle is in the correct subdomain
  // (would need to modify handle_received_particles so that it
  // returns num_comm_needed)
  //const int global_num_comm_needed = mpi_global_max(num_comm_needed);
  // once I see that this is anticipating the number of communications
  // correctly, I will eliminate the total_num_pcls_sent check below
  //dprint(global_num_comm_needed);

  // continue receiving and resending incoming particles until
  // global all-reduce of num_pcls_resent is zero, indicating
  // that there are no more particles to be received.
  //
  long long total_num_pcls_sent;
  MPI_Allreduce(&num_pcls_sent, &total_num_pcls_sent, 1, MPI_LONG_LONG, MPI_SUM, mpi_comm);

  //dprintf("spec %d pcls sent: %d, %d", ns, num_pcls_sent, total_num_pcls_sent);

  // the maximum number of neighbor communications that would
  // be needed to put a particle in the correct mesh cell
  int comm_max_times = vct->getXLEN()+vct->getYLEN()+vct->getZLEN();
  if(!do_apply_periodic_BC_global) comm_max_times*=2;
  int comm_count=0;
  while(total_num_pcls_sent)
  {
    if(comm_count>=(comm_max_times))
    {
      dprintf("spec %d particles still uncommunicated:",ns);
      flush_send();
      num_pcls_sent = handle_received_particles(PclCommMode::print_sent_pcls);
      eprintf("failed to finish up particle communication"
        " within %d communications", comm_max_times);
    }

    // flush sending of particles
    flush_send();
    num_pcls_sent = handle_received_particles();


    MPI_Allreduce(&num_pcls_sent, &total_num_pcls_sent, 1, MPI_LONG_LONG, MPI_SUM, mpi_comm);
    if(print_pcl_comm_counts){
      dprint(total_num_pcls_sent);
    }
    comm_count++;
  }
}

// exchange particles with neighboring processors
//
// sent particles are deleted from _pcls.
// holes are filled with particles from end.
// then received particles are appended to end.
//
void Particles3Dcomm::communicate_particles()
{
  timeTasks_set_communicating(); // communicating until end of scope

  separate_and_send_particles();

  recommunicate_particles_until_done(1);
}

double Particles3Dcomm::get_total_charge()
{
    double localQ = 0.0;
    double totalQ = 0.0;

    for (int i = 0; i < _pcls.size(); i++)
    {
        SpeciesParticle& pcl = _pcls[i];
        const double q = pcl.get_q();
        
        localQ += q;
    }
    
    MPI_Allreduce(&localQ, &totalQ, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return (totalQ);
}

int Particles3Dcomm::get_num_particles()
{
    long long localN = 0.0;
    long long totalN = 0.0;
    
    localN = _pcls.size();
    
    MPI_Allreduce(&localN, &totalN, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    return (totalN);
}

//? Kinetic energy of all particles
double Particles3Dcomm::get_kinetic_energy() 
{
    double localKe = 0.0;
    double totalKe = 0.0;
    double lorentz_factor = 0.0;
    
    for (int i = 0; i < _pcls.size(); i++)
    {
        SpeciesParticle& pcl = _pcls[i];
        const double u = pcl.get_u();
        const double v = pcl.get_v();
        const double w = pcl.get_w();
        const double q = pcl.get_q();

        if (Relativistic) 
		{
            lorentz_factor = sqrt(1.0 + (u*u + v*v + w*w)/(c*c));
            localKe += (q/qom) * (lorentz_factor - 1.0) * c*c;
        }
        else
        {
            localKe += 0.5 * (q/qom) * (u*u + v*v + w*w);
        }
    }
    
    MPI_Allreduce(&localKe, &totalKe, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return (totalKe);
}

//? Kinetic energy removed from the system (needed in presence of damping)
// double Particles3Dcomm::getKremoved() 
// {
//     double localK = K_removed;
//     double totalK = 0.0;
//     MPI_Allreduce(&localK, &totalK, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
//     return (totalK);
// }

/** return the total momentum */
//
// This is the sum over all particles of the magnitude of the
// momentum, which has no physical meaning that I can see.
// we should be summing each component of the momentum. -eaj
//
double Particles3Dcomm::get_momentum() 
{
    double localP = 0.0;
    double totalP = 0.0;
    for (int i = 0; i < _pcls.size(); i++)
    {
        SpeciesParticle& pcl = _pcls[i];
        const double u = pcl.get_u();
        const double v = pcl.get_v();
        const double w = pcl.get_w();
        const double q = pcl.get_q();
        localP += (q/qom)*sqrt(u*u + v*v + w*w);
    }
    MPI_Allreduce(&localP, &totalP, 1, MPI_DOUBLE, MPI_SUM, mpi_comm);
    return (totalP);
}

/** return the highest kinetic energy */
double Particles3Dcomm::getMaxVelocity() {
  double localVel = 0.0;
  double maxVel = 0.0;
  for (int i = 0; i < _pcls.size(); i++)
  {
    SpeciesParticle& pcl = _pcls[i];
    const double u = pcl.get_u();
    const double v = pcl.get_v();
    const double w = pcl.get_w();
    localVel = std::max(localVel, sqrt(u*u + v*v + w*w));
  }
  MPI_Allreduce(&localVel, &maxVel, 1, MPI_DOUBLE, MPI_MAX, mpi_comm);
  return (maxVel);
}

//* Get particle velocity (energy) spectrum
double *Particles3Dcomm::getVelocityDistribution(int nBins, double minVel, double maxVel) 
{
	double *f = new double[nBins];

	for (int i = 0; i < nBins; i++) 
		f[i] = 0.0;
	
	double Vel = 0.0;
	int bin = 0;
	
	if (!Relativistic)
	{ 
		//! Nonrelativistic simulations
		double dv = maxVel/nBins;

        for (int i = 0; i < _pcls.size(); i++)
        {
            SpeciesParticle& pcl = _pcls[i];
            const double u = pcl.get_u();
            const double v = pcl.get_v();
            const double w = pcl.get_w();

            Vel = sqrt(u*u + v*v + w*w);
            bin = int (floor(Vel / dv));
            
            if (bin >= nBins)
                f[nBins - 1] += 1;
            else
                f[bin] += 1;
        }
	}
	else 
	{ 
		//! Relativistic simulations: logspace u distribution
		double *uDist, *du;
		uDist = new double[nBins];
		du = new double[nBins-1];
		
		for (int iu = 0; iu < nBins; iu++)  
			uDist[iu] = pow(10.0,double(iu)*1.0/(double(nBins)-1.0)*(log10(maxVel)-log10(minVel))+log10(minVel));
		
		for (int iu = 0; iu < nBins-1; iu++)
			du[iu] = uDist[iu+1] - uDist[iu];

        for (int i = 0; i < _pcls.size(); i++)
		{
            SpeciesParticle& pcl = _pcls[i];
            const double u = pcl.get_u();
            const double v = pcl.get_v();
            const double w = pcl.get_w();
            const double q = pcl.get_q();

            Vel = sqrt(u*u + v*v + w*w);
			bin = floor((log10(Vel)-log10(minVel))/((log10(maxVel)-log10(minVel))/double(nBins-1)));
			
			if (bin >= 0 && bin < nBins-1)
				f[bin] += fabs(q)/du[bin];
		}
	}

	MPI_Allreduce(MPI_IN_PLACE, f, nBins, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

	return f;
}

/** print particles info */
void Particles3Dcomm::Print() const
{
    cout << endl;
    cout << "Number of Particles: " << _pcls.size() << endl;
    cout << "Subgrid (" << vct->getCoordinates(0) << "," << vct->getCoordinates(1) << "," << vct->getCoordinates(2) << ")" << endl;
    cout << "Xin = " << xstart << "; Xfin = " << xend << endl;
    cout << "Yin = " << ystart << "; Yfin = " << yend << endl;
    cout << "Zin = " << zstart << "; Zfin = " << zend << endl;
    cout << "Number of species = " << get_species_num() << endl;
    for (int i = 0; i < _pcls.size(); i++)
    {
        const SpeciesParticle& pcl = _pcls[i];
        cout << "Particle #" << i << ":"
        << " x=" << pcl.get_x()
        << " y=" << pcl.get_y()
        << " z=" << pcl.get_z()
        << " u=" << pcl.get_u()
        << " v=" << pcl.get_v()
        << " w=" << pcl.get_w()
        << endl;
    }
    cout << endl;
}
/** print just the number of particles */
void Particles3Dcomm::PrintNp()  const
{
    cout << endl;
    cout << "Number of Particles of species " << get_species_num() << ": " << getNOP() << endl;
    cout << "Subgrid (" << vct->getCoordinates(0) << "," << vct->getCoordinates(1) << "," << vct->getCoordinates(2) << ")" << endl;
    cout << endl;
}

/***** particle sorting routines *****/

void Particles3Dcomm::sort_particles_serial()
{
  switch(particleType)
  {
    case ParticleType::AoS:
      sort_particles_serial_AoS();
      break;
    case ParticleType::SoA:
      convertParticlesToAoS();
      sort_particles_serial_AoS();
      convertParticlesToSynched();
      break;
    default:
      unsupported_value_error(particleType);
  }
}

// need to sort and communicate particles after each iteration
void Particles3Dcomm::sort_particles_serial_AoS()
{
  convertParticlesToAoS();

  _pclstmp.reserve(_pcls.size());
  {
    numpcls_in_bucket->setall(0);
    // iterate through particles and count where they will go
    for (int pidx = 0; pidx < _pcls.size(); pidx++)
    {
      const SpeciesParticle& pcl = get_pcl(pidx);
      // get the cell indices of the particle
      int cx,cy,cz; // the cell index from their actual position
      grid->get_safe_cell_coordinates(cx,cy,cz,pcl.get_x(),pcl.get_y(),pcl.get_z());

      // increment the number of particles in bucket of this particle
      (*numpcls_in_bucket)[cx][cy][cz]++;
    }

    // compute prefix sum to determine initial position
    // of each bucket (could parallelize this)
    //
    int accpcls=0;
    for(int cx=0;cx<nxc;cx++)
    for(int cy=0;cy<nyc;cy++)
    for(int cz=0;cz<nzc;cz++)
    {
      (*bucket_offset)[cx][cy][cz] = accpcls;
      accpcls += (*numpcls_in_bucket)[cx][cy][cz];
    }
    assert_eq(accpcls,getNOP());

    numpcls_in_bucket_now->setall(0);
    // put the particles where they are supposed to go
    const int nop = getNOP();
    for (int pidx = 0; pidx < nop; pidx++)
    {
      const SpeciesParticle& pcl = get_pcl(pidx);
      // get the cell indices of the particle
      int cx,cy,cz; // the address of the new bucket of this particle
      grid->get_safe_cell_coordinates(cx,cy,cz,pcl.get_x(),pcl.get_y(),pcl.get_z());

      // compute where the data should go
      const int numpcls_now = (*numpcls_in_bucket_now)[cx][cy][cz]++;
      const int outpidx = (*bucket_offset)[cx][cy][cz] + numpcls_now;
      assert_lt(outpidx, nop);
      assert_ge(outpidx, 0);
      assert_lt(pidx, nop);
      assert_ge(pidx, 0);

      // copy particle data to new location
      //
      _pclstmp[outpidx] = pcl;
    }
    // swap the tmp particle memory with the official particle memory
    {
      // if using accessors rather than transposition,
      // here I would need not only to swap the pointers but also
      // to swap all the accessors.
      //
      _pcls.swap(_pclstmp);
    }

    // check if the particles were sorted incorrectly
    if(true)
    {
      for(int cx=0;cx<nxc;cx++)
      for(int cy=0;cy<nyc;cy++)
      for(int cz=0;cz<nzc;cz++)
      {
        // check the number of particles in every cell is reasonable
        assert_eq((*numpcls_in_bucket_now)[cx][cy][cz], (*numpcls_in_bucket)[cx][cy][cz]);
      }
    }
  }
  // SoA particle representation is no longer valid
  particleType = ParticleType::AoS;
}

// This can be called from within an omp parallel block
void Particles3Dcomm::copyParticlesToSoA()
{
  timeTasks_set_task(TimeTasks::TRANSPOSE_PCLS_TO_SOA);
  const int nop = _pcls.size();
  // create memory for SoA representation
  resize_SoA(nop);
 #ifndef __MIC__stub // replace with __MIC__ when this has been debugged
  #pragma omp for
  for(int pidx=0; pidx<nop; pidx++)
  {
    const SpeciesParticle& pcl = _pcls[pidx];
    u[pidx] = pcl.get_u();
    v[pidx] = pcl.get_v();
    w[pidx] = pcl.get_w();
    q[pidx] = pcl.get_q();
    x[pidx] = pcl.get_x();
    y[pidx] = pcl.get_y();
    z[pidx] = pcl.get_z();
    t[pidx] = pcl.get_t();
  }
 #else // __MIC__
  // rather than doing stride-8 scatter,
  // copy and transpose data 8 particles at a time
  assert_divides(8,u.capacity());
  #pragma omp for
  for(int pidx=0; pidx<nop; pidx+=8)
  {
    F64vec8* SoAdata[8] = {
      (F64vec8*) &u[pidx],
      (F64vec8*) &v[pidx],
      (F64vec8*) &w[pidx],
      (F64vec8*) &q[pidx],
      (F64vec8*) &x[pidx],
      (F64vec8*) &y[pidx],
      (F64vec8*) &z[pidx],
      (F64vec8*) &t[pidx]};
    F64vec8* AoSdata = reinterpret_cast<F64vec8*>(&_pcls[pidx]);
    transpose_8x8_double(AoSdata,SoAdata);
  }
 #endif // __MIC__
  particleType = ParticleType::synched;
}

// This can be called from within an omp parallel block
void Particles3Dcomm::copyParticlesToAoS()
{
  timeTasks_set_task(TimeTasks::TRANSPOSE_PCLS_TO_AOS);
  const int nop = u.size();
//   if(is_output_thread()) dprintf("copying to array of structs");
  resize_AoS(nop);
 #ifndef __MIC__
  // use a simple stride-8 gather
  #pragma omp for
  for(int pidx=0; pidx<nop; pidx++)
  {
    _pcls[pidx].set(
      u[pidx],v[pidx],w[pidx], q[pidx],
      x[pidx],y[pidx],z[pidx], t[pidx]);
  }
 #else // __MIC__
  // for efficiency, copy data 8 particles at a time,
  // transposing each block of particles
  assert_divides(8,_pcls.capacity());
  #pragma omp for
  for(int pidx=0; pidx<nop; pidx+=8)
  {
    F64vec8* AoSdata = reinterpret_cast<F64vec8*>(&_pcls[pidx]);
    F64vec8* SoAdata[8] ={
      (F64vec8*) &u[pidx],
      (F64vec8*) &v[pidx],
      (F64vec8*) &w[pidx],
      (F64vec8*) &q[pidx],
      (F64vec8*) &x[pidx],
      (F64vec8*) &y[pidx],
      (F64vec8*) &z[pidx],
      (F64vec8*) &t[pidx]};
    transpose_8x8_double(SoAdata, AoSdata);
  }
 #endif
  particleType = ParticleType::synched;
}

// synched AoS and SoA conceptually implies a write-lock
//
void Particles3Dcomm::convertParticlesToSynched()
{
  switch(particleType)
  {
    default:
      unsupported_value_error(particleType);
    case ParticleType::SoA:
      copyParticlesToAoS();
      break;
    case ParticleType::AoS:
      copyParticlesToSoA();
      break;
    case ParticleType::synched:
      break;
  }
  // this state conceptually implies a write-lock
  particleType = ParticleType::synched;
}


// defines AoS to be the authority
// (conceptually releasing any write-lock)
//
void Particles3Dcomm::convertParticlesToAoS()
{
  switch(particleType)
  {
    default:
      unsupported_value_error(particleType);
    case ParticleType::SoA:
      copyParticlesToAoS();
      break;
    case ParticleType::AoS:
    case ParticleType::synched:
      break;
  }
  particleType = ParticleType::AoS;
}

// check whether particles are SoA
bool Particles3Dcomm::particlesAreSoA()const
{
  switch(particleType)
  {
    default:
      unsupported_value_error(particleType);
    case ParticleType::AoS:
      return false;
      break;
    case ParticleType::SoA:
    case ParticleType::synched:
      break;
  }
  return true;
}

// defines SoA to be the authority
// (conceptually releasing any write-lock)
//
void Particles3Dcomm::convertParticlesToSoA()
{
  switch(particleType)
  {
    default:
      unsupported_value_error(particleType);
    case ParticleType::AoS:
      copyParticlesToSoA();
      break;
    case ParticleType::SoA:
    case ParticleType::synched:
      break;
  }
  particleType = ParticleType::SoA;
}
