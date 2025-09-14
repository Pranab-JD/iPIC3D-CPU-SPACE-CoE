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


#include "mpi.h"
#include "MPIdata.h"
#include "iPic3D.h"
#include "TimeTasks.h"
#include "ipicdefs.h"
#include "debug.h"
#include "Parameters.h"
#include "ompdefs.h"
#include "VCtopology3D.h"
#include "Collective.h"
#include "Grid3DCU.h"
#include "EMfields3D.h"
#include "Particles3D.h"
#include "Timing.h"
#include "ParallelIO.h"
#include "outputPrepare.h"

#ifndef NO_HDF5
    #include "WriteOutputParallel.h"
    #include "OutputWrapperFPP.h"
#endif

#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <sys/stat.h>
#include <filesystem>

#include "Moments.h" // for debugging
#include "../LeXInt_Timer.hpp"

#ifdef USE_CATALYST
    #include "Adaptor.h"
#endif

using namespace iPic3D;

c_Solver::~c_Solver()
{
    delete col; // configuration parameters ("collectiveIO")
    delete vct; // process topology
    delete grid; // grid
    delete EMf; // field
    #ifndef NO_HDF5
        delete outputWrapperFPP;
    #endif
  
    //? Delete particles
    if(particles)
    {
        for (int i = 0; i < ns; i++)
        {
            // placement delete
            particles[i].~Particles3D();
        }
        free(particles);
    }

    #ifdef USE_CATALYST
        Adaptor::Finalize();
    #endif
    delete [] kinetic_energy_species;
    delete [] bulk_energy_species;
    delete [] momentum_species;
    delete [] charge_species;
    delete [] num_particles_species;
    delete [] Qremoved;
    delete my_clock;
}

int c_Solver::Init(int argc, char **argv) 
{

    #if defined(__MIC__)
        assert_eq(DVECWIDTH, 8);
    #endif

    Parameters::init_parameters();
    nprocs = MPIdata::get_nprocs();
    myrank = MPIdata::get_rank();

    col = new Collective(argc, argv);               // Every proc loads the parameters of simulation from class Collective
    restart_cycle   = col->getRestartOutputCycle();
    SaveDirName     = col->getSaveDirName();
    RestartDirName  = col->getRestartDirName();
    restart_status  = col->getRestart_status();
    ns              = col->getNs();                 // get the number of species of particles involved in simulation

    if (restart_status == 0)
        first_cycle = col->getLast_cycle() + 1;     
    else if (restart_status == 1 || restart_status == 2)
        first_cycle = col->getLast_cycle();         // get the last cycle from the restart

    vct = new VCtopology3D(*col);

    //? Check if we can map the processes into a matrix ordering defined in Collective.cpp
    if (nprocs != vct->getNprocs()) 
        if (myrank == 0) 
        {
            cerr << "Error: " << nprocs << " processes cant be mapped as " << vct->getXLEN() << "x" << vct->getYLEN() << "x" << vct->getZLEN() << ". Change XLEN, YLEN, & ZLEN in input file. " << endl;
            MPIdata::instance().finalize_mpi();
            return (1);
        }

    // We create a new communicator with a 3D virtual Cartesian topology
    vct->setup_vctopology(MPIdata::get_PicGlobalComm());
    
    // initialize the central cell index
    #ifdef BATSRUS
        // set index offset for each processor
        col->setGlobalStartIndex(vct);
    #endif

    //* Print initial settings to stdout and file
    if (myrank == 0) 
    {
        if (restart_status == 0)
            checkOutputFolder(SaveDirName);     //* Create output directory

        MPIdata::instance().Print();
        vct->Print();
        col->Print();
        col->save();
    }

    //* Create local grid
    grid = new Grid3DCU(col, vct);          // Create the local grid
    EMf = new EMfields3D(col, grid, vct);   // Create Electromagnetic Fields Object

    //! ======================== Initial Field Distribution ======================== !//

    if (restart_status == 1 || restart_status == 2)
    {   
        //! RESTART
        EMf->init();
    }
    else if (restart_status == 0)
    {   
        //! NEW INITIAL CONDITION (FIELDS)
        if (col->getRelativistic())
        {
            //! Relativistic Cases
            if      (col->getCase()=="Relativistic_Double_Harris_pairs")            EMf->init_Relativistic_Double_Harris_pairs();
            else if (col->getCase()=="Relativistic_Double_Harris_ion_electron")     EMf->init_Relativistic_Double_Harris_ion_electron();
            else if (col->getCase()=="Shock1D")                                     EMf->initShock1D();
            else if (col->getCase()=="Double_Harris")                               EMf->init_double_Harris();              //* Works for small enough velocities
            else if (col->getCase()=="Maxwell_Juttner")                             EMf->init();
            else 
            {
                if (myrank==0)
                {
                    cout << " =================================================================== " << endl;
                    cout << " WARNING: The case '" << col->getCase() << "' was not recognized. " << endl;
                    cout << "     Runing relativistic simulation with the default initialisation. " << endl;
                    cout << " =================================================================== " << endl;
                }

                EMf->init();
            }
        }
        else
        {
            //! Non Relativistic Cases
            if      (col->getCase()=="GEMnoPert") 		        EMf->initGEMnoPert();
            else if (col->getCase()=="ForceFree") 		        EMf->initForceFree();
            else if (col->getCase()=="GEM")       		        EMf->initGEM();
            else if (col->getCase()=="Double_Harris")           EMf->init_double_Harris();
            else if (col->getCase()=="Double_Harris_Hump")      EMf->init_double_Harris_hump();
            else if (col->getCase()=="Dipole")    		        EMf->initDipole();
            else if (col->getCase()=="Dipole2D")  		        EMf->initDipole2D();
            else if (col->getCase()=="NullPoints")              EMf->initNullPoints();
            else if (col->getCase()=="TaylorGreen")             EMf->initTaylorGreen();
            else if (col->getCase()=="KHI_FLR")                 EMf->init_KHI_FLR();
            else if (col->getCase()=="Uniform")                 EMf->init();
            else if (col->getCase()=="Maxwellian")              EMf->init();
            #ifdef BATSRUS
                else if (col->getCase()=="BATSRUS")   	        EMf->initBATSRUS();
            #endif
            else if (col->getCase()=="RandomCase")              EMf->initRandomField();
            else 
            {
                if (myrank==0)
                {
                    cout << " ================================================================ " << endl;
                    cout << " WARNING: The case '" << col->getCase() << "' was not recognized. " << endl;
                    cout << "       Runing simulation with the default initialisation. " << endl;
                    cout << " ================================================================ " << endl;
                }

                EMf->init();
            }
        }
    }
    else
    {
        if (myrank==0)
        {
            cout << "Incorrect restart status!" << endl;
            cout << "restart_status = 0 ---> NO RESTART!" << endl;
            cout << "restart_status = 1 ---> RESTART! SaveDirName and RestartDirName are different" << endl;
            cout << "restart_status = 1 ---> RESTART! SaveDirName and RestartDirName are the same" << endl;
        }
        abort();
    }

    //! ======================= Initial Particle Distribution (if NOT starting from RESTART) ======================= !//

    //* Allocation of particles
    particles = (Particles3D*) malloc(sizeof(Particles3D)*ns);

    for (int is = 0; is < ns; is++)  
        new(&particles[is]) Particles3D(is, col, vct, grid);

    //! NEW INITIAL CONDITION (PARTICLES)
    if (restart_status == 0) 
    {
        for (int i = 0; i < ns; i++)
        {
            if (col->getRelativistic()) 
			{   
                //! Relativistic Cases
				if      (col->getCase()=="Relativistic_Double_Harris_pairs") 	        particles[i].Relativistic_Double_Harris_pairs(EMf);
				else if (col->getCase()=="Relativistic_Double_Harris_ion_electron") 	particles[i].Relativistic_Double_Harris_ion_electron(EMf);
				else if (col->getCase()=="Shock1D") 	                                particles[i].Shock1D(EMf);
				else if (col->getCase()=="Shock1D_DoublePiston") 	                    particles[i].Shock1D_DoublePiston(EMf);
                else if (col->getCase()=="Maxwell_Jutter") 	                            particles[i].Maxwell_Juttner(EMf);
                else if (col->getCase()=="Double_Harris")                               particles[i].maxwellian_Double_Harris(EMf);           //* Works for small enough velocities
				else                                                                    particles[i].Maxwell_Juttner(EMf);
			}
            else
            {
                //! Non Relativistic Cases
                if      (col->getCase()=="ForceFree") 		                            particles[i].force_free(EMf);
                #ifdef BATSRUS
                    else if (col->getCase()=="BATSRUS")   	                            particles[i].MaxwellianFromFluid(EMf, col, i);
                #endif
                else if (col->getCase()=="NullPoints")    	                            particles[i].maxwellianNullPoints(EMf);
                else if (col->getCase()=="Uniform")    	                                particles[i].uniform_background(EMf);
                else if (col->getCase()=="TaylorGreen")                                 particles[i].maxwellianNullPoints(EMf);     //* Flow is initiated from the current prescribed on the grid
                else if (col->getCase()=="Double_Harris")                               particles[i].maxwellian_Double_Harris(EMf);
                else if (col->getCase()=="Double_Harris_Hump")                          particles[i].maxwellian_Double_Harris(EMf);   // In the old code, particles are read from field files
                else if (col->getCase()=="Maxwellian") 		                            particles[i].maxwellian(EMf);
                else if (col->getCase()=="KHI_FLR")                                     particles[i].maxwellian_KHI_FLR(EMf);
                else                                  		                            particles[i].maxwellian(EMf);
            }
            
            particles[i].reserve_remaining_particle_IDs();
            particles[i].fixPosition();
        }
    }

    //* Allocate test particles (if any)
    nstestpart = col->getNsTestPart();
    if(nstestpart>0)
    {
        testpart = (Particles3D*) malloc(sizeof(Particles3D)*nstestpart);
        for (int i = 0; i < nstestpart; i++)
        {
            new(&testpart[i]) Particles3D(i+ns,col,vct,grid);//species id for test particles is increased by ns
            testpart[i].pitch_angle_energy(EMf);
        }
    }

    //? Write particle and field output data
    if (Parameters::get_doWriteOutput())
    {
        #ifndef NO_HDF5
        if(col->getWriteMethod() == "shdf5" || col->getCallFinalize() || restart_cycle>0 || (col->getWriteMethod()=="pvtk" && !col->particle_output_is_off()) )
        {
            outputWrapperFPP = new OutputWrapperFPP;
            fetch_outputWrapperFPP().init_output_files(col, vct, grid, EMf, particles, ns, testpart, nstestpart);
        }
        #endif

        if(!col->field_output_is_off())
        {
            if(col->getWriteMethod()=="pvtk")
            {
                if(!(col->getFieldOutputTag()).empty())
                    fieldwritebuffer = newArr4(float,(grid->getNZN()-3),grid->getNYN()-3,grid->getNXN()-3,3);
                if(!(col->getMomentsOutputTag()).empty())
                    momentwritebuffer=newArr3(float,(grid->getNZN()-3), grid->getNYN()-3, grid->getNXN()-3);
            }
            else if(col->getWriteMethod()=="nbcvtk")
            {
                momentreqcounter=0;
                fieldreqcounter = 0;
                
                if(!(col->getFieldOutputTag()).empty())
                    fieldwritebuffer = newArr4(float,(grid->getNZN()-3)*4,grid->getNYN()-3,grid->getNXN()-3,3);
                if(!(col->getMomentsOutputTag()).empty())
                    momentwritebuffer=newArr3(float,(grid->getNZN()-3)*14, grid->getNYN()-3, grid->getNXN()-3);
            }
        }
    }

    //? Write conserved parameters to files
    kinetic_energy_species = new double[ns];
    bulk_energy_species = new double[ns];
    momentum_species = new double[ns];
    charge_species = new double[ns];
    num_particles_species = new int[ns];

    cq = SaveDirName + "/ConservedQuantities.txt";
    cqs = SaveDirName + "/SpeciesQuantities.txt";
    if (myrank == 0 && restart_status == 0) 
    {
        ofstream my_file(cq.c_str());
        my_file.close();

        ofstream my_file_(cqs.c_str());
        my_file_.close();
    }

    Qremoved = new double[ns];

    #ifdef USE_CATALYST
    Adaptor::Initialize(col, \
                        (int)(grid->getXstart()/grid->getDX()), \
                        (int)(grid->getYstart()/grid->getDY()), \
                        (int)(grid->getZstart()/grid->getDZ()), \
                        grid->getNXN(),
                        grid->getNYN(),
                        grid->getNZN(),
                        grid->getDX(),
                        grid->getDY(),
                        grid->getDZ());
    #endif

    my_clock = new Timing(myrank);

    return 0;
}

//* =================================== iPIC3D main computation =================================== *//

//! Compute moments
void c_Solver::CalculateMoments() 
{
    timeTasks_set_main_task(TimeTasks::MOMENTS);

    EMf->set_fieldForPcls();

    //* Avoid SIMD array overrun
    pad_particle_capacities();

    //* Vectorised; assumes that particles are sorted by mesh cell
    // if(Parameters::get_VECTORIZE_MOMENTS())
    // {
    //     switch(Parameters::get_MOMENTS_TYPE())
    //     {
    //         case Parameters::SoA:
    //             // since particles are sorted,we can vectorize interpolation of particles to grid
    //             convertParticlesToSoA();
    //             sortParticles();
    //             EMf->sumMoments_vectorized(part);
    //         break;
            
    //         case Parameters::AoS:
    //             convertParticlesToAoS();
    //             sortParticles();
    //             EMf->sumMoments_vectorized_AoS(part);
    //         break;
            
    //         default:
    //         unsupported_value_error(Parameters::get_MOMENTS_TYPE());
    //     }
    // }
    // else
    // {
    //     if(Parameters::get_SORTING_PARTICLES())
    //         sortParticles();

    //     switch(Parameters::get_MOMENTS_TYPE())
    //     {
    //         case Parameters::SoA:
    //             EMf->setZeroPrimaryMoments();
    //             convertParticlesToSoA();
    //             EMf->sumMoments(part);
    //         break;
            
    //         case Parameters::AoS:
                
    //             cout << "Parameters::get_MOMENTS_TYPE --> Moments AoS" << endl;
 
    //             //* Set moments to 0
    //             EMf->setZeroDensities();
                
    //             convertParticlesToAoS();
                
    //             EMf->sumMoments_AoS(part);      // sum up the 10 densities of each particles of each species
    //             // then calculate the weight according to their position; map the 10 momentum to the grid(node) with the weight
    //             cout << "---------------------------------" <<  endl;

    //         break;
            
    //         case Parameters::AoSintr:
    //             EMf->setZeroPrimaryMoments();
    //             convertParticlesToAoS();
    //             EMf->sumMoments_AoS_intr(part);
    //         break;
            
    //         default:
    //         unsupported_value_error(Parameters::get_MOMENTS_TYPE());
    //     }
    // }

    #ifdef __PROFILING__
    LeXInt::timer time_cm, time_com, time_int, time_total;

    time_total.start();
    #endif

    //? Set all moments and densities to 0
    EMf->setZeroDensities();

    #ifdef __PROFILING__
    time_cm.start();
    #endif
    
    //? Interpolate Particles to grid (nodes)
    for (int is = 0; is < ns; is++)
        particles[is].computeMoments(EMf);
    
    #ifdef __PROFILING__
    time_cm.stop();
    #endif

    #ifdef __PROFILING__
    time_com.start();
    #endif

    //? Communicate moments
    for (int is = 0; is < ns; is++)
        EMf->communicateGhostP2G_ecsim(is);

    EMf->communicateGhostP2G_mass_matrix();

    #ifdef __PROFILING__
    time_com.stop();
    #endif

    #ifdef __PROFILING__
    time_int.start();
    #endif
    
    //? Sum over all the species (charge and current densities)
    EMf->sumOverSpecies();

    //?  Communicate average densities
    for (int is = 0; is < ns; is++)
        EMf->interpolateCenterSpecies(is);

    #ifdef __PROFILING__
    time_int.stop();
    #endif

    EMf->timeAveragedRho(col->getPoissonMArho());
    EMf->timeAveragedDivE(col->getPoissonMAdiv());

    #ifdef __PROFILING__
    time_total.stop();

    if(MPIdata::get_rank() == 0)
    {
        cout << endl << "Profiling of MOMENT GATHERER" << endl; 
        cout << "Compute moments             : " << time_cm.total()    << " s, fraction of time taken in CalculateMoments(): " << time_cm.total()/time_total.total() << endl;
        cout << "Communicate moments         : " << time_com.total()   << " s, fraction of time taken in CalculateMoments(): " << time_com.total()/time_total.total() << endl;
        cout << "Summation & interpolation   : " << time_int.total()   << " s, fraction of time taken in CalculateMoments(): " << time_int.total()/time_total.total() << endl;
        cout << "CalculateMoments()          : " << time_total.total() << " s" << endl << endl;
    }
    #endif
}

//! Compute electromagnetic field
void c_Solver::ComputeEMFields(int cycle)
{
    col->setCurrentCycle(cycle);

    #ifdef __PROFILING__
    LeXInt::timer time_e, time_b, time_div, time_total;
    
    time_total.start();
    #endif

    //TODO: Only needed for the cases "Shock1D_DoublePiston" and "LangevinAntenna"; TBD later
	//* Update external fields to n+1/2
	// EMf->updateExternalFields(vct, grid, col, cycle); 

    //TODO: Only needed for the cases "ForcedDynamo"; TBD later
	//* Update particle external forces to n+1/2
	// EMf->updateParticleExternalForces(vct, grid, col, cycle); 

    #ifdef __PROFILING__
    time_e.start();
    #endif
    
    //? Compute E
    EMf->calculateE();
    
    #ifdef __PROFILING__
    time_e.stop();
    #endif
    
    #ifdef __PROFILING__
    time_b.start();
    #endif
    
    //? Compute B
    EMf->calculateB();
	
    #ifdef __PROFILING__
    time_b.stop();
    #endif

    #ifdef __PROFILING__
    time_div.start();
    #endif

    //? Compute divergences of E and B
    EMf->timeAveragedDivE(col->getPoissonMAdiv());      // TODO: What do these functions do? Ask Fabio
    EMf->divergence_E(col->getPoissonMAres());          //* Used to compute residual divergence for charge conservation
    EMf->divergence_B();

    #ifdef __PROFILING__
    time_div.stop();
    #endif

    //TODO: TBD later
    // double dt = col->getDt();
	// if (col->getTimeFile() != "none")
	// 	EMf->updateReferenceStateFromFile(grid, col, vct, cycle * dt);

    #ifdef __PROFILING__
    time_total.stop();

    if(MPIdata::get_rank() == 0)
    {
        cout << endl << "Profiling of FIELD SOLVER" << endl; 
        cout << "Compute electric field     : " << time_e.total()     << " s, fraction of time taken in FieldSolver(): " << time_e.total()/time_total.total() << endl;
        cout << "Compute magnetic field     : " << time_b.total()     << " s, fraction of time taken in FieldSolver(): " << time_b.total()/time_total.total() << endl;
        cout << "Compute divergence of B    : " << time_div.total()   << " s, fraction of time taken in FieldSolver(): " << time_div.total()/time_total.total() << endl;
        cout << "FieldSolver()              : " << time_total.total() << " s" << endl << endl;
    }
    #endif
}

//! Compute positions and velocities of particles
bool c_Solver::ParticlesMover()
{
    #ifdef __PROFILING__
    LeXInt::timer time_vel, time_relvel, time_pos, time_com, time_mag, time_total;

    time_total.start();
    #endif

    timeTasks_set_main_task(TimeTasks::PARTICLES);

    // Should change this to add background field
    EMf->set_fieldForPcls();

    //* Avoid SIMD array overrun
    pad_particle_capacities();

    #ifdef __PROFILING__
    time_vel.start();
    #endif

    //* Iterate over each species to update velocities
    for (int i = 0; i < ns; i++)
    {
        switch(Parameters::get_MOVER_TYPE())
        {
            //? ECSIM & RelSIM
            case Parameters::SoA:
                if (col->getRelativistic())
                    particles[i].RelSIM_velocity(EMf);
                else
                    particles[i].ECSIM_velocity(EMf);
            break;
            case Parameters::AoS:
                if (col->getRelativistic())
                    particles[i].RelSIM_velocity(EMf);
                else
                    particles[i].ECSIM_velocity(EMf);
            break;
            default:
            unsupported_value_error(Parameters::get_MOVER_TYPE());
        }
    }

    #ifdef __PROFILING__
    time_vel.stop();
    #endif

    #ifdef __PROFILING__
    time_pos.start();
    #endif

    //* Iterate over each species to update positions
    for (int i = 0; i < ns; i++)
    {
        switch(Parameters::get_MOVER_TYPE())
        {
            case Parameters::SoA:
                particles[i].ECSIM_position(EMf);
            break;
            case Parameters::AoS:
                particles[i].ECSIM_position(EMf);
            break;
        }

        //* Should integrate BC into separate_and_send_particles
        //TODO: what does this do?
        particles[i].openbc_particles_outflow();
        particles[i].separate_and_send_particles();
    }

    #ifdef __PROFILING__
    time_pos.stop();
    #endif

    #ifdef __PROFILING__
    time_com.start();
    #endif

    //* Communicate each species
    for (int i = 0; i < ns; i++)  
        particles[i].recommunicate_particles_until_done(1);

    #ifdef __PROFILING__
    time_com.stop();
    #endif

    #ifdef __PROFILING__
    time_mag.start();
    #endif

    //? Update the values of magnetic field at the nodes at time n+1
    EMf->C2NB();

    #ifdef __PROFILING__
    time_mag.stop();
    #endif

    //? Repopulate the buffer zone at the edge
    for (int i=0; i < ns; i++) 
    {
        if (col->getRHOinject(i)>0.0)
            particles[i].repopulate_particles();
    }

    //? Remove particles from depopulation area
    if (col->getCase()=="Dipole") 
    {
        for (int i=0; i < ns; i++)
            Qremoved[i] = particles[i].deleteParticlesInsideSphere(col->getL_square(), col->getx_center(), col->gety_center(), col->getz_center());
    }
    else if (col->getCase()=="Dipole2D") 
    {
        for (int i=0; i < ns; i++)
            Qremoved[i] = particles[i].deleteParticlesInsideSphere2DPlaneXZ(col->getL_square(), col->getx_center(), col->getz_center());
    }

    //* =============== Test Particles =============== *//

    //TODO: uncomment and test

    // for (int i = 0; i < nstestpart; i++)
    // {
    //     switch(Parameters::get_MOVER_TYPE())
    //     {
    //         case Parameters::SoA:
    //             testpart[i].ECSIM_velocity(EMf);
    //         break;
            
    //         default:
    //             unsupported_value_error(Parameters::get_MOVER_TYPE());
    //     }

    //     testpart[i].openbc_delete_testparticles();
    //     testpart[i].separate_and_send_particles();
    // }

    // for (int i = 0; i < nstestpart; i++)
    // {
    //     switch(Parameters::get_MOVER_TYPE())
    //     {
    //         case Parameters::SoA:
    //             testpart[i].ECSIM_position(EMf);
    //         break;
            
    //         default:
    //             unsupported_value_error(Parameters::get_MOVER_TYPE());
    //     }

    //     testpart[i].openbc_delete_testparticles();
    //     testpart[i].separate_and_send_particles();
    // }

    // for (int i = 0; i < nstestpart; i++)
    //     testpart[i].recommunicate_particles_until_done(1);

    //* ============================================== *//

    #ifdef __PROFILING__
    time_total.stop();

    if(MPIdata::get_rank() == 0)
    {
        cout << endl << "Profiling of PARTICLE MOVER" << endl; 
        cout << "Compute velocities            : " << time_vel.total()   << " s, fraction of time taken in ParticlesMover(): " << time_vel.total()/time_total.total() << endl;
        cout << "Compute positions             : " << time_pos.total()   << " s, fraction of time taken in ParticlesMover(): " << time_pos.total()/time_total.total() << endl;
        cout << "Communicate particles         : " << time_com.total()   << " s, fraction of time taken in ParticlesMover(): " << time_com.total()/time_total.total() << endl;
        cout << "Interpolate B to cell centres : " << time_mag.total()   << " s, fraction of time taken in ParticlesMover(): " << time_mag.total()/time_total.total() << endl;
        cout << "ParticlesMover()              : " << time_total.total() << " s" << endl << endl;
    }
    #endif

    return (false);
}

//* ===================================== WRITE DATA TO FILES ===================================== *//

void c_Solver::SupplementaryMoments() 
{
    EMf->setZeroDensities();

    //? Compute charge, current, energy flux, heat flux, and pressure tensor
    for (int is = 0; is < ns; is++)
        particles[is].compute_supplementary_moments(EMf);

    //? Communicate density, current, energy flux, heat flux, and pressure tensor
    for (int is = 0; is < ns; is++)
        EMf->communicateGhostP2G_supplementary_moments(is);

    //? Sum over all the species (only charge and current densities)
    EMf->sumOverSpecies_supplementary();
}

void c_Solver::WriteOutput(int cycle) 
{
    #ifdef USE_CATALYST
        Adaptor::CoProcess(col->getDt()*cycle, cycle, EMf);
    #endif

    //* Compute additional moments (to be written to files)
    SupplementaryMoments();

    WriteConserved(cycle);
    WriteRestart(cycle);

    if(!Parameters::get_doWriteOutput())  return;

    if (col->getWriteMethod() == "nbcvtk")
    {
        //! Non-blocking collective MPI-IO
        if(!col->field_output_is_off() && (cycle%(col->getFieldOutputCycle()) == 0 || cycle == first_cycle) )
        {
            if(!(col->getFieldOutputTag()).empty())
            {
                if(fieldreqcounter>0)
                {
                    //MPI_Waitall(fieldreqcounter,&fieldreqArr[0],&fieldstsArr[0]);
                    for(int si=0;si< fieldreqcounter;si++)
                    {
                        int error_code = MPI_File_write_all_end(fieldfhArr[si],&fieldwritebuffer[si][0][0][0],&fieldstsArr[si]);//fieldstsArr[si].MPI_ERROR;
                        
                        if (error_code != MPI_SUCCESS)
                        {
                            char error_string[100];
                            int length_of_error_string, error_class;
                            MPI_Error_class(error_code, &error_class);
                            MPI_Error_string(error_class, error_string, &length_of_error_string);
                            dprintf("MPI_Waitall error at field output cycle %d  %d  %s\n",cycle, si, error_string);
                        }
                        else
                            MPI_File_close(&(fieldfhArr[si]));
                    }
                }

                fieldreqcounter = WriteFieldsVTKNonblk(grid, EMf, col, vct,cycle,fieldwritebuffer,fieldreqArr,fieldfhArr);
            }

            if(!(col->getMomentsOutputTag()).empty())
            {
                if(momentreqcounter>0)
                {
                    //MPI_Waitall(momentreqcounter,&momentreqArr[0],&momentstsArr[0]);
                    for(int si=0;si< momentreqcounter;si++)
                    {
                        int error_code = MPI_File_write_all_end(momentfhArr[si],&momentwritebuffer[si][0][0],&momentstsArr[si]);//momentstsArr[si].MPI_ERROR;
                        
                        if (error_code != MPI_SUCCESS) 
                        {
                            char error_string[100];
                            int length_of_error_string, error_class;
                            MPI_Error_class(error_code, &error_class);
                            MPI_Error_string(error_class, error_string, &length_of_error_string);
                            dprintf("MPI_Waitall error at moments output cycle %d  %d %s\n",cycle, si, error_string);
                        }
                        else
                            MPI_File_close(&(momentfhArr[si]));
                    }
                }

                momentreqcounter = WriteMomentsVTKNonblk(grid, EMf, col, vct,cycle,momentwritebuffer,momentreqArr,momentfhArr);
            }
        }

        //! Particle data is written to hdf5 files
        WriteParticles(cycle);
        WriteTestParticles(cycle);

    }
    else if (col->getWriteMethod() == "pvtk")
    {
        //! Blocking collective MPI-IO
        if(!col->field_output_is_off() && (cycle%(col->getFieldOutputCycle()) == 0 || cycle == first_cycle) )
        {
		    if(!(col->getFieldOutputTag()).empty())
			    WriteFieldsVTK(grid, EMf, col, vct, col->getFieldOutputTag() ,cycle, fieldwritebuffer);//B + E + Je + Ji + rho

		    if(!(col->getMomentsOutputTag()).empty())
			    WriteMomentsVTK(grid, EMf, col, vct, col->getMomentsOutputTag() ,cycle, momentwritebuffer);
	    }

        //! Particle data is written to hdf5 files
        WriteParticles(cycle);
        WriteTestParticles(cycle);
    }
    else
    {
        //! HDF-based
		#ifdef NO_HDF5
            cout << "ERROR: shdf5 and phdf5 requires iPIC3D to be compiled with HDF5" << endl;
		#else

            if (col->getWriteMethod() == "phdf5")
            {
                //! Parallel HDF5 (THIS DOES NOT WORK -- Maybe this could be debugged) - PJD
			    if (!col->field_output_is_off() && cycle%(col->getFieldOutputCycle())==0)
				    WriteOutputParallel(grid, EMf, particles, col, vct, cycle);

			    if (!col->particle_output_is_off() && cycle%(col->getParticlesOutputCycle())==0)
			    {
				    if(MPIdata::get_rank()==0)
				        warning_printf("WriteParticlesParallel() is not yet implemented.");
			    }
			}
            else if (col->getWriteMethod() == "shdf5")
            {
                //! Serial HDF5
                if (!col->field_output_is_off() && (cycle%(col->getFieldOutputCycle()) == 0  || cycle == col->getNcycles() + first_cycle))
                {
                    if (restart_status == 0)
                        WriteFields(cycle);
                    
                    if ((restart_status == 1 || restart_status == 2) && cycle != first_cycle)
                        WriteFields(cycle);
                }

                if (!col->particle_output_is_off() && (cycle%(col->getParticlesOutputCycle()) == 0 || cycle == col->getNcycles() + first_cycle))
                {
                    if (restart_status == 0)
                        WriteParticles(cycle);
                    
                    if ((restart_status == 1 || restart_status == 2) && cycle != first_cycle)
                        WriteParticles(cycle);
                }

                if (!col->DS_particle_output_is_off() && col->getParticlesDownsampleFactor() > 1 && (cycle%(col->getParticlesDownsampleOutputCycle()) == 0 || cycle == col->getNcycles() + first_cycle))
                {
                    if (restart_status == 0)
                        WriteDSParticles(cycle);

                    if ((restart_status == 1 || restart_status == 2) && cycle != first_cycle)
                        WriteDSParticles(cycle);
                }
			}
            else
            {
			    cout << "ERROR: Invalid WriteMethod in input file. Available options: shdf5, pvtk, and nbcvtk." << endl;
			    invalid_value_error(col->getWriteMethod().c_str());
                abort();
			}

		#endif
  	}

    //! Output particle distributions to .txt files
    if (col->getRelativistic()) 
    {
        if (col->getParticleDistOutputCycle() != 0) 
        {
            if (cycle == first_cycle || cycle%(col->getParticleDistOutputCycle()) == 0 || cycle == col->getNcycles() + first_cycle) 
            {
                double vmin, vmax;
                int N_bins = col->getParticleDistBins();

                double* uDist = new double[N_bins];
                double* du = new double[N_bins-1];
                double* fDist = new double[N_bins-1];

                char place[128];
                const char* dirtxt = SaveDirName.c_str();
                FILE *fd;
                for (int is=0; is<ns; is++) 
                {
                    vmin = col->getParticleDistMinVelocity()*fabs(col->getQOM(is));
                    vmax = col->getParticleDistMaxVelocity()*fabs(col->getQOM(is));
                    
                    //* Output of u (velocity ninning) at first cycle
                    if (cycle == first_cycle && myrank == 0) 
                    {
                        for (int iu=0; iu<N_bins; iu++)  
                            uDist[iu] = pow(10.0,double(iu)*1.0/(double(N_bins)-1.0)*(log10(vmax)-log10(vmin))+log10(vmin));
                        
                        for (int iu=0; iu<N_bins-1; iu++)
                            du[iu] = uDist[iu+1] - uDist[iu];
            
                        sprintf(place,"%s/uDist_%d.txt",dirtxt,is);
                        fd = fopen(place,"w");
                        
                        for (int iu=0; iu<N_bins; iu++)
                            fprintf(fd,"%13.6e \n", uDist[iu]);
                        
                        fclose(fd);
                    } 
                    
                    fDist = particles[is].getVelocityDistribution(N_bins, vmin, vmax);

                    // Output this species' distribution to txt
                    if (myrank == 0) 
                    {
                        sprintf(place, "%s/fDist_%d_%06d.txt", dirtxt, is, cycle);
                        fd = fopen(place, "w");
                        
                        for (int iu = 0; iu < N_bins-1; iu++)
                            fprintf(fd, "%13.6e \n", fDist[iu]);
                        
                        fclose(fd);
                    }
                }
            }
        }
    }
}

void c_Solver::WriteConserved(int cycle)
{
    if (cycle == first_cycle) 
    {
        //? Total energy = Electric field energy + Magnetic field energy
        initial_total_energy = EMf->get_E_field_energy() + EMf->get_B_field_energy();
        
        //? Total energy = Electric field energy + Magnetic field energy + Kinetic Energy
        for (int is = 0; is < ns; is++) 
            initial_total_energy = initial_total_energy + particles[is].get_kinetic_energy();
    }

    if(col->getDiagnosticsOutputCycle() > 0 && cycle % col->getDiagnosticsOutputCycle() == 0)
    {
        double E_field_energy  = EMf->get_E_field_energy();
        double Ex_field_energy = EMf->get_Ex_field_energy();
        double Ey_field_energy = EMf->get_Ey_field_energy();
        double Ez_field_energy = EMf->get_Ez_field_energy();

        double B_field_energy  = EMf->get_B_field_energy();
        double Bx_field_energy = EMf->get_Bx_field_energy();
        double By_field_energy = EMf->get_By_field_energy();
        double Bz_field_energy = EMf->get_Bz_field_energy();

        double total_momentum = 0.0;
        double kinetic_energy = 0.0;

        for (int is = 0; is < ns; is++) 
        {
            momentum_species[is] += particles[is].get_momentum();
            kinetic_energy_species[is] = particles[is].get_kinetic_energy();
            bulk_energy_species[is] = EMf->get_bulk_energy(is);
            
            kinetic_energy += kinetic_energy_species[is];
            total_momentum += momentum_species[is];
        }
        
        if (myrank == (nprocs-1)) 
        {
            //? Conserved Quantities
            ofstream my_file(cq.c_str(), std::ios::app);

            if(cycle == 0 && restart_status == 0)
            {
                my_file << endl << "I.    Cycle" 
                        << endl << "II.   Electric field energy (total)"
                        << endl << "IIa.  Electric field energy along X"
                        << endl << "IIb.  Electric field energy along Y"
                        << endl << "IIc.  Electric field energy along Z" 
                        << endl << "III.  Magnetic field energy (total)"
                        << endl << "IIIa. Magnetic field energy along X"
                        << endl << "IIIb. Magnetic field energy along Y"
                        << endl << "IIIc. Magnetic field energy along Z" 
                        << endl << "IV.   Kinetic Energy (all species)"
                        << endl << "V.    Total Energy" 
                        << endl << "VI.   Energy(cycle) - Energy(initial)" 
                        << endl << "VII.  Momentum" << endl << endl;

                my_file << "=====================================================================================================================================" << endl << endl;

                my_file << setw(7) 
                        << "I"   << setw(25) 
                        << "II"  << setw(25) << "IIa"  << setw(25) << setw(25) << "IIb"  << setw(25) << setw(25) << "IIc"  << setw(25) 
                        << "III" << setw(25) << "IIIa" << setw(25) << setw(25) << "IIIb" << setw(25) << setw(25) << "IIIc" << setw(25) 
                        << "IV"  << setw(25) << "V"    << setw(25) << "VI" << setw(25) 
                        << "VII" << endl << endl;
            }
            
            if ((restart_status == 1 || restart_status == 2) && cycle == 0)
                    col->trim_conserved_quantities_file(cq, col->getLast_cycle() + 1);

            if (restart_status == 0 || ((restart_status == 1 || restart_status == 2) && cycle > 0))
            {
                my_file << setw(7)  << cycle << scientific << setprecision(15)
                        << setw(25) << E_field_energy << setw(25) << Ex_field_energy << setw(25) << Ey_field_energy  << setw(25) << Ez_field_energy 
                        << setw(25) << B_field_energy << setw(25) << Bx_field_energy << setw(25) << By_field_energy  << setw(25) << Bz_field_energy 
                        << setw(25) << kinetic_energy  
                        << setw(25) << E_field_energy + B_field_energy + kinetic_energy
                        << setw(25) << abs(initial_total_energy - (E_field_energy + B_field_energy + kinetic_energy))
                        << setw(25) << total_momentum << endl;
            }
            my_file.close();


            //? Species-specific quantities
            ofstream my_file_(cqs.c_str(), fstream::app);

            if(cycle == 0 && restart_status == 0)
            {
                my_file_ << endl << "I.   Cycle" 
                         << endl << "II.  Species" 
                         << endl << "III. Momentum (of each species)" 
                         << endl << "IV.  Total Kinetic Energy (of each species)"
                         << endl << "V.   Bulk Kinetic Energy (of each species)"
                         << endl << "VI.  Thermal Kinetic Energy (of each species)"  << endl << endl;

                my_file_ << "=====================================================================================================================================" << endl << endl;

                my_file_ << setw(7)  << "I"  << setw(7) << "II" << setw(25) << "III"
                         << setw(25) << "IV" << setw(25) << "V"  << setw(25) << "VI" 
                         << setw(25) << endl << endl;
            }

            if ((restart_status == 1 || restart_status == 2) && cycle == 0)
                col->trim_conserved_quantities_file(cqs, col->getLast_cycle() + 1);

            if (restart_status == 0 || ((restart_status == 1 || restart_status == 2) && cycle > 0))
            {
                for (int is = 0; is < ns; ++is) 
                {
                    my_file_ << setw(7) << cycle << setw(7) << is << scientific << setprecision(15)
                            << setw(25) << momentum_species[is] << setw(25) << kinetic_energy_species[is]
                            << setw(25) << bulk_energy_species[is] << setw(25) << kinetic_energy_species[is] - bulk_energy_species[is]
                            << endl;
                }
                my_file_ << endl;
            }
        }
    }
}

// This seems to record values at a grid of sample points
void c_Solver::WriteVirtualSatelliteTraces()
{
    if(ns <= 2) return;
    assert_eq(ns,4);

    ofstream my_file(cqsat.c_str(), fstream::app);
    const int nx0 = grid->get_nxc_r();
    const int ny0 = grid->get_nyc_r();
    const int nz0 = grid->get_nzc_r();

    for (int isat = 0; isat < nsat; isat++) 
        for (int jsat = 0; jsat < nsat; jsat++) 
            for (int ksat = 0; ksat < nsat; ksat++) 
            {
                int index1 = 1 + isat * nx0 / nsat + nx0 / nsat / 2;
                int index2 = 1 + jsat * ny0 / nsat + ny0 / nsat / 2;
                int index3 = 1 + ksat * nz0 / nsat + nz0 / nsat / 2;
                my_file << EMf->getBx(index1, index2, index3) << "\t" << EMf->getBy(index1, index2, index3) << "\t" << EMf->getBz(index1, index2, index3) << "\t";
                my_file << EMf->getEx(index1, index2, index3) << "\t" << EMf->getEy(index1, index2, index3) << "\t" << EMf->getEz(index1, index2, index3) << "\t";
                my_file << EMf->getJxs(index1, index2, index3, 0) + EMf->getJxs(index1, index2, index3, 2) << "\t" << EMf->getJys(index1, index2, index3, 0) + EMf->getJys(index1, index2, index3, 2) << "\t" << EMf->getJzs(index1, index2, index3, 0) + EMf->getJzs(index1, index2, index3, 2) << "\t";
                my_file << EMf->getJxs(index1, index2, index3, 1) + EMf->getJxs(index1, index2, index3, 3) << "\t" << EMf->getJys(index1, index2, index3, 1) + EMf->getJys(index1, index2, index3, 3) << "\t" << EMf->getJzs(index1, index2, index3, 1) + EMf->getJzs(index1, index2, index3, 3) << "\t";
                my_file << EMf->getRHOns(index1, index2, index3, 0) + EMf->getRHOns(index1, index2, index3, 2) << "\t";
                my_file << EMf->getRHOns(index1, index2, index3, 1) + EMf->getRHOns(index1, index2, index3, 3) << "\t";
            }

    my_file << endl;
    my_file.close();
}

//! HDF5 only
void c_Solver::WriteFields(int cycle) 
{
    #ifndef NO_HDF5
        if(col->field_output_is_off()) return;

        if (vct->getCartesian_rank() == 0)
            cout << endl << "Writing FIELD data at cycle " << cycle << endl;

        //* Fields
        if(!(col->getFieldOutputTag()).empty())
            fetch_outputWrapperFPP().append_output_fields((col->getFieldOutputTag()).c_str(), cycle, col->get_output_data_precision());      
        
        //* Moments
        if(!(col->getMomentsOutputTag()).empty())
            fetch_outputWrapperFPP().append_output_fields((col->getMomentsOutputTag()).c_str(), cycle, col->get_output_data_precision());
    #endif
}

void c_Solver::WriteParticles(int cycle)
{
    #ifndef NO_HDF5
        if(col->particle_output_is_off()) return;

        if (vct->getCartesian_rank() == 0)
            cout << endl << "Writing PARTICLE data at cycle " << cycle << endl;

        //* This is crucial
        convertParticlesToSynched();

        fetch_outputWrapperFPP().append_output_particles((col->getPclOutputTag()).c_str(), cycle, col->get_output_data_precision()); //* "position + velocity + q "
    #endif
}

void c_Solver::WriteDSParticles(int cycle)
{
    #ifndef NO_HDF5
        if(col->DS_particle_output_is_off() || col->getParticlesDownsampleFactor() <= 1) return;

        if (vct->getCartesian_rank() == 0)
            cout << endl << "Writing DOWNSAMPLED PARTICLE data at cycle " << cycle << endl;

        //* This is crucial
        convertParticlesToSynched();

        fetch_outputWrapperFPP().append_particles_DS((col->getPclDSOutputTag()).c_str(), cycle, col->getParticlesDownsampleFactor(), col->get_output_data_precision());
    #endif
}

void c_Solver::WriteTestParticles(int cycle)
{
    #ifndef NO_HDF5
        if(nstestpart == 0 || col->testparticle_output_is_off() || cycle%(col->getTestParticlesOutputCycle())!=0) return;

        //* This is crucial
        convertParticlesToSynched();

        fetch_outputWrapperFPP().append_output_particles("testpartpos + testpartvel+ testparttag", cycle, col->get_output_data_precision()); // + testpartcharge
    #endif
}

void c_Solver::WriteRestart(int cycle)
{
    #ifndef NO_HDF5
    if (restart_cycle > 0 && cycle%restart_cycle == 0 && cycle > 0)
    {
        if (myrank == 0)
            cout << endl << "Writing RESTART data at time cycle " << cycle << endl;

        //* This is crucial
        convertParticlesToSynched();

        fetch_outputWrapperFPP().append_restart(cycle, "DOUBLE");
    }
    #endif
}

//* =============================================================================================== *//

// This needs to be separated into methods that save particles and methods that save field data
void c_Solver::Finalize() 
{
    if (col->getCallFinalize() && Parameters::get_doWriteOutput())
    {
        #ifndef NO_HDF5
            convertParticlesToSynched();
            fetch_outputWrapperFPP().append_restart((col->getNcycles() + first_cycle), "DOUBLE");
        #endif
    }

    // stop profiling
    my_clock->stopTiming();
}

//? Place particles into new cells according to their current position
void c_Solver::sortParticles() 
{
    for(int species_idx = 0; species_idx < ns; species_idx++)
        particles[species_idx].sort_particles_serial();
}

void c_Solver::pad_particle_capacities()
{
    for (int i = 0; i < ns; i++)
        particles[i].pad_capacities();

    for (int i = 0; i < nstestpart; i++)
        testpart[i].pad_capacities();
}

//? Convert particle data to struct of arrays (assumed by I/O)
void c_Solver::convertParticlesToSoA()
{
    for (int i = 0; i < ns; i++)
        particles[i].convertParticlesToSoA();
}

//? Convert particle data to array of structs (used in computing)
void c_Solver::convertParticlesToAoS()
{
    for (int i = 0; i < ns; i++)
        particles[i].convertParticlesToAoS();
}

// convert particle to array of structs (used in computing)
void c_Solver::convertParticlesToSynched()
{
    for (int i = 0; i < ns; i++)
        particles[i].convertParticlesToSynched();

    for (int i = 0; i < nstestpart; i++)
        testpart[i].convertParticlesToSynched();
}

int c_Solver::LastCycle() 
{
    return (col->getNcycles() + first_cycle);
}