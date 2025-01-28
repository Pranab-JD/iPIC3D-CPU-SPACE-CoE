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
//
#ifndef NO_HDF5
#include "WriteOutputParallel.h"
#include "OutputWrapperFPP.h"
#endif

#include <iostream>
#include <fstream>
#include <sstream>

#include "Moments.h" // for debugging

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
  // delete particles
  //
  if(part)
  {
    for (int i = 0; i < ns; i++)
    {
      // placement delete
      part[i].~Particles3D();
    }
    free(part);
  }

#ifdef USE_CATALYST
  Adaptor::Finalize();
#endif
  delete [] Ke;
  delete [] momentum;
  delete [] Qremoved;
  delete my_clock;
}

int c_Solver::Init(int argc, char **argv) 
{
    #if defined(__MIC__)
        assert_eq(DVECWIDTH,8);
    #endif

    Parameters::init_parameters();
    //mpi = &MPIdata::instance();
    nprocs = MPIdata::get_nprocs();
    myrank = MPIdata::get_rank();

    col = new Collective(argc, argv);               // Every proc loads the parameters of simulation from class Collective
    restart_cycle   = col->getRestartOutputCycle();
    SaveDirName     = col->getSaveDirName();
    RestartDirName  = col->getRestartDirName();
    restart_status  = col->getRestart_status();
    ns              = col->getNs();                 // get the number of species of particles involved in simulation
    first_cycle     = col->getLast_cycle() + 1;     // get the last cycle from the restart

    vct = new VCtopology3D(*col);

    //? Check if we can map the processes into a matrix ordering defined in Collective.cpp
    if (nprocs != vct->getNprocs()) 
    {
        if (myrank == 0) 
        {
            cerr << "Error: " << nprocs << " processes cant be mapped as " << vct->getXLEN() << "x" << vct->getYLEN() << "x" << vct->getZLEN() << ". Change XLEN, YLEN, & ZLEN in input file. " << endl;
            MPIdata::instance().finalize_mpi();
            return (1);
        }
    }

    // We create a new communicator with a 3D virtual Cartesian topology
    vct->setup_vctopology(MPIdata::get_PicGlobalComm());
    {
        stringstream num_proc_ss;
        num_proc_ss << vct->getCartesian_rank();
        num_proc_str = num_proc_ss.str();
    }
    
    // initialize the central cell index
    #ifdef BATSRUS
        // set index offset for each processor
        col->setGlobalStartIndex(vct);
    #endif

    //* Print initial settings to stdout and file
    if (myrank == 0) 
    {
        //* Create output directory
        checkOutputFolder(SaveDirName);
        MPIdata::instance().Print();
        vct->Print();
        col->Print();
        col->save();
    }

    //* Create local grid
    grid = new Grid3DCU(col, vct);          // Create the local grid
    EMf = new EMfields3D(col, grid, vct);   // Create Electromagnetic Fields Object

    if      (col->getCase()=="GEMnoPert") 		EMf->initGEMnoPert();
    else if (col->getCase()=="ForceFree") 		EMf->initForceFree();
    else if (col->getCase()=="GEM")       		EMf->initGEM();
    else if (col->getCase()=="GEMDoubleHarris") EMf->initGEMDoubleHarris();
    else if (col->getCase()=="Dipole")    		EMf->initDipole();
    else if (col->getCase()=="Dipole2D")  		EMf->initDipole2D();
    else if (col->getCase()=="NullPoints")      EMf->initNullPoints();
    else if (col->getCase()=="TaylorGreen")     EMf->initTaylorGreen();
    #ifdef BATSRUS
        else if (col->getCase()=="BATSRUS")   	EMf->initBATSRUS();
    #endif
    else if (col->getCase()=="RandomCase") 
    {
        EMf->initRandomField();
        if (myrank==0) 
        {
            cout << "Case is " << col->getCase() <<"\n";
            cout <<"total # of particle per cell is " << col->getNpcel(0) << "\n";
        }
    }
    else 
    {
    if (myrank==0) 
        {
            cout << " =========================================================== " << endl;
            cout << " WARNING: The case '" << col->getCase() << "' was not recognized. " << endl;
            cout << "          Runing simulation with the default initialization. " << endl;
            cout << " =========================================================== " << endl;
        }

        EMf->init();
    }

    //? Allocation of particles
    part = (Particles3D*) malloc(sizeof(Particles3D)*ns);
    for (int i = 0; i < ns; i++)
    {
        new(&part[i]) Particles3D(i,col,vct,grid);
    }

    //? Initial Condition for particles (if NOT starting from RESTART)
    if (restart_status == 0) 
    {
        for (int i = 0; i < ns; i++)
        {
            if      (col->getCase()=="ForceFree") 		part[i].force_free(EMf);
            #ifdef BATSRUS
                else if (col->getCase()=="BATSRUS")   	part[i].MaxwellianFromFluid(EMf,col,i);
            #endif
            else if (col->getCase()=="NullPoints")    	part[i].maxwellianNullPoints(EMf);
            else if (col->getCase()=="TaylorGreen")     part[i].maxwellianNullPoints(EMf); // Flow is initiated from the current prescribed on the grid.
            else if (col->getCase()=="GEMDoubleHarris") part[i].maxwellianDoubleHarris(EMf);
            else                                  		part[i].maxwellian(EMf);
            
            part[i].reserve_remaining_particle_IDs();
        }
    }

    //? Allocate test particles (if any)
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

    if (Parameters::get_doWriteOutput())
    {
        #ifndef NO_HDF5
        if(col->getWriteMethod() == "shdf5" || col->getCallFinalize() || restart_cycle>0 || (col->getWriteMethod()=="pvtk" && !col->particle_output_is_off()) )
        {
            outputWrapperFPP = new OutputWrapperFPP;
            fetch_outputWrapperFPP().init_output_files(col,vct,grid,EMf,part,ns,testpart,nstestpart);
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

    Ke = new double[ns];
    BulkEnergy = new double[ns];
    momentum = new double[ns];
    cq = SaveDirName + "/ConservedQuantities.txt";
    if (myrank == 0) 
    {
        ofstream my_file(cq.c_str());
        my_file.close();
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

    pad_particle_capacities();

    //* Vectorised; assumes that particles are sorted by mesh cell
    if(Parameters::get_VECTORIZE_MOMENTS())
    {
        switch(Parameters::get_MOMENTS_TYPE())
        {
            case Parameters::SoA:
            // since particles are sorted,we can vectorize interpolation of particles to grid
            convertParticlesToSoA();
            sortParticles();
            EMf->sumMoments_vectorized(part);
            break;
            
            case Parameters::AoS:
            convertParticlesToAoS();
            sortParticles();
            EMf->sumMoments_vectorized_AoS(part);
            break;
            
            default:
            unsupported_value_error(Parameters::get_MOMENTS_TYPE());
        }
    }
    else
    {
        if(Parameters::get_SORTING_PARTICLES())
            sortParticles();
        
        switch(Parameters::get_MOMENTS_TYPE())
        {
            case Parameters::SoA:
            EMf->setZeroPrimaryMoments();
            convertParticlesToSoA();
            EMf->sumMoments(part);
            break;
            
            case Parameters::AoS:
            EMf->setZeroPrimaryMoments();   // clear the data to 0
            convertParticlesToAoS();        // convert 
            EMf->sumMoments_AoS(part);      // sum up the 10 densities of each particles of each species
            // then calculate the weight according to their position; map the 10 momentum to the grid(node) with the weight
            break;
            
            case Parameters::AoSintr:
            EMf->setZeroPrimaryMoments();
            convertParticlesToAoS();
            EMf->sumMoments_AoS_intr(part);
            break;
            
            default:
            unsupported_value_error(Parameters::get_MOMENTS_TYPE());
        }
    }

    EMf->setZeroDensities();

    // TODO: Do we need (0) and (1) separately? - Ask Fabio
    EMf->setZeroRho(0);

    //? Interpolate Particles to grid (nodes)
    for (int i = 0; i < ns; i++)
		part[i].computeMoments(EMf);

    //TODO: Implement this fully and correctly (for each species)
    EMf->communicateGhostP2G(vct);
    
    //* Sum all over the species (mass and charge density)
    EMf->sumOverSpecies();
    EMf->interpolateCenterSpecies();
    
    // TODO: These following 2 lines are in both sumOverSpecies() and interpolateCenterSpecies(). Do we need these twice? - Ask Fabio
    // TODO:    grid->interpN2C(rhoc, rhon);
    // TODO:    communicateCenterBC(nxc, nyc, nzc, rhoc, 2, 2, 2, 2, 2, 2, vct);

    //TODO: Communication for 4D array (EMf->interpolateCenterSpecies(), setZeroRho(0)) is not included (not for all functions in ECSim) Why? - Ask Fabio

    EMf->timeAveragedRho(col->getPoissonMArho());
	EMf->timeAveragedDivE(col->getPoissonMAdiv());
}

//! Compute electric field
void c_Solver::ComputeE(int cycle) 
{
    timeTasks_set_main_task(TimeTasks::FIELDS);

    EMf->calculateE(cycle);
}

//! Compute magnetic field (assuming electric field has already been calculated)
void c_Solver::CalculateB() 
{
    timeTasks_set_main_task(TimeTasks::FIELDS);

    EMf->calculateB();
}

//! Compute electromagnetic field (E and B needs to computed inside 1 function for ECSim)
void c_Solver::ComputeEMFields(int cycle)
{
    //TODO: TBD later
    // if (col->getLambdaDamping() == "piston")
		// EMf->evolveLambda(vct, grid, col, piston);

    //TODO: Only needed for the cases "Shock1D_DoublePiston" and "LangevinAntenna" - Should I implement this? Ask Fabio
	//* Update external fields to n+1/2
	// EMf->updateExternalFields(vct, grid, col, cycle); 

    //TODO: Only needed for the cases "ForcedDynamo" - Should I implement this? Ask Fabio
	//* Update particle external forces to n+1/2
	// EMf->updateParticleExternalForces(vct, grid, col, cycle); 

    //? Compute E and B fields
    EMf->calculateE(cycle);
    EMf->calculateB();

    //TODO: TBD later
    // Impose MHD Buffer Zone
    // if (col->getCase()=="MHDUCLA" || col->getCase()=="AnnularHarris" || col->getCase()=="GEM" || col->getCase()=="Discontinuity" || col->getCase()=="ForceFree") 
    // {
    //     if (col->getLambdaField() == "yes")
    //         EMf->setMHDBufferZone(grid, vct, col);
    // }

    //* Compute divergence of E and B and accessory variable
	EMf->timeAveragedDivE(col->getPoissonMAdiv());
	EMf->divergenceOfE(col->getPoissonMAres());
	EMf->divergenceOfB();

    //TODO: TBD later
    // double dt = col->getDt();
	// if (col->getTimeFile() != "none")
	// 	EMf->updateReferenceStateFromFile(grid, col, vct, cycle * dt);
}

//! Compute positions and velocities of particles
bool c_Solver::ParticlesMover()
{
    //? Move all species of particles
    {
        timeTasks_set_main_task(TimeTasks::PARTICLES);
        // Should change this to add background field
        EMf->set_fieldForPcls();

        pad_particle_capacities();

        //* Iterate over each species to update velocities
        for (int i = 0; i < ns; i++)
        {
            switch(Parameters::get_MOVER_TYPE())
            {
                //? ECSim
                case Parameters::SoA:
                    part[i].ECSIM_velocity(EMf);
                break;
                
                default:
                unsupported_value_error(Parameters::get_MOVER_TYPE());
            }

            //* Should integrate BC into separate_and_send_particles
            part[i].openbc_particles_outflow();
            part[i].separate_and_send_particles();
        }

        //* Iterate over each species to update positions
        for (int i = 0; i < ns; i++)
        {
            switch(Parameters::get_MOVER_TYPE())
            {
                case Parameters::SoA:
                    part[i].ECSIM_position(EMf);
                break;
            }

            //* Should integrate BC into separate_and_send_particles
            part[i].openbc_particles_outflow();
            part[i].separate_and_send_particles();
        }

        //* Communicate each species
        for (int i = 0; i < ns; i++)  
            part[i].recommunicate_particles_until_done(1);
    }

    //? Repopulate the buffer zone at the edge
    for (int i=0; i < ns; i++) 
    {
        if (col->getRHOinject(i)>0.0)
            part[i].repopulate_particles();
    }

    //? Remove particles from depopulation area
    if (col->getCase()=="Dipole") 
    {
        for (int i=0; i < ns; i++)
            Qremoved[i] = part[i].deleteParticlesInsideSphere(col->getL_square(), col->getx_center(), col->gety_center(), col->getz_center());
    }
    else if (col->getCase()=="Dipole2D") 
    {
        for (int i=0; i < ns; i++)
            Qremoved[i] = part[i].deleteParticlesInsideSphere2DPlaneXZ(col->getL_square(), col->getx_center(), col->getz_center());
    }

    //* =============== Test Particles =============== *//

    for (int i = 0; i < nstestpart; i++)
    {
        switch(Parameters::get_MOVER_TYPE())
        {
            case Parameters::SoA:
                testpart[i].ECSIM_velocity(EMf);
            break;
            
            default:
                unsupported_value_error(Parameters::get_MOVER_TYPE());
        }

        testpart[i].openbc_delete_testparticles();
        testpart[i].separate_and_send_particles();
    }

    for (int i = 0; i < nstestpart; i++)
    {
        switch(Parameters::get_MOVER_TYPE())
        {
            case Parameters::SoA:
                testpart[i].ECSIM_position(EMf);
            break;
            
            default:
                unsupported_value_error(Parameters::get_MOVER_TYPE());
        }

        testpart[i].openbc_delete_testparticles();
        testpart[i].separate_and_send_particles();
    }

    for (int i = 0; i < nstestpart; i++)
        testpart[i].recommunicate_particles_until_done(1);

    //* ============================================== *//

    return (false);
}

//* ===================================== WRITE DATA TO FILES ===================================== *//

void c_Solver::WriteOutput(int cycle) 
{
    #ifdef USE_CATALYST
        Adaptor::CoProcess(col->getDt()*cycle, cycle, EMf);
    #endif

    WriteConserved(cycle);
    WriteRestart(cycle);

    if(!Parameters::get_doWriteOutput())  return;

//! Non-blocking collective MPI-IO
  if (col->getWriteMethod() == "nbcvtk")
  {

	  if(!col->field_output_is_off() && (cycle%(col->getFieldOutputCycle()) == 0 || cycle == first_cycle) ){
		  if(!(col->getFieldOutputTag()).empty()){

			  if(fieldreqcounter>0){
			          
			          //MPI_Waitall(fieldreqcounter,&fieldreqArr[0],&fieldstsArr[0]);
				  for(int si=0;si< fieldreqcounter;si++){
				    int error_code = MPI_File_write_all_end(fieldfhArr[si],&fieldwritebuffer[si][0][0][0],&fieldstsArr[si]);//fieldstsArr[si].MPI_ERROR;
					  if (error_code != MPI_SUCCESS) {
						  char error_string[100];
						  int length_of_error_string, error_class;
						  MPI_Error_class(error_code, &error_class);
						  MPI_Error_string(error_class, error_string, &length_of_error_string);
						  dprintf("MPI_Waitall error at field output cycle %d  %d  %s\n",cycle, si, error_string);
					  }else{
						  MPI_File_close(&(fieldfhArr[si]));
					  }
				  }
			  }
			  fieldreqcounter = WriteFieldsVTKNonblk(grid, EMf, col, vct,cycle,fieldwritebuffer,fieldreqArr,fieldfhArr);
		  }

		  if(!(col->getMomentsOutputTag()).empty()){

			  if(momentreqcounter>0){
			    //MPI_Waitall(momentreqcounter,&momentreqArr[0],&momentstsArr[0]);
				  for(int si=0;si< momentreqcounter;si++){
				    int error_code = MPI_File_write_all_end(momentfhArr[si],&momentwritebuffer[si][0][0],&momentstsArr[si]);//momentstsArr[si].MPI_ERROR;
					  if (error_code != MPI_SUCCESS) {
						  char error_string[100];
						  int length_of_error_string, error_class;
						  MPI_Error_class(error_code, &error_class);
						  MPI_Error_string(error_class, error_string, &length_of_error_string);
						  dprintf("MPI_Waitall error at moments output cycle %d  %d %s\n",cycle, si, error_string);
					  }else{
						  MPI_File_close(&(momentfhArr[si]));
					  }
				  }
			  }
			  momentreqcounter = WriteMomentsVTKNonblk(grid, EMf, col, vct,cycle,momentwritebuffer,momentreqArr,momentfhArr);
		  }
	  }

	  //Particle information is still in hdf5
	  	WriteParticles(cycle);
	  //Test Particle information is still in hdf5
	    WriteTestParticles(cycle);

  }else if (col->getWriteMethod() == "pvtk"){//Blocking collective MPI-IO
	  if(!col->field_output_is_off() && (cycle%(col->getFieldOutputCycle()) == 0 || cycle == first_cycle) ){
		  if(!(col->getFieldOutputTag()).empty()){
			  //WriteFieldsVTK(grid, EMf, col, vct, col->getFieldOutputTag() ,cycle);//B + E + Je + Ji + rho
			  WriteFieldsVTK(grid, EMf, col, vct, col->getFieldOutputTag() ,cycle, fieldwritebuffer);//B + E + Je + Ji + rho
		  }
		  if(!(col->getMomentsOutputTag()).empty()){
			  WriteMomentsVTK(grid, EMf, col, vct, col->getMomentsOutputTag() ,cycle, momentwritebuffer);
		  }
	  }

	  //Particle information is still in hdf5
	  	WriteParticles(cycle);
	  //Test Particle information is still in hdf5
	    WriteTestParticles(cycle);

  }else{

		#ifdef NO_HDF5
			eprintf("The selected output option must be compiled with HDF5");

		#else
			if (col->getWriteMethod() == "H5hut"){

			  if (!col->field_output_is_off() && cycle%(col->getFieldOutputCycle())==0)
				WriteFieldsH5hut(ns, grid, EMf, col, vct, cycle);
			  if (!col->particle_output_is_off() && cycle%(col->getParticlesOutputCycle())==0)
				WritePartclH5hut(ns, grid, part, col, vct, cycle);

			}else if (col->getWriteMethod() == "phdf5"){

			  if (!col->field_output_is_off() && cycle%(col->getFieldOutputCycle())==0)
				WriteOutputParallel(grid, EMf, part, col, vct, cycle);

			  if (!col->particle_output_is_off() && cycle%(col->getParticlesOutputCycle())==0)
			  {
				if(MPIdata::get_rank()==0)
				  warning_printf("WriteParticlesParallel() is not yet implemented.");
			  }

			}else if (col->getWriteMethod() == "shdf5"){

					WriteFields(cycle);

					WriteParticles(cycle);

					WriteTestParticles(cycle);

			}else{
			  warning_printf(
				"Invalid output option. Options are: H5hut, phdf5, shdf5, pvtk");
			  invalid_value_error(col->getWriteMethod().c_str());
			}
		#endif
  	  }
}

void c_Solver::WriteRestart(int cycle)
{
    #ifndef NO_HDF5
        if (restart_cycle>0 && cycle%restart_cycle==0)
        {
            convertParticlesToSynched();
            fetch_outputWrapperFPP().append_restart(cycle);
        }
    #endif
}

void c_Solver::WriteConserved(int cycle) 
{
    if(col->getDiagnosticsOutputCycle() > 0 && cycle % col->getDiagnosticsOutputCycle() == 0)
    {
        Eenergy = EMf->getEenergy();
        Benergy = EMf->getBenergy();
        TOTenergy = 0.0;
        TOTmomentum = 0.0;
        
        for (int is = 0; is < ns; is++) 
        {
            Ke[is] = part[is].getKe();
            BulkEnergy[is] = EMf->getBulkEnergy(is);
            TOTenergy += Ke[is];
            momentum[is] = part[is].getP();
            TOTmomentum += momentum[is];
        }
        
        if (myrank == (nprocs-1)) 
        {
            ofstream my_file(cq.c_str(), fstream::app);
            if(cycle == 0) my_file << "\t" << "\t" << "\t" << "Total_Energy" << "\t" << "Momentum" << "\t" << "Eenergy" << "\t" << "Benergy" << "\t" << "Kenergy" << "\t" << "Kenergy(species)" << "\t" << "BulkEnergy(species)" << endl;
            my_file << cycle << "\t" << "\t" << (Eenergy + Benergy + TOTenergy) << "\t" << TOTmomentum << "\t" << Eenergy << "\t" << Benergy << "\t" << TOTenergy;
            for (int is = 0; is < ns; is++) my_file << "\t" << Ke[is];
            for (int is = 0; is < ns; is++) my_file << "\t" << BulkEnergy[is];
            my_file << endl;
            my_file.close();
        }
    }
}

void c_Solver::WriteVelocityDistribution(int cycle)
{
  // Velocity distribution
  //if(cycle % col->getVelocityDistributionOutputCycle() == 0)
  {
    for (int is = 0; is < ns; is++) {
      double maxVel = part[is].getMaxVelocity();
      long long *VelocityDist = part[is].getVelocityDistribution(nDistributionBins, maxVel);
      if (myrank == 0) {
        ofstream my_file(ds.c_str(), fstream::app);
        my_file << cycle << "\t" << is << "\t" << maxVel;
        for (int i = 0; i < nDistributionBins; i++)
          my_file << "\t" << VelocityDist[i];
        my_file << endl;
        my_file.close();
      }
      delete [] VelocityDist;
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

void c_Solver::WriteFields(int cycle) 
{
    #ifndef NO_HDF5
        if(col->field_output_is_off()) return;

        if(cycle % (col->getFieldOutputCycle()) == 0 || cycle == first_cycle)
        {
            //* E + B + Js
            if(!(col->getFieldOutputTag()).empty())
                fetch_outputWrapperFPP().append_output((col->getFieldOutputTag()).c_str(), cycle);      
            
            //* rhos + pressure
            if(!(col->getMomentsOutputTag()).empty())
                fetch_outputWrapperFPP().append_output((col->getMomentsOutputTag()).c_str(), cycle);
        }
    #endif
}

void c_Solver::WriteParticles(int cycle)
{
    #ifndef NO_HDF5
        if(col->particle_output_is_off() || cycle%(col->getParticlesOutputCycle())!=0) return;

        // this is a hack
        for (int i = 0; i < ns; i++)
            part[i].convertParticlesToSynched();

        fetch_outputWrapperFPP().append_output((col->getPclOutputTag()).c_str(), cycle, 0);//"position + velocity + q "
    #endif
}

void c_Solver::WriteTestParticles(int cycle)
{
    #ifndef NO_HDF5
        if(nstestpart == 0 || col->testparticle_output_is_off() || cycle%(col->getTestParticlesOutputCycle())!=0) return;

        // this is a hack
        for (int i = 0; i < nstestpart; i++)
            testpart[i].convertParticlesToSynched();

        fetch_outputWrapperFPP().append_output("testpartpos + testpartvel+ testparttag", cycle, 0); // + testpartcharge
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
            fetch_outputWrapperFPP().append_restart((col->getNcycles() + first_cycle) - 1);
        #endif
    }

    // stop profiling
    my_clock->stopTiming();
}

//! place the particles into new cells according to their current position
void c_Solver::sortParticles() 
{
    for(int species_idx=0; species_idx<ns; species_idx++)
        part[species_idx].sort_particles_serial();
}

void c_Solver::pad_particle_capacities()
{
    for (int i = 0; i < ns; i++)
        part[i].pad_capacities();

    for (int i = 0; i < nstestpart; i++)
        testpart[i].pad_capacities();
}

// convert particle to struct of arrays (assumed by I/O)
void c_Solver::convertParticlesToSoA()
{
    for (int i = 0; i < ns; i++)
        part[i].convertParticlesToSoA();
}

// convert particle to array of structs (used in computing)
void c_Solver::convertParticlesToAoS()
{
    for (int i = 0; i < ns; i++)
        part[i].convertParticlesToAoS();
}

// convert particle to array of structs (used in computing)
void c_Solver::convertParticlesToSynched()
{
    for (int i = 0; i < ns; i++)
        part[i].convertParticlesToSynched();

    for (int i = 0; i < nstestpart; i++)
        testpart[i].convertParticlesToSynched();
}

int c_Solver::LastCycle() 
{
    return (col->getNcycles() + first_cycle);
}