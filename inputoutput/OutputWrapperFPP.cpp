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
#include "OutputWrapperFPP.h"
#include "VCtopology3D.h"
#include "Grid3DCU.h"
#include "EMfields3D.h"
#include "Particles3D.h"

void OutputWrapperFPP::init_output_files(Collective *col, VCtopology3D *vct, Grid3DCU *grid, EMfields3D *EMf, 
                                         Particles3D *part, int ns, Particles3D *testpart, int nstestpart)
{
    #ifndef NO_HDF5
        cartesian_rank = vct->getCartesian_rank();
        stringstream num_proc_ss;
        num_proc_ss << cartesian_rank;
        string num_proc_str = num_proc_ss.str();
        SaveDirName = col->getSaveDirName();
        RestartDirName = col->getRestartDirName();
        int restart_status = col->getRestart_status();

        output_file = SaveDirName + "/proc" + num_proc_str + ".hdf";
        restart_file = RestartDirName + "/restart" + num_proc_str + ".hdf";

        //* Initialize the output (simulation results and restart file)
        hdf5_agent.set_simulation_pointers(EMf, grid, vct, col);

        for (int i = 0; i < ns; ++i)
            hdf5_agent.set_simulation_pointers_part(&part[i]);

        for (int i = 0; i < nstestpart; ++i)
            hdf5_agent.set_simulation_pointers_part(&testpart[i]);

        //* Add the HDF5 output agent to the Output Manager's list
        output_mgr.push_back(&hdf5_agent);

        if(col->getWriteMethod() == "shdf5" || (col->getWriteMethod() == "pvtk" && !col->particle_output_is_off()))
        {
            if (cartesian_rank == 0 && restart_status < 2) 
            {
                hdf5_agent.open(SaveDirName + "/settings.hdf");
                output_mgr.output("collective + total_topology + proc_topology", 0);
                hdf5_agent.close();
            }
        }

        if (col->getWriteMethod() == "shdf5")
            if(col->getCallFinalize() || col->getRestartOutputCycle()>0)
            {
                if (cartesian_rank == 0 && restart_status < 2) 
                {
                    hdf5_agent.open(RestartDirName + "/settings.hdf");
                    output_mgr.output("collective + total_topology + proc_topology", 0);
                    hdf5_agent.close();
                }

                if (restart_status == 0)
                {
                    hdf5_agent.open(output_file);
                    output_mgr.output("proc_topology ", 0);
                    hdf5_agent.close();
                }
                else 
                {   
                    // restart, append the results to the previous simulation 
                    hdf5_agent.open_append(restart_file);
                    output_mgr.output("proc_topology ", 0);
                    hdf5_agent.close();
                }
            }
    #endif
}

//* Particles + Fields + Moments
void OutputWrapperFPP::append_output_fields(const char* tag, int cycle, string precision)
{
    #ifndef NO_HDF5
        hdf5_agent.open_append(output_file);
        output_mgr.output_fields(tag, cycle, precision);
        hdf5_agent.close();
    #endif
}

//* Particles + Fields + Moments
void OutputWrapperFPP::append_output_particles(const char* tag, int cycle, string precision)
{
    #ifndef NO_HDF5
        hdf5_agent.open_append(output_file);
        output_mgr.output_particles(tag, cycle, precision);
        hdf5_agent.close();
    #endif
}

//* Downsampled Particles
void OutputWrapperFPP::append_particles_DS(const char* tag, int cycle, int sample, string precision)
{
    #ifndef NO_HDF5
        hdf5_agent.open_append(output_file);
        output_mgr.output_particles_DS(tag, cycle, sample, precision);
        hdf5_agent.close();
    #endif
}

//* Restart data
void OutputWrapperFPP::append_restart(int cycle, string precision)
{
    #ifndef NO_HDF5
        hdf5_agent.open_append(restart_file);
        output_mgr.output("proc_topology ", cycle);
        output_mgr.output_fields("E + B + B_c + divergence", cycle, precision); 
        output_mgr.output_particles("position + velocity + q", cycle, precision);
        // output_mgr.output("testpartpos + testpartvel + testpartcharge", cycle);
        output_mgr.output("last_cycle", cycle);
        hdf5_agent.close();
    #endif
}