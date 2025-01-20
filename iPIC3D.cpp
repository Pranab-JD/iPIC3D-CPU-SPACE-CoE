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


#include "MPIdata.h"
#include "iPic3D.h"
#include "debug.h"
#include "TimeTasks.h"
#include <stdio.h>

#include "LeXInt_Timer.hpp"

using namespace iPic3D;

int main(int argc, char **argv) 
{
    MPIdata::init(&argc, &argv);
    {
        #ifdef DEBUG_MODE
            //DBG
            int volatile j = 0;
            while(j == 0)
            {

            }
        #endif

        iPic3D::c_Solver KCode;
        KCode.Init(argc, argv); //! load param from file, init the grid, fields

        timeTasks.resetCycle(); //reset timer
        KCode.CalculateMoments();

        //? Use LeXInt timer
        LeXInt::timer time_EF, time_PM, time_MF, time_MG, time_loop, time_total;
        time_total.start();

        for (int i = KCode.FirstCycle(); i < KCode.LastCycle(); i++) 
        {
            time_loop.start();

            if (KCode.get_myrank() == 0)
                printf("================ Cycle %d ================ \n",i);

            timeTasks.resetCycle();
            
            time_EF.start();
            KCode.ComputeEMFields(i);        //* E & B fields
            time_EF.stop();
            
            time_PM.start();
            KCode.ParticlesMover();         //* Compute the new v and x for particles using the fields
            time_PM.stop();

            // time_MF.start();
            // KCode.CalculateB();             //* B field
            // time_MF.stop();

            time_MG.start();
            KCode.CalculateMoments();       //* Charge density, current density, and pressure tensor
            time_MG.stop();

            //calculated from particles position and celocity, then mapped to node(grid) for further solving
            // some are mapped to cell center

            KCode.WriteOutput(i);
            
            //? Print out total time for all tasks
            #ifdef LOG_TASKS_TOTAL_TIME
                timeTasks.print_cycle_times(i);
            #endif

            time_loop.stop();

            if(MPIdata::get_rank() == 0)
            {
                std::cout << std::endl << "LeXInt timer (cummulative) " << std::endl;
                std::cout << "Field solver       : " << time_EF.total() << " s" << std::endl;
                std::cout << "Particle mover     : " << time_PM.total() << " s" << std::endl;
                // std::cout << "Magnetic field     : " << time_MF.total() << " s" << std::endl;
                std::cout << "Moment gatherer    : " << time_MG.total() << " s" << std::endl;
                std::cout << "Cycle time         : " << time_loop.total() << " s" << std::endl << std::endl;
            }
        }

        time_total.stop();

        #ifdef LOG_TASKS_TOTAL_TIME
            timeTasks.print_tasks_total_times();
        #endif

        if(MPIdata::get_rank() == 0)
            std::cout << std::endl << "Total time (LeXInt timer) : " << time_total.total() << std::endl << std::endl;

        KCode.Finalize();
    }

    //? close MPI
    MPIdata::instance().finalize_mpi();

    return 0;
}
