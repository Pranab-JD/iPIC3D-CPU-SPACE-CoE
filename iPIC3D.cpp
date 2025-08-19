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
            {  }
        #endif

        //? LeXInt timer
        LeXInt::timer time_IN, time_EF, time_PM, time_MF, time_MG, time_WD, time_total;
        time_total.start();

        iPic3D::c_Solver KCode;

        time_IN.start();
        KCode.Init(argc, argv); //! load param from file, init the grid, fields
        time_IN.stop();

        if(MPIdata::get_rank() == 0)
            std::cout << std::endl << "Time needed for initialisation: " << time_IN.total()   << " s" << std::endl;

        KCode.WriteOutput(KCode.FirstCycle());

        for (int i = KCode.FirstCycle() + 1; i <= KCode.LastCycle(); i++) 
        {
            if (KCode.get_myrank() == 0)
                std::cout << std::endl << "=================== Cycle " << i << " ===================" << std::endl ;
            
            //? Moment Gatherer --> Compute charge density, current density, and mass matrix
            time_MG.start();
            KCode.CalculateMoments();        
            time_MG.stop();
            
            //? Field Solver --> Compute E & B fields 
            time_EF.start();
            KCode.ComputeEMFields(i);        
            time_EF.stop();
            
            //? Particle Pusher --> Compute new velocities and positions of the particles
            time_PM.start();
            KCode.ParticlesMover();          
            time_PM.stop();

            time_WD.start();
            KCode.WriteOutput(i);
            time_WD.stop();

            if(MPIdata::get_rank() == 0)
            {
                std::cout << std::endl << "Runtime of iPIC3D modules " << std::endl;
                std::cout << "Field solver       : " << time_EF.total()   << " s" << std::endl;
                std::cout << "Particle mover     : " << time_PM.total()   << " s" << std::endl;
                std::cout << "Moment gatherer    : " << time_MG.total()   << " s" << std::endl;
                std::cout << "Write data         : " << time_WD.total()   << " s" << std::endl;
            }
        }

        time_total.stop();

        KCode.Finalize();
    }

    //? close MPI
    MPIdata::instance().finalize_mpi();

    return 0;
}
