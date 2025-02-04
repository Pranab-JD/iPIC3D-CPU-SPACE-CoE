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

#include <mpi.h>
#include "stdio.h"
#include "Timing.h"
#include "ipicdefs.h"
#include "MPIdata.h"
/**
 * 
 * series of methods for timing and profiling
 * @date Fri Jun 4 2007
 * @author Stefano Markidis, Giovanni Lapenta
 * @version 2.0
 *
 */

/** default constructor */
Timing::Timing() {
}

/** constructor with the initialization of the log file */
Timing::Timing(int my_rank) 
{
    rank_id = my_rank;

    startTiming();

    former_MPI_Barrier(MPIdata::get_PicGlobalComm());
}

/** start the timer */
void Timing::startTiming() 
{
    ttick = MPI_Wtick();
    former_MPI_Barrier(MPIdata::get_PicGlobalComm());
    tstart = MPI_Wtime();
}

/** stop the timer */
void Timing::stopTiming() 
{
    former_MPI_Barrier(MPIdata::get_PicGlobalComm());
    tend = MPI_Wtime();
    texecution = tend - tstart;
    if (rank_id == 0) 
    {
        printf( "\n*** SIMULATION COMPLETED SUCESSFULLY! ***\n\n"
                "Simulation Runtime: %g sec (%g hours)\n", texecution, texecution / 3600);

    }

}