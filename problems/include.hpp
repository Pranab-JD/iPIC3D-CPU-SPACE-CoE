#pragma once

#include <mpi.h>
#include <cstring>
#include <iomanip>
#include <fstream>
#include <iostream>
#include <filesystem>

#include "CG.h"
#include "Basic.h"
#include "Alloc.h"
#include "debug.h"
#include "GMRES.h"
#include "string.h"
#include "ompdefs.h"
#include "Moments.h"
#include "asserts.h"
#include "Grid3DCU.h"
#include "ipichdf5.h"
#include "ipicmath.h"
#include "TimeTasks.h"
#include "EMfields3D.h"
#include "Collective.h"
#include "Parameters.h"
#include "Com3DNonblk.h"
#include "VCtopology3D.h"
#include "mic_particles.h"
#include "Particles3Dcomm.h"

#include "../LeXInt_Timer.hpp"

////? ============================================================= ?////

////? Some Generic Functions

//* sech^2(x) up to arbitrary precision
double sech_square(double x) 
{
    double y, res;
  
    if (fabs(x) > 354.0) 
        res = 1.31e-307;
    else 
    {                                                                                    
        y = 1.0/cosh(x);
        res = y*y;
    }
    return res;
}