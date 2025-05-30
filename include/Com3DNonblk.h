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
  Library for Nonblocking Halo Exchange
  -------------------
begin                : Feb 2015
copyright            : KTH
developers           : Stefano Markidis, Ivy Bo Peng

 ***************************************************************************/

#ifndef Com3DNonblk_H
#define Com3DNonblk_H

#include "arraysfwd.h"
#include "ipicfwd.h"
#include "mpi.h"
#include "BcFields3D.h"
#include "VCtopology3D.h"
#include "ipicdefs.h"
#include "Alloc.h"
#include "debug.h"
#include "EMfields3D.h"
#include "ComParser3D.h"

//* Communicate ghost cells (nodes)
void communicateNodeBC( int nx, int ny, int nz, arr3_double vector, 
                        int bcFaceXright, int bcFaceXleft, int bcFaceYright, int bcFaceYleft, int bcFaceZright, int bcFaceZleft, 
                        const VirtualTopology3D * vct, EMfields3D *EMf);

void communicateNodeBC_old( int nx, int ny, int nz, double ***vector, 
                            int bcFaceXright, int bcFaceXleft, int bcFaceYright, int bcFaceYleft, int bcFaceZright, int bcFaceZleft, 
                            const VirtualTopology3D *vct, EMfields3D *EMf);

void communicateNodeBoxStencilBC(int nx, int ny, int nz, arr3_double vector, 
                                int bcFaceXright, int bcFaceXleft, int bcFaceYright, int bcFaceYleft, int bcFaceZright, int bcFaceZleft, 
                                const VirtualTopology3D * vct, EMfields3D *EMf);

void communicateNodeBoxStencilBC_P( int nx, int ny, int nz, arr3_double vector, 
                                    int bcFaceXright, int bcFaceXleft, int bcFaceYright, int bcFaceYleft, int bcFaceZright, int bcFaceZleft, 
                                    const VirtualTopology3D * vct, EMfields3D *EMf);

void communicateNodeBC_P(int nx, int ny, int nz, arr3_double vector, 
                        int bcFaceXright, int bcFaceXleft, int bcFaceYright, int bcFaceYleft, int bcFaceZright, int bcFaceZleft, 
                        const VirtualTopology3D * vct, EMfields3D *EMf);


//* Communication + BC
void communicateCenterBC(int nx, int ny, int nz, arr3_double vector, 
                        int bcFaceXright, int bcFaceXleft, int bcFaceYright, int bcFaceYleft, int bcFaceZright, int bcFaceZleft, 
                        const VirtualTopology3D * vct, EMfields3D *EMf);

void communicateCenterBC(int nx, int ny, int nz, double*** vector, 
                        int bcFaceXright, int bcFaceXleft, int bcFaceYright, int bcFaceYleft, int bcFaceZright, int bcFaceZleft, 
                        const VirtualTopology3D * vct, EMfields3D *EMf);

//* Communication + BC
void communicateCenterBC_P( int nx, int ny, int nz, arr3_double vector, 
                            int bcFaceXright, int bcFaceXleft, int bcFaceYright, int bcFaceYleft, int bcFaceZright, int bcFaceZleft, 
                            const VirtualTopology3D * vct, EMfields3D *EMf);

//* Communicate ghost cells (cell centres) with box stencil
void communicateCenterBoxStencilBC( int nx, int ny, int nz, arr3_double vector, 
                                    int bcFaceXright, int bcFaceXleft, int bcFaceYright, int bcFaceYleft, int bcFaceZright, int bcFaceZleft, 
                                    const VirtualTopology3D * vct, EMfields3D *EMf);

//* Particles communicate ghost cells (cell centres) with BOX stencil*/
void communicateCenterBoxStencilBC_P(int nx, int ny, int nz, arr3_double vector, 
                                    int bcFaceXright, int bcFaceXleft, int bcFaceYright, int bcFaceYleft, int bcFaceZright, int bcFaceZleft, 
                                    const VirtualTopology3D * vct, EMfields3D *EMf);

//* This communication is used in interpolating from particle to grid
void communicateInterp(int nx, int ny, int nz, double*** vector, const VirtualTopology3D * vct, EMfields3D *EMf);
void communicateInterp(int nx, int ny, int nz, arr3_double _vector, const VirtualTopology3D * vct, EMfields3D *EMf);
void communicateNode_P(int nx, int ny, int nz, double*** vector, const VirtualTopology3D * vct, EMfields3D *EMf);
void communicateNode_P(int nx, int ny, int nz, arr3_double _vector, const VirtualTopology3D * vct, EMfields3D *EMf);

void communicateNode_P_old(int nx, int ny, int nz, int ns, double ****vector, const VirtualTopology3D *vct, EMfields3D *EMf);
// void communicateInterp(int nx, int ny, int nz, double ****vector, int ns, const VirtualTopology3D * vct);
void communicateInterp_old(int nx, int ny, int nz, int ns, double ****vector, 
                           int bcFaceXright, int bcFaceXleft, int bcFaceYright, int bcFaceYleft, int bcFaceZright, int bcFaceZleft, 
                           const VirtualTopology3D *vct, EMfields3D *EMf);

void addCorner(int nx, int ny, int nz, double ***vector, const VirtualTopology3D * vct);
void addEdgeX (int nx, int ny, int nz, double ***vector, const VirtualTopology3D * vct);
void addEdgeY (int nx, int ny, int nz, double ***vector, const VirtualTopology3D * vct);
void addEdgeZ (int nx, int ny, int nz, double ***vector, const VirtualTopology3D * vct);
void addFace  (int nx, int ny, int nz, double ***vector, const VirtualTopology3D * vct);

#endif