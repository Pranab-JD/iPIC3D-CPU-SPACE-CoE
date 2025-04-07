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

#include "Com3DNonblk.h"

//! isCenterFlag: 1 = communicateCenter; 0 = communicateNode
void NBDerivedHaloComm(int nx, int ny, int nz, double ***vector, const VirtualTopology3D * vct, EMfields3D *EMf,
                        bool isCenterFlag, bool isFaceOnlyFlag, bool needInterp, bool isParticle)
{
    const MPI_Comm comm       = isParticle ?vct->getParticleComm()      :vct->getFieldComm();
    #ifdef DEBUG
	    MPI_Errhandler_set(comm,MPI_ERRORS_RETURN);
    #endif
    MPI_Status  stat[12];
    MPI_Request reqList[12];				  //at most 6 requests x 2 (send recv)
    int communicationCnt[6] = {0,0,0,0,0,0};  //1 if there communication on that dir
    int recvcnt = 0,sendcnt = 0;              //request counter

    const int tag_XL=1,tag_YL=2,tag_ZL=3,tag_XR=4,tag_YR=5,tag_ZR=6;//To address same rank as left and right neighbour in periodic case
    const int myrank    	  = vct->getCartesian_rank();
    const int right_neighborX = isParticle ?vct->getXright_neighbor_P() :vct->getXright_neighbor();
    const int left_neighborX  = isParticle ?vct->getXleft_neighbor_P()  :vct->getXleft_neighbor();
    const int right_neighborY = isParticle ?vct->getYright_neighbor_P() :vct->getYright_neighbor();
    const int left_neighborY  = isParticle ?vct->getYleft_neighbor_P()  :vct->getYleft_neighbor();
    const int right_neighborZ = isParticle ?vct->getZright_neighbor_P() :vct->getZright_neighbor();
    const int left_neighborZ  = isParticle ?vct->getZleft_neighbor_P()  :vct->getZleft_neighbor();
    
    bool isCenterDim = (isCenterFlag&&!needInterp);//This is to address moment interp use nodes dimension but exchange center face
    
    const MPI_Datatype yzFacetype = EMf->getYZFacetype(isCenterDim);
    const MPI_Datatype xzFacetype = EMf->getXZFacetype(isCenterDim);
    const MPI_Datatype xyFacetype = EMf->getXYFacetype(isCenterDim);
    const MPI_Datatype xEdgetype = EMf->getXEdgetype(isCenterDim);
    const MPI_Datatype yEdgetype = EMf->getYEdgetype(isCenterDim);
    const MPI_Datatype zEdgetype = EMf->getZEdgetype(isCenterDim);
    const MPI_Datatype xEdgetype2 = EMf->getXEdgetype2(isCenterDim);
    const MPI_Datatype yEdgetype2 = EMf->getYEdgetype2(isCenterDim);
    const MPI_Datatype zEdgetype2 = EMf->getZEdgetype2(isCenterDim);
    const MPI_Datatype cornertype = EMf->getCornertype(isCenterDim);


    //Face Exchange as long as neighbor exits on that direction
    //Tag is based on the sender: XL for sender = XR for receiver
    //Post Recv before Send
    if(left_neighborX != MPI_PROC_NULL && left_neighborX != myrank)
    {
        MPI_Irecv(&vector[0][1][1], 1, yzFacetype, left_neighborX,tag_XR, comm, &reqList[recvcnt++]);
        communicationCnt[0] = 1;
    }
    if(right_neighborX != MPI_PROC_NULL && right_neighborX != myrank)
    {
        MPI_Irecv(&vector[nx-1][1][1], 1, yzFacetype, right_neighborX,tag_XL, comm, &reqList[recvcnt++]);
        communicationCnt[1] = 1;
    }
    if(left_neighborY != MPI_PROC_NULL && left_neighborY != myrank)
    {
        MPI_Irecv(&vector[1][0][1], 1, xzFacetype, left_neighborY,tag_YR, comm, &reqList[recvcnt++]);
        communicationCnt[2] = 1;
    }
    if(right_neighborY != MPI_PROC_NULL && right_neighborY != myrank)
    {
        MPI_Irecv(&vector[1][ny-1][1], 1, xzFacetype, right_neighborY,tag_YL, comm, &reqList[recvcnt++]);
        communicationCnt[3] = 1;
    }
    if(left_neighborZ != MPI_PROC_NULL && left_neighborZ != myrank)
    {
        MPI_Irecv(&vector[1][1][0],    1, xyFacetype, left_neighborZ,tag_ZR, comm, &reqList[recvcnt++]);
        communicationCnt[4] = 1;
    }
    if(right_neighborZ != MPI_PROC_NULL&& right_neighborZ != myrank)
    {
        MPI_Irecv(&vector[1][1][nz-1], 1, xyFacetype, right_neighborZ,tag_ZL, comm, &reqList[recvcnt++]);
        communicationCnt[5] = 1;
    }

    sendcnt = recvcnt;

    int offset = (isCenterFlag ?0:1);

    if(communicationCnt[0] == 1)
        MPI_Isend(&vector[1+offset][1][1],    1, yzFacetype, left_neighborX, tag_XL, comm, &reqList[sendcnt++]);
    if(communicationCnt[1] == 1)
        MPI_Isend(&vector[nx-2-offset][1][1], 1, yzFacetype, right_neighborX,tag_XR, comm, &reqList[sendcnt++]);
    if(communicationCnt[2] == 1)
        MPI_Isend(&vector[1][1+offset][1],    1, xzFacetype, left_neighborY, tag_YL, comm, &reqList[sendcnt++]);
    if(communicationCnt[3] == 1)
        MPI_Isend(&vector[1][ny-2-offset][1], 1, xzFacetype, right_neighborY,tag_YR, comm, &reqList[sendcnt++]);
    if(communicationCnt[4] == 1)
        MPI_Isend(&vector[1][1][1+offset],    1, xyFacetype, left_neighborZ, tag_ZL, comm, &reqList[sendcnt++]);
    if(communicationCnt[5] == 1)
        MPI_Isend(&vector[1][1][nz-2-offset], 1, xyFacetype, right_neighborZ,tag_ZR, comm, &reqList[sendcnt++]);

    assert_eq(recvcnt,sendcnt-recvcnt);

    //Buffer swap if any (done before waiting for receiving msg done to delay sync)
    if (right_neighborX == myrank &&  left_neighborX== myrank)
    {
        for (int iy = 1; iy < ny-1; iy++)
            for (int iz = 1; iz < nz-1; iz++) 
            {
                vector[0][iy][iz] = vector[nx-2][iy][iz];
                vector[nx-1][iy][iz] = vector[1][iy][iz];
            }
    }
    if (right_neighborY == myrank &&  left_neighborY == myrank)
    {
        for (int ix = 1; ix < nx-1; ix++)
            for (int iz = 1; iz < nz-1; iz++) 
            {
                vector[ix][0][iz] = vector[ix][ny-2][iz];
                vector[ix][ny-1][iz] = vector[ix][1][iz];
            }
    }
    if (right_neighborZ == myrank &&  left_neighborZ == myrank)
    {
        for (int ix = 1; ix < nx-1; ix++)
            for (int iy = 1; iy < ny-1; iy++) 
            {
                vector[ix][iy][0]    = vector[ix][iy][nz-2];
                vector[ix][iy][nz-1] = vector[ix][iy][1];
            }
    }

    //* Need to finish receiving + sending before edge exchange
    if(sendcnt>0)
    {
        MPI_Waitall(sendcnt,  &reqList[0], &stat[0]);
        bool stopFlag = false;
        for(int si=0;si< sendcnt;si++){
            int error_code = stat[si].MPI_ERROR;
            if (error_code != MPI_SUCCESS) 
            {
                stopFlag = true;
                
                #ifdef DEBUG
                    char error_string[100];
                    int length_of_error_string, error_class;

                    MPI_Error_class(error_code, &error_class);
                    MPI_Error_string(error_class, error_string, &length_of_error_string);
                    dprintf("MPI_Waitall error at %d  %s\n",si, error_string);
                #endif
            }
        }
        
        if(stopFlag) exit (EXIT_FAILURE);
    }


	if( !isFaceOnlyFlag ){

		//Exchange yEdge only when Z X neighbours exist
		//if Zleft + Zright, use merged yEdgeType2 to send two edges in one msg
		//Otherwise, only send one yEdge
		recvcnt = 0,sendcnt = 0;
		if(communicationCnt[0] == 1){
			if(communicationCnt[4] == 1 && communicationCnt[5] == 1){
				MPI_Irecv(&vector[0][1][0],   1,  yEdgetype2, left_neighborX, tag_XR, comm, &reqList[recvcnt++]);
			}else if(communicationCnt[4] == 1){
				MPI_Irecv(&vector[0][1][0],   1,  yEdgetype, left_neighborX, tag_XR, comm, &reqList[recvcnt++]);
			}else if(communicationCnt[5] == 1){
				MPI_Irecv(&vector[0][1][nz-1],1,  yEdgetype, left_neighborX, tag_XR, comm, &reqList[recvcnt++]);
			}
		}
		if(communicationCnt[1] == 1){
			if(communicationCnt[4] == 1 && communicationCnt[5] == 1){
				MPI_Irecv(&vector[nx-1][1][0],  1,  yEdgetype2,right_neighborX,tag_XL, comm, &reqList[recvcnt++]);
			}else if(communicationCnt[4] == 1){
				MPI_Irecv(&vector[nx-1][1][0],   1, yEdgetype, right_neighborX, tag_XL, comm, &reqList[recvcnt++]);
			}else if(communicationCnt[5] == 1){
				MPI_Irecv(&vector[nx-1][1][nz-1],1, yEdgetype, right_neighborX, tag_XL, comm, &reqList[recvcnt++]);
			}
		}
		//Exchange zEdge only when X Y neighbours exist
		//if Xleft + Xright, use merged zEdgeType2
		//Otherwise, only send one zEdge
		if(communicationCnt[2] == 1){
			if(communicationCnt[0] == 1 && communicationCnt[1] == 1){
				MPI_Irecv(&vector[0][0][1],   1, zEdgetype2,left_neighborY, tag_YR,comm, &reqList[recvcnt++]);
			}else if(communicationCnt[0] == 1){
				MPI_Irecv(&vector[0][0][1],   1, zEdgetype, left_neighborY, tag_YR,comm, &reqList[recvcnt++]);
			}else if(communicationCnt[1] == 1){
				MPI_Irecv(&vector[nx-1][0][1],1, zEdgetype, left_neighborY, tag_YR,comm, &reqList[recvcnt++]);
			}
		}
		if(communicationCnt[3] == 1){
			if(communicationCnt[0] == 1 && communicationCnt[1] == 1){
				MPI_Irecv(&vector[0][ny-1][1],	 1, zEdgetype2,right_neighborY,tag_YL,comm, &reqList[recvcnt++]);
			}else if(communicationCnt[0] == 1){
				MPI_Irecv(&vector[0][ny-1][1],   1, zEdgetype, right_neighborY,tag_YL,comm, &reqList[recvcnt++]);
			}else if(communicationCnt[1] == 1){
				MPI_Irecv(&vector[nx-1][ny-1][1],1, zEdgetype, right_neighborY,tag_YL,comm, &reqList[recvcnt++]);
			}
		}
		//Exchange xEdge only when Y Z neighbours exist
		//if Yleft + Yright exist, use merged xEdgeType2
		//Otherwise, only send one xEdge
		if(communicationCnt[4] == 1){
			if(communicationCnt[2] == 1 && communicationCnt[3] == 1){
				MPI_Irecv(&vector[1][0][0],1,    xEdgetype2,left_neighborZ, tag_ZR,   comm, &reqList[recvcnt++]);
			}else if(communicationCnt[2] == 1){
				MPI_Irecv(&vector[1][0][0],1,    xEdgetype, left_neighborZ, tag_ZR,comm, &reqList[recvcnt++]);
			}else if(communicationCnt[3] == 1){
				MPI_Irecv(&vector[1][ny-1][0],1, xEdgetype, left_neighborZ,tag_ZR,comm, &reqList[recvcnt++]);
			}
		}
		if(communicationCnt[5] == 1){
			if(communicationCnt[2] == 1 && communicationCnt[3] == 1){
				MPI_Irecv(&vector[1][0][nz-1],1,  xEdgetype2,right_neighborZ, tag_ZL,comm, &reqList[recvcnt++]);
			}else if(communicationCnt[2] == 1){
				MPI_Irecv(&vector[1][0][nz-1],1,   xEdgetype,right_neighborZ, tag_ZL,comm, &reqList[recvcnt++]);
			}else if(communicationCnt[3] == 1){
				MPI_Irecv(&vector[1][ny-1][nz-1],1,xEdgetype,right_neighborZ, tag_ZL,comm, &reqList[recvcnt++]);
			}
		}

		sendcnt = recvcnt;

		if(communicationCnt[0] == 1){
			if(communicationCnt[4] == 1 && communicationCnt[5] == 1){
				MPI_Isend(&vector[1][1][0],   1,  yEdgetype2,left_neighborX, tag_XL, comm, &reqList[sendcnt++]);
			}else if(communicationCnt[4] == 1){
				MPI_Isend(&vector[1][1][0],   1,  yEdgetype, left_neighborX, tag_XL, comm, &reqList[sendcnt++]);
			}else if(communicationCnt[5] == 1){
				MPI_Isend(&vector[1][1][nz-1],1,  yEdgetype, left_neighborX, tag_XL, comm, &reqList[sendcnt++]);
			}
		}
		if(communicationCnt[1] == 1){
			if(communicationCnt[4] == 1 && communicationCnt[5] == 1){
				MPI_Isend(&vector[nx-2][1][0],1,   yEdgetype2,right_neighborX, tag_XR, comm,&reqList[sendcnt++]);
			}else if(communicationCnt[4] == 1){
				MPI_Isend(&vector[nx-2][1][0],1,   yEdgetype,right_neighborX, tag_XR, comm, &reqList[sendcnt++]);
			}else if(communicationCnt[5] == 1){
				MPI_Isend(&vector[nx-2][1][nz-1],1,yEdgetype,right_neighborX, tag_XR,comm,&reqList[sendcnt++]);
			}
		}
		if(communicationCnt[2] == 1){
			if(communicationCnt[0] == 1 && communicationCnt[1] == 1){
				MPI_Isend(&vector[0][1][1],1,   zEdgetype2, left_neighborY, tag_YL,   comm, &reqList[sendcnt++]);
			}else if(communicationCnt[0] == 1){
				MPI_Isend(&vector[0][1][1],1,   zEdgetype,  left_neighborY, tag_YL,   comm, &reqList[sendcnt++]);
			}else if(communicationCnt[1] == 1){
				MPI_Isend(&vector[nx-1][1][1],1,zEdgetype,left_neighborY, tag_YL,   comm, &reqList[sendcnt++]);
			}
		}
		if(communicationCnt[3] == 1){
			if(communicationCnt[0] == 1 && communicationCnt[1] == 1){
				MPI_Isend(&vector[0][ny-2][1],1,   zEdgetype2, right_neighborY, tag_YR,   comm, &reqList[sendcnt++]);
			}else if(communicationCnt[0] == 1){
				MPI_Isend(&vector[0][ny-2][1],1,   zEdgetype,  right_neighborY, tag_YR,   comm, &reqList[sendcnt++]);
			}else if(communicationCnt[1] == 1){
				MPI_Isend(&vector[nx-1][ny-2][1],1,zEdgetype,right_neighborY, tag_YR,   comm, &reqList[sendcnt++]);
			}
		}
		if(communicationCnt[4] == 1){
			if(communicationCnt[2] == 1 && communicationCnt[3] == 1){
				MPI_Isend(&vector[1][0][1],1, xEdgetype2, left_neighborZ, tag_ZL,   comm, &reqList[sendcnt++]);
			}else if(communicationCnt[2] == 1){
				MPI_Isend(&vector[1][0][1],1, xEdgetype,  left_neighborZ, tag_ZL,   comm, &reqList[sendcnt++]);
			}else if(communicationCnt[3] == 1){
				MPI_Isend(&vector[1][ny-1][1],1,xEdgetype,left_neighborZ, tag_ZL,   comm, &reqList[sendcnt++]);
			}
		}
		if(communicationCnt[5] == 1){
			if(communicationCnt[2] == 1 && communicationCnt[3] == 1){
				MPI_Isend(&vector[1][0][nz-2],1,    xEdgetype2, right_neighborZ, tag_ZR, comm, &reqList[sendcnt++]);
			}else if(communicationCnt[2] == 1){
				MPI_Isend(&vector[1][0][nz-2],1,    xEdgetype,  right_neighborZ, tag_ZR, comm, &reqList[sendcnt++]);
			}else if(communicationCnt[3] == 1){
				MPI_Isend(&vector[1][ny-1][nz-2],1, xEdgetype,  right_neighborZ, tag_ZR, comm, &reqList[sendcnt++]);
			}
		}

		assert_eq(recvcnt,sendcnt-recvcnt);

		//Swap Local Edges
		if(right_neighborX == myrank &&  left_neighborX== myrank){
           if(right_neighborZ != MPI_PROC_NULL ){
                for (int iy = 1; iy < ny-1; iy++){
                    vector[0][iy][nz-1]    = vector[nx-2][iy][nz-1];
                    vector[nx-1][iy][nz-1] = vector[1][iy][nz-1];
                }
           }
           if(left_neighborZ != MPI_PROC_NULL ){
                for (int iy = 1; iy < ny-1; iy++){
                    vector[0][iy][0]    = vector[nx-2][iy][0];
                    vector[nx-1][iy][0] = vector[1][iy][0];
                }
           }
           if(right_neighborY != MPI_PROC_NULL ){
                for (int iz = 1; iz < nz-1; iz++) {
                    vector[0][ny-1][iz]    = vector[nx-2][ny-1][iz];
                    vector[nx-1][ny-1][iz] = vector[1][ny-1][iz];
                }
           }
           if(left_neighborY != MPI_PROC_NULL ){
                for (int iz = 1; iz < nz-1; iz++) {
                    vector[0][0][iz]       = vector[nx-2][0][iz];
                    vector[nx-1][0][iz]    = vector[1][0][iz];
                }
           }
        }
		if(right_neighborY == myrank &&  left_neighborY == myrank){
		   if(right_neighborX != MPI_PROC_NULL ){
                for (int iz = 1; iz < nz-1; iz++) {
                    vector[nx-1][0][iz]    = vector[nx-1][ny-2][iz];
                    vector[nx-1][ny-1][iz] = vector[nx-1][1][iz];
			}
		  }
          if(left_neighborX != MPI_PROC_NULL){
                for (int iz = 1; iz < nz-1; iz++) {
                    vector[0][0][iz]       = vector[0][ny-2][iz];
                    vector[0][ny-1][iz]    = vector[0][1][iz];
                }
		  }
		  if(right_neighborZ != MPI_PROC_NULL){
                for (int ix = 1; ix < nx-1; ix++){
                    vector[ix][0][nz-1]    = vector[ix][ny-2][nz-1];
                    vector[ix][ny-1][nz-1] = vector[ix][1][nz-1];
                }
		  }
		  if(left_neighborZ != MPI_PROC_NULL){
                for (int ix = 1; ix < nx-1; ix++){
                    vector[ix][0][0]       = vector[ix][ny-2][0];
                    vector[ix][ny-1][0]    = vector[ix][1][0];
                }
		  }
		}

		if(right_neighborZ == myrank &&  left_neighborZ == myrank){
            if(right_neighborY != MPI_PROC_NULL ){
                for (int ix = 1; ix < nx-1; ix++){
                    vector[ix][ny-1][0]    = vector[ix][ny-1][nz-2];
                    vector[ix][ny-1][nz-1] = vector[ix][ny-1][1];
                }
            }
            if(left_neighborY != MPI_PROC_NULL ){
                for (int ix = 1; ix < nx-1; ix++){
                    vector[ix][0][0]    = vector[ix][0][nz-2];
                    vector[ix][0][nz-1] = vector[ix][0][1];
                }
            }
            if(right_neighborX != MPI_PROC_NULL ){
                for (int iy = 1; iy < ny-1; iy++){
                    vector[nx-1][iy][0]    = vector[nx-1][iy][nz-2];
                    vector[nx-1][iy][nz-1] = vector[nx-1][iy][1];
                }
            }
            if(left_neighborX != MPI_PROC_NULL ){
                for (int iy = 1; iy < ny-1; iy++){
                    vector[0][iy][0]    = vector[0][iy][nz-2];
                    vector[0][iy][nz-1] = vector[0][iy][1];
                }
            }
        }
		//Need to finish receiving edges for corner exchange
		if(sendcnt>0){
			MPI_Waitall(sendcnt,&reqList[0],&stat[0]);
			bool stopFlag = false;
			for(int si=0;si< sendcnt;si++){
				int error_code = stat[si].MPI_ERROR;
				if (error_code != MPI_SUCCESS) {
					stopFlag = true;
#ifdef DEBUG
					char error_string[100];
					int length_of_error_string, error_class;

					MPI_Error_class(error_code, &error_class);
					MPI_Error_string(error_class, error_string, &length_of_error_string);
					dprintf("MPI_Waitall error at %d  %s\n",si, error_string);
#endif
				}
			}
			if(stopFlag) exit (EXIT_FAILURE);;
		}


		//Corner Exchange only needed if XYZ neighbours all exist
		//4 corners communicated in one message
		//Assume Non-periodic will be handled in BC
		//Define corner types for X communication
		recvcnt = 0,sendcnt = 0;
		if((communicationCnt[2] == 1 || communicationCnt[3] == 1) && (communicationCnt[4] == 1 || communicationCnt[5] == 1)){
			//if XLeft exists, send 4 corners to XLeft
			if(communicationCnt[0] == 1){
				MPI_Irecv(&vector[0][0][0],1,   cornertype, left_neighborX, tag_XR,comm, &reqList[recvcnt++]);
			}
			//if XRight exist
			if(communicationCnt[1] == 1){
				MPI_Irecv(&vector[nx-1][0][0],1,cornertype, right_neighborX, tag_XL,comm, &reqList[recvcnt++]);
			}

			sendcnt=recvcnt;

			if(communicationCnt[0] == 1){
				MPI_Isend(&vector[1][0][0],   1,cornertype,left_neighborX, tag_XL, comm, &reqList[sendcnt++]);
			}
			if(communicationCnt[1] == 1){
				MPI_Isend(&vector[nx-2][0][0],1,cornertype,right_neighborX, tag_XR,comm, &reqList[sendcnt++]);
			}
		}

		assert_eq(recvcnt,sendcnt-recvcnt);


		//Delay local data copy
		if (left_neighborX== myrank && right_neighborX == myrank){
			if( (left_neighborY != MPI_PROC_NULL) && (left_neighborZ != MPI_PROC_NULL)){
				vector[0][0][0]       = vector[nx-2][0][0];
				vector[nx-1][0][0]    = vector[1][0][0];
			}
			if( (left_neighborY != MPI_PROC_NULL) && (right_neighborZ != MPI_PROC_NULL)){
				vector[0][0][nz-1]    = vector[nx-2][0][nz-1];
				vector[nx-1][0][nz-1] = vector[1][0][nz-1];
			}
			if( (right_neighborY != MPI_PROC_NULL) && (left_neighborZ != MPI_PROC_NULL)){
				vector[0][ny-1][0]    = vector[nx-2][ny-1][0];
				vector[nx-1][ny-1][0] = vector[1][ny-1][0];
			}
			if( (right_neighborY != MPI_PROC_NULL) && (right_neighborZ != MPI_PROC_NULL)){
				vector[0][ny-1][nz-1]    = vector[nx-2][ny-1][nz-1];
				vector[nx-1][ny-1][nz-1] = vector[1][ny-1][nz-1];
			}
		}
		else if (left_neighborY== myrank && right_neighborY == myrank){
			if( (left_neighborX != MPI_PROC_NULL) && (left_neighborZ != MPI_PROC_NULL)){
				vector[0][0][0]       = vector[0][ny-2][0];
				vector[0][ny-1][0]    = vector[0][1][0];
			}
			if( (left_neighborX != MPI_PROC_NULL) && (right_neighborZ != MPI_PROC_NULL)){
				vector[0][0][nz-1]    = vector[0][ny-2][nz-1];
				vector[0][ny-1][nz-1]    = vector[0][1][nz-1];
			}
			if( (right_neighborX != MPI_PROC_NULL) && (left_neighborZ != MPI_PROC_NULL)){
				vector[nx-1][0][0]    = vector[nx-1][ny-2][0];
				vector[nx-1][ny-1][0] = vector[nx-1][1][0];
			}
			if( (right_neighborX != MPI_PROC_NULL) && (right_neighborZ != MPI_PROC_NULL)){
				vector[nx-1][0][nz-1]    = vector[nx-1][ny-2][nz-1];
				vector[nx-1][ny-1][nz-1] = vector[nx-1][1][nz-1];
			}
		}
		else if (left_neighborZ== myrank && right_neighborZ == myrank){
			if( (left_neighborY != MPI_PROC_NULL) && (left_neighborX != MPI_PROC_NULL)){
				vector[0][0][0]       = vector[0][0][nz-2];
				vector[0][0][nz-1]    = vector[0][0][1];
			}

			if( (left_neighborY != MPI_PROC_NULL) && (right_neighborX != MPI_PROC_NULL)){
				vector[nx-1][0][0]    = vector[nx-1][0][nz-2];
				vector[nx-1][0][nz-1] = vector[nx-1][0][1];
			}

			if( (right_neighborY != MPI_PROC_NULL) && (left_neighborX != MPI_PROC_NULL)){
				vector[0][ny-1][0]    = vector[0][ny-1][nz-2];
				vector[0][ny-1][nz-1] = vector[0][ny-1][1];
			}

			if( (right_neighborY != MPI_PROC_NULL) && (right_neighborX != MPI_PROC_NULL)){
				vector[nx-1][ny-1][0]    = vector[nx-1][ny-1][nz-2];
				vector[nx-1][ny-1][nz-1] = vector[nx-1][ny-1][1];
			}
		}


		if(sendcnt>0){
			MPI_Waitall(sendcnt,  &reqList[0], &stat[0]);
			bool stopFlag = false;
			for(int si=0;si< sendcnt;si++){
				int error_code = stat[si].MPI_ERROR;
				if (error_code != MPI_SUCCESS) {
					stopFlag = true;
#ifdef DEBUG
					char error_string[100];
					int length_of_error_string, error_class;

					MPI_Error_class(error_code, &error_class);
					MPI_Error_string(error_class, error_string, &length_of_error_string);
					dprintf("MPI_Waitall error at %d  %s\n",si, error_string);
#endif
				}
			}
			if(stopFlag) exit (EXIT_FAILURE);
		}
	}

	//if this is NodeInterpolation operation
	if(needInterp){
		addFace(nx, ny, nz, vector,  vct);

		addEdgeZ(nx, ny, nz, vector, vct);
		addEdgeY(nx, ny, nz, vector, vct);
		addEdgeX(nx, ny, nz, vector, vct);

		addCorner(nx, ny, nz, vector, vct);
	}

}


void communicateNodeBC(int nx, int ny, int nz, arr3_double _vector,
                        int bcFaceXrght, int bcFaceXleft,
                        int bcFaceYrght, int bcFaceYleft,
                        int bcFaceZrght, int bcFaceZleft,
                        const VirtualTopology3D * vct, EMfields3D *EMf)
{
	double ***vector=_vector.fetch_arr3();
	NBDerivedHaloComm(nx, ny, nz, vector, vct, EMf, false,false,false,false);
	BCface(nx, ny, nz, vector, bcFaceXrght, bcFaceXleft, bcFaceYrght, bcFaceYleft, bcFaceZrght, bcFaceZleft, vct);
}

void communicateNodeBC_old( int nx, int ny, int nz, double ***vector, 
                            int bcFaceXright, int bcFaceXleft, int bcFaceYright, int bcFaceYleft, int bcFaceZright, int bcFaceZleft, 
                            const VirtualTopology3D *vct, EMfields3D *EMf)
{
    // allocate 6 ghost cell Faces
    double *ghostXrightFace = new double[(ny - 2) * (nz - 2)];
    double *ghostXleftFace = new double[(ny - 2) * (nz - 2)];
    double *ghostYrightFace = new double[(nx - 2) * (nz - 2)];
    double *ghostYleftFace = new double[(nx - 2) * (nz - 2)];
    double *ghostZrightFace = new double[(nx - 2) * (ny - 2)];
    double *ghostZleftFace = new double[(nx - 2) * (ny - 2)];
    // allocate 12 ghost cell Edges
    // X EDGE
    double *ghostXsameYleftZleftEdge = new double[nx - 2];
    double *ghostXsameYrightZleftEdge = new double[nx - 2];
    double *ghostXsameYleftZrightEdge = new double[nx - 2];
    double *ghostXsameYrightZrightEdge = new double[nx - 2];
    // Y EDGE
    double *ghostXrightYsameZleftEdge = new double[ny - 2];
    double *ghostXleftYsameZleftEdge = new double[ny - 2];
    double *ghostXrightYsameZrightEdge = new double[ny - 2];
    double *ghostXleftYsameZrightEdge = new double[ny - 2];
    // Z EDGE
    double *ghostXrightYleftZsameEdge = new double[nz - 2];
    double *ghostXrightYrightZsameEdge = new double[nz - 2];
    double *ghostXleftYleftZsameEdge = new double[nz - 2];
    double *ghostXleftYrightZsameEdge = new double[nz - 2];
    // allocate 8 ghost cell corner
    double ghostXrightYrightZrightCorner, ghostXleftYrightZrightCorner, ghostXrightYleftZrightCorner, ghostXleftYleftZrightCorner;
    double ghostXrightYrightZleftCorner, ghostXleftYrightZleftCorner, ghostXrightYleftZleftCorner, ghostXleftYleftZleftCorner;

    // apply boundary condition to 6 Ghost Faces and communicate if necessary to 6 processors: along 3 DIRECTIONS
    makeNodeFace(nx, ny, nz, vector, ghostXrightFace, ghostXleftFace, ghostYrightFace, ghostYleftFace, ghostZrightFace, ghostZleftFace);
    communicateGhostFace((ny - 2) * (nz - 2), vct->getCartesian_rank(), vct->getXright_neighbor(), vct->getXleft_neighbor(), 0, vct->getXLEN(), vct->getYLEN(), vct->getZLEN(), ghostXrightFace, ghostXleftFace);
    communicateGhostFace((nx - 2) * (nz - 2), vct->getCartesian_rank(), vct->getYright_neighbor(), vct->getYleft_neighbor(), 1, vct->getXLEN(), vct->getYLEN(), vct->getZLEN(), ghostYrightFace, ghostYleftFace);
    communicateGhostFace((nx - 2) * (ny - 2), vct->getCartesian_rank(), vct->getZright_neighbor(), vct->getZleft_neighbor(), 2, vct->getXLEN(), vct->getYLEN(), vct->getZLEN(), ghostZrightFace, ghostZleftFace);
    parseFace(nx, ny, nz, vector, ghostXrightFace, ghostXleftFace, ghostYrightFace, ghostYleftFace, ghostZrightFace, ghostZleftFace);

    // prepare ghost cell Edge Y for communication: these are communicate: these are communicated in X direction */
    makeNodeGhostEdgeY(nx, ny, nz, ghostZleftFace, ghostZrightFace, ghostXrightYsameZrightEdge, ghostXleftYsameZleftEdge, ghostXleftYsameZrightEdge, ghostXrightYsameZleftEdge);
    // prepare ghost cell Edge Z for communication: these are communicated in Y direction */
    makeNodeGhostEdgeZ(nx, ny, nz, ghostXleftFace, ghostXrightFace, ghostXrightYrightZsameEdge, ghostXleftYleftZsameEdge, ghostXrightYleftZsameEdge, ghostXleftYrightZsameEdge);
    // prepare ghost cell Edge X for communication: these are communicated in Z direction*/
    makeNodeGhostEdgeX(nx, ny, nz, ghostYleftFace, ghostYrightFace, ghostXsameYrightZrightEdge, ghostXsameYleftZleftEdge, ghostXsameYleftZrightEdge, ghostXsameYrightZleftEdge);

    // communicate twice each direction
    // X-DIRECTION: Z -> X
    MPI_Barrier(MPI_COMM_WORLD);
    communicateGhostFace(ny - 2, vct->getCartesian_rank(), vct->getXright_neighbor(), vct->getXleft_neighbor(), 0, vct->getXLEN(), vct->getYLEN(), vct->getZLEN(), ghostXrightYsameZleftEdge, ghostXleftYsameZleftEdge);
    communicateGhostFace(ny - 2, vct->getCartesian_rank(), vct->getXright_neighbor(), vct->getXleft_neighbor(), 0, vct->getXLEN(), vct->getYLEN(), vct->getZLEN(), ghostXrightYsameZrightEdge, ghostXleftYsameZrightEdge);
    // Y-DIRECTION: X -> Y
    MPI_Barrier(MPI_COMM_WORLD);
    communicateGhostFace(nz - 2, vct->getCartesian_rank(), vct->getYright_neighbor(), vct->getYleft_neighbor(), 1, vct->getXLEN(), vct->getYLEN(), vct->getZLEN(), ghostXleftYrightZsameEdge, ghostXleftYleftZsameEdge);
    communicateGhostFace(nz - 2, vct->getCartesian_rank(), vct->getYright_neighbor(), vct->getYleft_neighbor(), 1, vct->getXLEN(), vct->getYLEN(), vct->getZLEN(), ghostXrightYrightZsameEdge, ghostXrightYleftZsameEdge);
    // Z-DIRECTION: Y -> Z
    MPI_Barrier(MPI_COMM_WORLD);
    communicateGhostFace(nx - 2, vct->getCartesian_rank(), vct->getZright_neighbor(), vct->getZleft_neighbor(), 2, vct->getXLEN(), vct->getYLEN(), vct->getZLEN(), ghostXsameYleftZrightEdge, ghostXsameYleftZleftEdge);
    communicateGhostFace(nx - 2, vct->getCartesian_rank(), vct->getZright_neighbor(), vct->getZleft_neighbor(), 2, vct->getXLEN(), vct->getYLEN(), vct->getZLEN(), ghostXsameYrightZrightEdge, ghostXsameYrightZleftEdge);
    // parse
    MPI_Barrier(MPI_COMM_WORLD);

    parseEdgeZ(nx, ny, nz, vector, ghostXrightYrightZsameEdge, ghostXleftYleftZsameEdge, ghostXrightYleftZsameEdge, ghostXleftYrightZsameEdge);
    parseEdgeY(nx, ny, nz, vector, ghostXrightYsameZrightEdge, ghostXleftYsameZleftEdge, ghostXleftYsameZrightEdge, ghostXrightYsameZleftEdge);
    parseEdgeX(nx, ny, nz, vector, ghostXsameYrightZrightEdge, ghostXsameYleftZleftEdge, ghostXsameYleftZrightEdge, ghostXsameYrightZleftEdge);

    // apply boundary condition to 8 Ghost Corners and communicate if necessary to 8 processors
    makeNodeGhostCorner(nx, ny, nz, ghostXsameYrightZrightEdge, ghostXsameYleftZleftEdge, ghostXsameYleftZrightEdge, ghostXsameYrightZleftEdge, &ghostXrightYrightZrightCorner, &ghostXleftYrightZrightCorner, &ghostXrightYleftZrightCorner, &ghostXleftYleftZrightCorner, &ghostXrightYrightZleftCorner, &ghostXleftYrightZleftCorner, &ghostXrightYleftZleftCorner, &ghostXleftYleftZleftCorner);
    // communicate only in the X-DIRECTION
    communicateGhostFace(1, vct->getCartesian_rank(), vct->getXright_neighbor(), vct->getXleft_neighbor(), 0, vct->getXLEN(), vct->getYLEN(), vct->getZLEN(), &ghostXrightYrightZrightCorner, &ghostXleftYrightZrightCorner);
    communicateGhostFace(1, vct->getCartesian_rank(), vct->getXright_neighbor(), vct->getXleft_neighbor(), 0, vct->getXLEN(), vct->getYLEN(), vct->getZLEN(), &ghostXrightYleftZrightCorner, &ghostXleftYleftZrightCorner);
    communicateGhostFace(1, vct->getCartesian_rank(), vct->getXright_neighbor(), vct->getXleft_neighbor(), 0, vct->getXLEN(), vct->getYLEN(), vct->getZLEN(), &ghostXrightYleftZleftCorner, &ghostXleftYleftZleftCorner);
    communicateGhostFace(1, vct->getCartesian_rank(), vct->getXright_neighbor(), vct->getXleft_neighbor(), 0, vct->getXLEN(), vct->getYLEN(), vct->getZLEN(), &ghostXrightYrightZleftCorner, &ghostXleftYrightZleftCorner);
    // parse
    parseCorner(nx, ny, nz, vector, &ghostXrightYrightZrightCorner, &ghostXleftYrightZrightCorner, &ghostXrightYleftZrightCorner, &ghostXleftYleftZrightCorner, &ghostXrightYrightZleftCorner, &ghostXleftYrightZleftCorner, &ghostXrightYleftZleftCorner, &ghostXleftYleftZleftCorner);

    // APPLY the boundary conditions
    BCface(nx, ny, nz, vector, bcFaceXright, bcFaceXleft, bcFaceYright, bcFaceYleft, bcFaceZright, bcFaceZleft, vct);

    delete[]ghostXrightFace;
    delete[]ghostXleftFace;
    delete[]ghostYrightFace;
    delete[]ghostYleftFace;
    delete[]ghostZrightFace;
    delete[]ghostZleftFace;
    // X EDGE
    delete[]ghostXsameYleftZleftEdge;
    delete[]ghostXsameYrightZleftEdge;
    delete[]ghostXsameYleftZrightEdge;
    delete[]ghostXsameYrightZrightEdge;
    // Y EDGE
    delete[]ghostXrightYsameZleftEdge;
    delete[]ghostXleftYsameZleftEdge;
    delete[]ghostXrightYsameZrightEdge;
    delete[]ghostXleftYsameZrightEdge;
    // Z EDGE
    delete[]ghostXrightYleftZsameEdge;
    delete[]ghostXrightYrightZsameEdge;
    delete[]ghostXleftYleftZsameEdge;
    delete[]ghostXleftYrightZsameEdge;
}

void communicateNodeBoxStencilBC( int nx, int ny, int nz, arr3_double _vector,
								  int bcFaceXrght, int bcFaceXleft,
								  int bcFaceYrght, int bcFaceYleft,
								  int bcFaceZrght, int bcFaceZleft,
								  const VirtualTopology3D * vct, EMfields3D *EMf)
{
    double ***vector=_vector.fetch_arr3();
    NBDerivedHaloComm(nx, ny, nz, vector, vct, EMf, false,true,false,false);
    BCface(nx, ny, nz, vector, bcFaceXrght, bcFaceXleft, bcFaceYrght, bcFaceYleft, bcFaceZrght, bcFaceZleft, vct);
}

void communicateNodeBoxStencilBC_P(int nx, int ny, int nz, arr3_double _vector,
                                    int bcFaceXrght, int bcFaceXleft,
                                    int bcFaceYrght, int bcFaceYleft,
                                    int bcFaceZrght, int bcFaceZleft,
                                    const VirtualTopology3D * vct, EMfields3D *EMf)
{
    double ***vector=_vector.fetch_arr3();
    NBDerivedHaloComm(nx, ny, nz, vector, vct, EMf, false,true,false,true);
    BCface_P(nx, ny, nz, vector, bcFaceXrght, bcFaceXleft, bcFaceYrght, bcFaceYleft, bcFaceZrght, bcFaceZleft, vct);
}

void communicateNodeBC_P(int nx, int ny, int nz, arr3_double _vector,
                        int bcFaceXrght, int bcFaceXleft,
                        int bcFaceYrght, int bcFaceYleft,
                        int bcFaceZrght, int bcFaceZleft,
                        const VirtualTopology3D * vct, EMfields3D *EMf)
{
	double ***vector=_vector.fetch_arr3();
	NBDerivedHaloComm(nx, ny, nz, vector, vct, EMf, false,false,false,true);
	BCface_P(nx, ny, nz, vector, bcFaceXrght, bcFaceXleft, bcFaceYrght, bcFaceYleft, bcFaceZrght, bcFaceZleft, vct);
}


void communicateCenterBC(int nx, int ny, int nz, arr3_double _vector,
                        int bcFaceXrght, int bcFaceXleft,
                        int bcFaceYrght, int bcFaceYleft,
                        int bcFaceZrght, int bcFaceZleft,
                        const VirtualTopology3D * vct, EMfields3D *EMf)
{
	double ***vector=_vector.fetch_arr3();
	NBDerivedHaloComm(nx, ny, nz, vector, vct, EMf, true, false,false,false);
    BCface(nx, ny, nz, vector, bcFaceXrght, bcFaceXleft, bcFaceYrght, bcFaceYleft, bcFaceZrght, bcFaceZleft, vct);
}

void communicateCenterBC(int nx, int ny, int nz, double*** vector,
                        int bcFaceXrght, int bcFaceXleft,
                        int bcFaceYrght, int bcFaceYleft,
                        int bcFaceZrght, int bcFaceZleft,
                        const VirtualTopology3D * vct, EMfields3D *EMf)
{
	NBDerivedHaloComm(nx, ny, nz, vector, vct, EMf, true, false, false, false);
    BCface(nx, ny, nz, vector, bcFaceXrght, bcFaceXleft, bcFaceYrght, bcFaceYleft, bcFaceZrght, bcFaceZleft, vct);
}

void communicateCenterBC_P( int nx, int ny, int nz, arr3_double _vector,
                            int bcFaceXrght, int bcFaceXleft,
                            int bcFaceYrght, int bcFaceYleft,
                            int bcFaceZrght, int bcFaceZleft,
                            const VirtualTopology3D * vct, EMfields3D *EMf)
{
    double ***vector=_vector.fetch_arr3();
    NBDerivedHaloComm(nx, ny, nz, vector, vct, EMf, true, false,false,true);
    BCface_P(nx, ny, nz, vector, bcFaceXrght, bcFaceXleft, bcFaceYrght, bcFaceYleft, bcFaceZrght, bcFaceZleft, vct);
}

void communicateCenterBoxStencilBC( int nx, int ny, int nz, arr3_double _vector,
                                    int bcFaceXrght, int bcFaceXleft,
                                    int bcFaceYrght, int bcFaceYleft,
                                    int bcFaceZrght, int bcFaceZleft,
                                    const VirtualTopology3D * vct, EMfields3D *EMf)
{
    double ***vector=_vector.fetch_arr3();
    NBDerivedHaloComm(nx, ny, nz, vector, vct, EMf, true,true,false,false);
    BCface(nx, ny, nz, vector, bcFaceXrght, bcFaceXleft, bcFaceYrght, bcFaceYleft, bcFaceZrght, bcFaceZleft, vct);
}

void communicateCenterBoxStencilBC_P( int nx, int ny, int nz, arr3_double _vector,
									  int bcFaceXrght, int bcFaceXleft,
									  int bcFaceYrght, int bcFaceYleft,
									  int bcFaceZrght, int bcFaceZleft,
									  const VirtualTopology3D * vct, EMfields3D *EMf)
{
    double ***vector=_vector.fetch_arr3();
    NBDerivedHaloComm(nx, ny, nz, vector, vct, EMf, true,true,false,true);
    BCface_P(nx, ny, nz, vector, bcFaceXrght, bcFaceXleft, bcFaceYrght, bcFaceYleft, bcFaceZrght, bcFaceZleft, vct);
}


/** add the values of ghost cells faces to the 3D physical vector */
void addFace(int nx, int ny, int nz, double ***vector, const VirtualTopology3D * vct)
{
    const int nxr = nx-2;
    const int nyr = ny-2;
    const int nzr = nz-2;

    // Xright
    if (vct->hasXrghtNeighbor_P())
    {
        for (int j = 1; j <= nyr; j++)
            for (int k = 1; k <= nzr; k++)
                vector[nx - 2][j][k] += vector[nx - 1][j][k];
    }
    // XLEFT
    if (vct->hasXleftNeighbor_P())
    {
        for (int j = 1; j <= nyr; j++)
            for (int k = 1; k <= nzr; k++)
                vector[1][j][k] += vector[0][j][k];
    }

    // Yright
    if (vct->hasYrghtNeighbor_P())
    {
        for (int i = 1; i <= nxr; i++)
            for (int k = 1; k <= nzr; k++)
                vector[i][ny - 2][k] += vector[i][ny - 1][k];
    }
    // Yleft
    if (vct->hasYleftNeighbor_P())
    {
        for (int i = 1; i <= nxr; i++)
            for (int k = 1; k <= nzr; k++)
                vector[i][1][k] += vector[i][0][k];
    }
    // Zright
    if (vct->hasZrghtNeighbor_P())
    {
        for (int i = 1; i <= nxr; i++)
            for (int j = 1; j <= nyr; j++)
                vector[i][j][nz - 2] += vector[i][j][nz - 1];
    }
    // ZLEFT
    if (vct->hasZleftNeighbor_P())
    {
        for (int i = 1; i <= nxr; i++)
            for (int j = 1; j <= nyr; j++)
                vector[i][j][1] += vector[i][j][0];
    }
}

/** insert the ghost cells Edge Z in the 3D physical vector */
void addEdgeZ(int nx, int ny, int nz, double ***vector, const VirtualTopology3D * vct)
{
    if (vct->hasXrghtNeighbor_P() && vct->hasYrghtNeighbor_P()) 
    {
        for (int i = 1; i < (nz - 1); i++)
            vector[nx - 2][ny - 2][i] += vector[nx - 1][ny - 1][i];
    }
    if (vct->hasXleftNeighbor_P() && vct->hasYleftNeighbor_P()) 
    {
        for (int i = 1; i < (nz - 1); i++)
            vector[1][1][i] += vector[0][0][i];
    }
    if (vct->hasXrghtNeighbor_P() && vct->hasYleftNeighbor_P()) 
    {
        for (int i = 1; i < (nz - 1); i++)
            vector[nx - 2][1][i] += vector[nx - 1][0][i];
    }
    if (vct->hasXleftNeighbor_P() && vct->hasYrghtNeighbor_P()) 
    {
        for (int i = 1; i < (nz - 1); i++)
            vector[1][ny - 2][i] += vector[0][ny - 1][i];
    }
}
/** add the ghost cell values Edge Y to the 3D physical vector */
void addEdgeY(int nx, int ny, int nz, double ***vector, const VirtualTopology3D * vct)
{
    if (vct->hasXrghtNeighbor_P() && vct->hasZrghtNeighbor_P()) 
    {
        for (int i = 1; i < (ny - 1); i++)
            vector[nx - 2][i][nz - 2] += vector[nx - 1][i][nz - 1];
    }
    if (vct->hasXleftNeighbor_P() && vct->hasZleftNeighbor_P()) 
    {
        for (int i = 1; i < (ny - 1); i++)
            vector[1][i][1] += vector[0][i][0];
    }
    if (vct->hasXleftNeighbor_P() && vct->hasZrghtNeighbor_P()) 
    {
        for (int i = 1; i < (ny - 1); i++)
            vector[1][i][nz - 2] += vector[0][i][nz - 1];
    }
    if (vct->hasXrghtNeighbor_P() && vct->hasZleftNeighbor_P()) 
    {
        for (int i = 1; i < (ny - 1); i++)
            vector[nx - 2][i][1] += vector[nx - 1][i][0];
    }
}

/** add the ghost values Edge X to the 3D physical vector */
void addEdgeX(int nx, int ny, int nz, double ***vector, const VirtualTopology3D * vct)
{
    if (vct->hasYrghtNeighbor_P() && vct->hasZrghtNeighbor_P()) {
        for (int i = 1; i < (nx - 1); i++)
        vector[i][ny - 2][nz - 2] += vector[i][ny - 1][nz - 1];
    }
    if (vct->hasYleftNeighbor_P() && vct->hasZleftNeighbor_P()) {
        for (int i = 1; i < (nx - 1); i++)
        vector[i][1][1] += vector[i][0][0];
    }
    if (vct->hasYleftNeighbor_P() && vct->hasZrghtNeighbor_P()) {
        for (int i = 1; i < (nx - 1); i++)
        vector[i][1][nz - 2] += vector[i][0][nz - 1];
    }
    if (vct->hasYrghtNeighbor_P() && vct->hasZleftNeighbor_P()) {
        for (int i = 1; i < (nx - 1); i++)
        vector[i][ny - 2][1] += vector[i][ny - 1][0];
    }
}

/** add ghost cells values Corners in the 3D physical vector */
void addCorner(int nx, int ny, int nz, double ***vector,const VirtualTopology3D * vct)
{
    if (vct->hasXrghtNeighbor_P() && vct->hasYrghtNeighbor_P() && vct->hasZrghtNeighbor_P())
        vector[nx - 2][ny - 2][nz - 2] += vector[nx - 1][ny - 1][nz - 1];
    if (vct->hasXleftNeighbor_P() && vct->hasYrghtNeighbor_P() && vct->hasZrghtNeighbor_P())
        vector[1][ny - 2][nz - 2] += vector[0][ny - 1][nz - 1];
    if (vct->hasXrghtNeighbor_P() && vct->hasYleftNeighbor_P() && vct->hasZrghtNeighbor_P())
        vector[nx - 2][1][nz - 2] += vector[nx - 1][0][nz - 1];
    if (vct->hasXleftNeighbor_P() && vct->hasYleftNeighbor_P() && vct->hasZrghtNeighbor_P())
        vector[1][1][nz - 2] += vector[0][0][nz - 1];
    if (vct->hasXrghtNeighbor_P() && vct->hasYrghtNeighbor_P() && vct->hasZleftNeighbor_P())
        vector[nx - 2][ny - 2][1] += vector[nx - 1][ny - 1][0];
    if (vct->hasXleftNeighbor_P() && vct->hasYrghtNeighbor_P() && vct->hasZleftNeighbor_P())
        vector[1][ny - 2][1] += vector[0][ny - 1][0] ;
    if (vct->hasXrghtNeighbor_P() && vct->hasYleftNeighbor_P() && vct->hasZleftNeighbor_P())
        vector[nx - 2][1][1] += vector[nx - 1][0][0] ;
    if (vct->hasXleftNeighbor_P() && vct->hasYleftNeighbor_P() && vct->hasZleftNeighbor_P())
        vector[1][1][1] += vector[0][0][0];

}
/** communicate and sum shared ghost cells */

//? Used for communicating moments
void communicateInterp(int nx, int ny, int nz, double*** vector, const VirtualTopology3D * vct, EMfields3D *EMf)
{
	NBDerivedHaloComm(nx, ny, nz, vector, vct, EMf, true, false, true, true);
}

void communicateInterp(int nx, int ny, int nz, arr3_double _vector, const VirtualTopology3D * vct, EMfields3D *EMf)
{
    double ***vector=_vector.fetch_arr3();
	NBDerivedHaloComm(nx, ny, nz, vector, vct, EMf, true, false, true, true);
}

void communicateNode_P(int nx, int ny, int nz, double*** vector, const VirtualTopology3D * vct, EMfields3D *EMf)
{
	NBDerivedHaloComm(nx, ny, nz, vector, vct, EMf, false, false, false, true);
}

void communicateNode_P(int nx, int ny, int nz, arr3_double _vector, const VirtualTopology3D * vct, EMfields3D *EMf)
{
    double ***vector=_vector.fetch_arr3();
	NBDerivedHaloComm(nx, ny, nz, vector, vct, EMf, false, false, false, true);
}

//? Old functions used for communicating 4D vectors (each species of particles in moments)
void communicateNode_P_old(int nx, int ny, int nz, int ns, double ****vector, const VirtualTopology3D *vct, EMfields3D *EMf) 
{
    // allocate 6 ghost cell Faces
    double *ghostXrightFace = new double[(ny - 2) * (nz - 2)];
    double *ghostXleftFace = new double[(ny - 2) * (nz - 2)];
    double *ghostYrightFace = new double[(nx - 2) * (nz - 2)];
    double *ghostYleftFace = new double[(nx - 2) * (nz - 2)];
    double *ghostZrightFace = new double[(nx - 2) * (ny - 2)];
    double *ghostZleftFace = new double[(nx - 2) * (ny - 2)];
    // allocate 12 ghost cell Edges
    // X EDGE
    double *ghostXsameYleftZleftEdge = new double[nx - 2];
    double *ghostXsameYrightZleftEdge = new double[nx - 2];
    double *ghostXsameYleftZrightEdge = new double[nx - 2];
    double *ghostXsameYrightZrightEdge = new double[nx - 2];
    // Y EDGE
    double *ghostXrightYsameZleftEdge = new double[ny - 2];
    double *ghostXleftYsameZleftEdge = new double[ny - 2];
    double *ghostXrightYsameZrightEdge = new double[ny - 2];
    double *ghostXleftYsameZrightEdge = new double[ny - 2];
    // Z EDGE
    double *ghostXrightYleftZsameEdge = new double[nz - 2];
    double *ghostXrightYrightZsameEdge = new double[nz - 2];
    double *ghostXleftYleftZsameEdge = new double[nz - 2];
    double *ghostXleftYrightZsameEdge = new double[nz - 2];
    // allocate 8 ghost cell corner
    double ghostXrightYrightZrightCorner, ghostXleftYrightZrightCorner, ghostXrightYleftZrightCorner, ghostXleftYleftZrightCorner;
    double ghostXrightYrightZleftCorner, ghostXleftYrightZleftCorner, ghostXrightYleftZleftCorner, ghostXleftYleftZleftCorner;
  
    // apply boundary condition to 6 Ghost Faces and communicate if necessary to 6 processors: along 3 DIRECTIONS
    makeNodeFace(nx, ny, nz, vector, ns, ghostXrightFace, ghostXleftFace, ghostYrightFace, ghostYleftFace, ghostZrightFace, ghostZleftFace);
    communicateGhostFace((ny - 2) * (nz - 2), vct->getCartesian_rank(), vct->getXright_neighbor_P(), vct->getXleft_neighbor_P(), 0, vct->getXLEN(), vct->getYLEN(), vct->getZLEN(), ghostXrightFace, ghostXleftFace);
    communicateGhostFace((nx - 2) * (nz - 2), vct->getCartesian_rank(), vct->getYright_neighbor_P(), vct->getYleft_neighbor_P(), 1, vct->getXLEN(), vct->getYLEN(), vct->getZLEN(), ghostYrightFace, ghostYleftFace);
    communicateGhostFace((nx - 2) * (ny - 2), vct->getCartesian_rank(), vct->getZright_neighbor_P(), vct->getZleft_neighbor_P(), 2, vct->getXLEN(), vct->getYLEN(), vct->getZLEN(), ghostZrightFace, ghostZleftFace);
    parseFace(nx, ny, nz, vector, ns, ghostXrightFace, ghostXleftFace, ghostYrightFace, ghostYleftFace, ghostZrightFace, ghostZleftFace);
  
    /** prepare ghost cell Edge Y for communication: these are communicate: these are communicated in X direction */
    makeNodeGhostEdgeY(nx, ny, nz, ghostZleftFace, ghostZrightFace, ghostXrightYsameZrightEdge, ghostXleftYsameZleftEdge, ghostXleftYsameZrightEdge, ghostXrightYsameZleftEdge);
    /** prepare ghost cell Edge Z for communication: these are communicated in Y direction */
    makeNodeGhostEdgeZ(nx, ny, nz, ghostXleftFace, ghostXrightFace, ghostXrightYrightZsameEdge, ghostXleftYleftZsameEdge, ghostXrightYleftZsameEdge, ghostXleftYrightZsameEdge);
    /** prepare ghost cell Edge X for communication: these are communicated in  Z direction*/
    makeNodeGhostEdgeX(nx, ny, nz, ghostYleftFace, ghostYrightFace, ghostXsameYrightZrightEdge, ghostXsameYleftZleftEdge, ghostXsameYleftZrightEdge, ghostXsameYrightZleftEdge);
  
    // communicate twice each direction
    // X-DIRECTION: Z -> X
    MPI_Barrier(MPI_COMM_WORLD);
    communicateGhostFace(ny - 2, vct->getCartesian_rank(), vct->getXright_neighbor_P(), vct->getXleft_neighbor_P(), 0, vct->getXLEN(), vct->getYLEN(), vct->getZLEN(), ghostXrightYsameZleftEdge, ghostXleftYsameZleftEdge);
    communicateGhostFace(ny - 2, vct->getCartesian_rank(), vct->getXright_neighbor_P(), vct->getXleft_neighbor_P(), 0, vct->getXLEN(), vct->getYLEN(), vct->getZLEN(), ghostXrightYsameZrightEdge, ghostXleftYsameZrightEdge);
    // Y-DIRECTION: X -> Y
    MPI_Barrier(MPI_COMM_WORLD);
    communicateGhostFace(nz - 2, vct->getCartesian_rank(), vct->getYright_neighbor_P(), vct->getYleft_neighbor_P(), 1, vct->getXLEN(), vct->getYLEN(), vct->getZLEN(), ghostXleftYrightZsameEdge, ghostXleftYleftZsameEdge);
    communicateGhostFace(nz - 2, vct->getCartesian_rank(), vct->getYright_neighbor_P(), vct->getYleft_neighbor_P(), 1, vct->getXLEN(), vct->getYLEN(), vct->getZLEN(), ghostXrightYrightZsameEdge, ghostXrightYleftZsameEdge);
    // Z-DIRECTION: Y -> Z
    MPI_Barrier(MPI_COMM_WORLD);
    communicateGhostFace(nx - 2, vct->getCartesian_rank(), vct->getZright_neighbor_P(), vct->getZleft_neighbor_P(), 2, vct->getXLEN(), vct->getYLEN(), vct->getZLEN(), ghostXsameYleftZrightEdge, ghostXsameYleftZleftEdge);
    communicateGhostFace(nx - 2, vct->getCartesian_rank(), vct->getZright_neighbor_P(), vct->getZleft_neighbor_P(), 2, vct->getXLEN(), vct->getYLEN(), vct->getZLEN(), ghostXsameYrightZrightEdge, ghostXsameYrightZleftEdge);
    
    // parse
    MPI_Barrier(MPI_COMM_WORLD);
    parseEdgeZ(nx, ny, nz, vector, ns, ghostXrightYrightZsameEdge, ghostXleftYleftZsameEdge, ghostXrightYleftZsameEdge, ghostXleftYrightZsameEdge);
    parseEdgeY(nx, ny, nz, vector, ns, ghostXrightYsameZrightEdge, ghostXleftYsameZleftEdge, ghostXleftYsameZrightEdge, ghostXrightYsameZleftEdge);
    parseEdgeX(nx, ny, nz, vector, ns, ghostXsameYrightZrightEdge, ghostXsameYleftZleftEdge, ghostXsameYleftZrightEdge, ghostXsameYrightZleftEdge);

    makeNodeGhostCorner(nx, ny, nz, ghostXsameYrightZrightEdge, ghostXsameYleftZleftEdge, ghostXsameYleftZrightEdge, ghostXsameYrightZleftEdge, &ghostXrightYrightZrightCorner, &ghostXleftYrightZrightCorner, &ghostXrightYleftZrightCorner, &ghostXleftYleftZrightCorner, &ghostXrightYrightZleftCorner, &ghostXleftYrightZleftCorner, &ghostXrightYleftZleftCorner, &ghostXleftYleftZleftCorner);
    // communicate only in the X-DIRECTION
    communicateGhostFace(1, vct->getCartesian_rank(), vct->getXright_neighbor_P(), vct->getXleft_neighbor_P(), 0, vct->getXLEN(), vct->getYLEN(), vct->getZLEN(), &ghostXrightYrightZrightCorner, &ghostXleftYrightZrightCorner);
    communicateGhostFace(1, vct->getCartesian_rank(), vct->getXright_neighbor_P(), vct->getXleft_neighbor_P(), 0, vct->getXLEN(), vct->getYLEN(), vct->getZLEN(), &ghostXrightYleftZrightCorner, &ghostXleftYleftZrightCorner);
    communicateGhostFace(1, vct->getCartesian_rank(), vct->getXright_neighbor_P(), vct->getXleft_neighbor_P(), 0, vct->getXLEN(), vct->getYLEN(), vct->getZLEN(), &ghostXrightYleftZleftCorner, &ghostXleftYleftZleftCorner);
    communicateGhostFace(1, vct->getCartesian_rank(), vct->getXright_neighbor_P(), vct->getXleft_neighbor_P(), 0, vct->getXLEN(), vct->getYLEN(), vct->getZLEN(), &ghostXrightYrightZleftCorner, &ghostXleftYrightZleftCorner);
    
    // parse
    parseCorner(nx, ny, nz, vector, ns, &ghostXrightYrightZrightCorner, &ghostXleftYrightZrightCorner, &ghostXrightYleftZrightCorner, &ghostXleftYleftZrightCorner, &ghostXrightYrightZleftCorner, &ghostXleftYrightZleftCorner, &ghostXrightYleftZleftCorner, &ghostXleftYleftZleftCorner);

    delete[]ghostXrightFace;
    delete[]ghostXleftFace;
    delete[]ghostYrightFace;
    delete[]ghostYleftFace;
    delete[]ghostZrightFace;
    delete[]ghostZleftFace;
    // X EDGE
    delete[]ghostXsameYleftZleftEdge;
    delete[]ghostXsameYrightZleftEdge;
    delete[]ghostXsameYleftZrightEdge;
    delete[]ghostXsameYrightZrightEdge;
    // Y EDGE
    delete[]ghostXrightYsameZleftEdge;
    delete[]ghostXleftYsameZleftEdge;
    delete[]ghostXrightYsameZrightEdge;
    delete[]ghostXleftYsameZrightEdge;
    // Z EDGE
    delete[]ghostXrightYleftZsameEdge;
    delete[]ghostXrightYrightZsameEdge;
    delete[]ghostXleftYleftZsameEdge;
    delete[]ghostXleftYrightZsameEdge;
    // timeTasks.addto_communicate(); 
}
  
void communicateInterp_old(int nx, int ny, int nz, int ns, double ****vector,
                           int bcFaceXright, int bcFaceXleft, int bcFaceYright, int bcFaceYleft, int bcFaceZright, int bcFaceZleft, 
                           const VirtualTopology3D *vct, EMfields3D *EMf)
{
    // allocate 6 ghost cell Faces
    double *ghostXrightFace = new double[(ny - 2) * (nz - 2)];
    double *ghostXleftFace = new double[(ny - 2) * (nz - 2)];
    double *ghostYrightFace = new double[(nx - 2) * (nz - 2)];
    double *ghostYleftFace = new double[(nx - 2) * (nz - 2)];
    double *ghostZrightFace = new double[(nx - 2) * (ny - 2)];
    double *ghostZleftFace = new double[(nx - 2) * (ny - 2)];
    // allocate 12 ghost cell Edges
    // X EDGE
    double *ghostXsameYleftZleftEdge = new double[nx - 2];
    double *ghostXsameYrightZleftEdge = new double[nx - 2];
    double *ghostXsameYleftZrightEdge = new double[nx - 2];
    double *ghostXsameYrightZrightEdge = new double[nx - 2];
    // Y EDGE
    double *ghostXrightYsameZleftEdge = new double[ny - 2];
    double *ghostXleftYsameZleftEdge = new double[ny - 2];
    double *ghostXrightYsameZrightEdge = new double[ny - 2];
    double *ghostXleftYsameZrightEdge = new double[ny - 2];
    // Z EDGE
    double *ghostXrightYleftZsameEdge = new double[nz - 2];
    double *ghostXrightYrightZsameEdge = new double[nz - 2];
    double *ghostXleftYleftZsameEdge = new double[nz - 2];
    double *ghostXleftYrightZsameEdge = new double[nz - 2];
    // allocate 8 ghost cell corner
    double ghostXrightYrightZrightCorner, ghostXleftYrightZrightCorner, ghostXrightYleftZrightCorner, ghostXleftYleftZrightCorner;
    double ghostXrightYrightZleftCorner, ghostXleftYrightZleftCorner, ghostXrightYleftZleftCorner, ghostXleftYleftZleftCorner;
  
    // apply boundary condition to 6 Ghost Faces and communicate if necessary to 6 processors: along 3 DIRECTIONS
    makeCenterFace(nx, ny, nz, vector, ns, ghostXrightFace, ghostXleftFace, ghostYrightFace, ghostYleftFace, ghostZrightFace, ghostZleftFace);
    // communication
    communicateGhostFace((ny - 2) * (nz - 2), vct->getCartesian_rank(), vct->getXright_neighbor_P(), vct->getXleft_neighbor_P(), 0, vct->getXLEN(), vct->getYLEN(), vct->getZLEN(), ghostXrightFace, ghostXleftFace);
    communicateGhostFace((nx - 2) * (nz - 2), vct->getCartesian_rank(), vct->getYright_neighbor_P(), vct->getYleft_neighbor_P(), 1, vct->getXLEN(), vct->getYLEN(), vct->getZLEN(), ghostYrightFace, ghostYleftFace);
    communicateGhostFace((nx - 2) * (ny - 2), vct->getCartesian_rank(), vct->getZright_neighbor_P(), vct->getZleft_neighbor_P(), 2, vct->getXLEN(), vct->getYLEN(), vct->getZLEN(), ghostZrightFace, ghostZleftFace);
    addFace(nx, ny, nz, vector, ns, ghostXrightFace, ghostXleftFace, ghostYrightFace, ghostYleftFace, ghostZrightFace, ghostZleftFace, vct);
  /** prepare ghost cell Edge Y for communication: these are communicated in X direction */
    makeNodeEdgeY(nx, ny, nz, ghostZleftFace, ghostZrightFace, ghostXrightYsameZrightEdge, ghostXleftYsameZleftEdge, ghostXleftYsameZrightEdge, ghostXrightYsameZleftEdge);
  /** prepare ghost cell Edge Z for communication: these are communicated in Y direction */
    makeNodeEdgeZ(nx, ny, nz, ghostXleftFace, ghostXrightFace, ghostXrightYrightZsameEdge, ghostXleftYleftZsameEdge, ghostXrightYleftZsameEdge, ghostXleftYrightZsameEdge);
  /** prepare ghost cell Edge X for communication: these are communicated in  Z direction*/
    makeNodeEdgeX(nx, ny, nz, ghostYleftFace, ghostYrightFace, ghostXsameYrightZrightEdge, ghostXsameYleftZleftEdge, ghostXsameYleftZrightEdge, ghostXsameYrightZleftEdge);
  
    // communicate twice each direction
    // X-DIRECTION: Z -> X -> Y
    MPI_Barrier(MPI_COMM_WORLD);
    communicateGhostFace(ny - 2, vct->getCartesian_rank(), vct->getXright_neighbor_P(), vct->getXleft_neighbor_P(), 0, vct->getXLEN(), vct->getYLEN(), vct->getZLEN(), ghostXrightYsameZleftEdge, ghostXleftYsameZleftEdge);
    communicateGhostFace(ny - 2, vct->getCartesian_rank(), vct->getXright_neighbor_P(), vct->getXleft_neighbor_P(), 0, vct->getXLEN(), vct->getYLEN(), vct->getZLEN(), ghostXrightYsameZrightEdge, ghostXleftYsameZrightEdge);
    // Y-DIRECTION: X -> Y -> Z
    MPI_Barrier(MPI_COMM_WORLD);
    communicateGhostFace(nz - 2, vct->getCartesian_rank(), vct->getYright_neighbor_P(), vct->getYleft_neighbor_P(), 1, vct->getXLEN(), vct->getYLEN(), vct->getZLEN(), ghostXleftYrightZsameEdge, ghostXleftYleftZsameEdge);
    communicateGhostFace(nz - 2, vct->getCartesian_rank(), vct->getYright_neighbor_P(), vct->getYleft_neighbor_P(), 1, vct->getXLEN(), vct->getYLEN(), vct->getZLEN(), ghostXrightYrightZsameEdge, ghostXrightYleftZsameEdge);
    // Z-DIRECTION: Y -> Z
    MPI_Barrier(MPI_COMM_WORLD);
    communicateGhostFace(nx - 2, vct->getCartesian_rank(), vct->getZright_neighbor_P(), vct->getZleft_neighbor_P(), 2, vct->getXLEN(), vct->getYLEN(), vct->getZLEN(), ghostXsameYleftZrightEdge, ghostXsameYleftZleftEdge);
    communicateGhostFace(nx - 2, vct->getCartesian_rank(), vct->getZright_neighbor_P(), vct->getZleft_neighbor_P(), 2, vct->getXLEN(), vct->getYLEN(), vct->getZLEN(), ghostXsameYrightZrightEdge, ghostXsameYrightZleftEdge);
    // parse
    MPI_Barrier(MPI_COMM_WORLD);
    addEdgeZ(nx, ny, nz, vector, ns, ghostXrightYrightZsameEdge, ghostXleftYleftZsameEdge, ghostXrightYleftZsameEdge, ghostXleftYrightZsameEdge, vct);
    addEdgeY(nx, ny, nz, vector, ns, ghostXrightYsameZrightEdge, ghostXleftYsameZleftEdge, ghostXleftYsameZrightEdge, ghostXrightYsameZleftEdge, vct);
    addEdgeX(nx, ny, nz, vector, ns, ghostXsameYrightZrightEdge, ghostXsameYleftZleftEdge, ghostXsameYleftZrightEdge, ghostXsameYrightZleftEdge, vct);
  
    makeNodeCorner(nx, ny, nz, ghostXsameYrightZrightEdge, ghostXsameYleftZleftEdge, ghostXsameYleftZrightEdge, ghostXsameYrightZleftEdge, &ghostXrightYrightZrightCorner, &ghostXleftYrightZrightCorner, &ghostXrightYleftZrightCorner, &ghostXleftYleftZrightCorner, &ghostXrightYrightZleftCorner, &ghostXleftYrightZleftCorner, &ghostXrightYleftZleftCorner, &ghostXleftYleftZleftCorner);
    // communicate only in the X-DIRECTION
    communicateGhostFace(1, vct->getCartesian_rank(), vct->getXright_neighbor_P(), vct->getXleft_neighbor_P(), 0, vct->getXLEN(), vct->getYLEN(), vct->getZLEN(), &ghostXrightYrightZrightCorner, &ghostXleftYrightZrightCorner);
    communicateGhostFace(1, vct->getCartesian_rank(), vct->getXright_neighbor_P(), vct->getXleft_neighbor_P(), 0, vct->getXLEN(), vct->getYLEN(), vct->getZLEN(), &ghostXrightYleftZrightCorner, &ghostXleftYleftZrightCorner);
    communicateGhostFace(1, vct->getCartesian_rank(), vct->getXright_neighbor_P(), vct->getXleft_neighbor_P(), 0, vct->getXLEN(), vct->getYLEN(), vct->getZLEN(), &ghostXrightYleftZleftCorner, &ghostXleftYleftZleftCorner);
    communicateGhostFace(1, vct->getCartesian_rank(), vct->getXright_neighbor_P(), vct->getXleft_neighbor_P(), 0, vct->getXLEN(), vct->getYLEN(), vct->getZLEN(), &ghostXrightYrightZleftCorner, &ghostXleftYrightZleftCorner);
    // parse
    addCorner(nx, ny, nz, vector, ns, &ghostXrightYrightZrightCorner, &ghostXleftYrightZrightCorner, &ghostXrightYleftZrightCorner, &ghostXleftYleftZrightCorner, &ghostXrightYrightZleftCorner, &ghostXleftYrightZleftCorner, &ghostXrightYleftZleftCorner, &ghostXleftYleftZleftCorner, vct);
  
  
    delete[]ghostXrightFace;
    delete[]ghostXleftFace;
    delete[]ghostYrightFace;
    delete[]ghostYleftFace;
    delete[]ghostZrightFace;
    delete[]ghostZleftFace;
    // X EDGE
    delete[]ghostXsameYleftZleftEdge;
    delete[]ghostXsameYrightZleftEdge;
    delete[]ghostXsameYleftZrightEdge;
    delete[]ghostXsameYrightZrightEdge;
    // Y EDGE
    delete[]ghostXrightYsameZleftEdge;
    delete[]ghostXleftYsameZleftEdge;
    delete[]ghostXrightYsameZrightEdge;
    delete[]ghostXleftYsameZrightEdge;
    // Z EDGE
    delete[]ghostXrightYleftZsameEdge;
    delete[]ghostXrightYrightZsameEdge;
    delete[]ghostXleftYleftZsameEdge;
    delete[]ghostXleftYrightZsameEdge;
}

//* ============================================================================================================================ *//