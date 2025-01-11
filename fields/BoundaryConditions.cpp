/* 
* This file consists of functions defining boundary conditions that are 
* are members of the class EMfields3D.h. - PJD
*/

#include "EMfields3D.h"

#define NUM_NODE_BC 3
#define NUM_CELL_BC 3

/*
 * Boundary Conditions for B:
 *  
 * 0 - perfect conductor modified: B in the first/last NUM_CELL_BC cells is set equal to the next ones inside
 *
 * 1 - magnetic mirror modified: B in the first/last NUM_CELL_BC cells is set  from reference state
 *
 * 2 - B in the first/last NUM_CELL_BC cells is set  from the values in the input file
 *
 * 3 - open boundary (linear extrapolation)
 *
 * Boundary Conditions for E (assuming non Lagrangian):
 *
 * Source in x:
 * 0: Source in Maxwell equations is set to VxB on the first/last node in x; V --> speed of species 0 and B --> magnetic field from input file
 * 1: Source in Maxwell equations is set to E in the reference state
 * 2: Source in Maxwell equations is set to VxB on the first/last NUM_NODE_BC node in x, with vxB taken from input file
 * 3: Source left as is to continue applying the Maxwell equations, appropriate for open BC without local shock along the boundary
 *
 * Source in y & z:
 * 0, 2: Source in Maxwell equations is set to VxB on the first/last node in y and z
 * 1: Source is equal to E in the reference state
 * 3: Source in last node equal to the next one inside, as needed to try to let a shock slide along the boundary, this avoids boundary issues in the sources
 *
 *
 * Image (operator) in x:
 * 0, 1: Image in Maxwell equations is set to E on the first/last node in x
 * 2: Image in Maxwell equations is set to E on the first/last NUM_NODE_BC node in x
 * 1: Before Newman condition is applied to dEx/dx, other components are set to zero
 * 3: Image left as is to continue applying the Maxwell equations, appropriate for open BC without local shock along the boundary
 *
 * Image in y & z:
 * 0, 1, 2: Image in Maxwell equations is set to zero on the first/last node  in y and z
 * 1: Before Newman condition is applied to dEx/dx, other components are set to zero
 * 3: Image left as is to continue applying the Maxwell equations, appropriate for open BC without local shock along the boundary

 */

void EMfields3D::fixBC_B() 
{
    double vx    = u_bulk;
    double vy    = v_bulk;
    double vz    = w_bulk;
    double tmpBx = Fzro*B0x;
    double tmpBy = Fzro*B0y;
    double tmpBz = Fzro*B0z;
    double tmpEx = vz*Fzro*B0y-vy*Fzro*B0z;
    double tmpEy = vx*Fzro*B0z-vz*Fzro*B0x;
    double tmpEz = vy*Fzro*B0x-vx*Fzro*B0y;

    double valx = 0;
    double valy = 0;
    double valz = 0;

    double* vect;
    vect = new double[NUM_CELL_BC];
    double lin_reg0, lin_reg1;

    if (col->getLagrangian() == 1) 
    {
        valx = tmpEx + (vy*tmpBz - vz*tmpBy);
        valy = tmpEy + (vz*tmpBx - vx*tmpBz);
        valz = tmpEz + (vx*tmpBy - vy*tmpBx);
    } 
    else 
    {
        valx = tmpEx;
        valy = tmpEy;
        valz = tmpEz;
    }

  //? X left
  if (vct->getXleft_neighbor() == MPI_PROC_NULL) 
  {
	  if (bcEMfaceXleft == 0) 
      {
		  for (int i=0; i<NUM_CELL_BC; i++)
			  for (int j=0; j<nyc; j++)
				  for (int k=0; k<nzc; k++) {
					  Bxc[i][j][k] = Bxc[i+NUM_CELL_BC][j][k];
					  Byc[i][j][k] = Byc[i+NUM_CELL_BC][j][k];
					  Bzc[i][j][k] = Bzc[i+NUM_CELL_BC][j][k];
				  }
	  }
    else if (bcEMfaceXleft == 1) {
    	for (int i=0; i<NUM_CELL_BC; i++)
        for (int j=0; j<nyc; j++)
          for (int k=0; k<nzc; k++) {
            Bxc[i][j][k] = Bxc_rs[i][j][k];
            Byc[i][j][k] = Byc_rs[i][j][k];
            Bzc[i][j][k] = Bzc_rs[i][j][k];
          }
    }
    else if (bcEMfaceXleft == 2) {
      for (int i=0; i<NUM_CELL_BC; i++)
        for (int j=0; j<nyc; j++) 
          for (int k=0; k<nzc; k++) {
            Bxc[i][j][k] = tmpBx;
            Byc[i][j][k] = tmpBy;
            Bzc[i][j][k] = tmpBz;
          }
    } else if (bcEMfaceXleft == 3) {

    	for (int j=0; j<nyc; j++)
    	          for (int k=0; k<nzc; k++) {
    	        	 for (int i=0; i<NUM_CELL_BC; i++)
    				vect[i] = Bxc[i+1][j][k];
    			estimate_coef(vect, NUM_CELL_BC, lin_reg0, lin_reg1);
    			//cout << "Bxc: "<< lin_reg0 <<"  "<< lin_reg1<<endl;
    			Bxc[0][j][k] = lin_reg0 - lin_reg1;

    			for (int i=0; i<NUM_CELL_BC; i++)
    				vect[i] = Byc[i+1][j][k];
    			estimate_coef(vect, NUM_CELL_BC, lin_reg0, lin_reg1);
    			//cout << "Byc: "<< lin_reg0 <<"  "<< lin_reg1<<endl;
    			Byc[0][j][k] = lin_reg0 - lin_reg1;

    			for (int i=0; i<NUM_CELL_BC; i++)
    				vect[i] = Bzc[i+1][j][k];
    			estimate_coef(vect, NUM_CELL_BC, lin_reg0, lin_reg1);
    			//cout << "Bzc: "<< lin_reg0 <<"  "<< lin_reg1<<endl;
    			Bzc[0][j][k] = lin_reg0 - lin_reg1;

            }
      }else {
      fprintf(stderr,"ERROR: Boundary B-Xl contitions (%d) not implemented for the fields\n", bcEMfaceXleft);
      MPI_Abort(MPI_COMM_WORLD, -1);
    }
  }

  //? X right
  if (vct->getXright_neighbor() == MPI_PROC_NULL) {
	  if (bcEMfaceXright == 0) {
		  for (int i=0; i<NUM_CELL_BC; i++)
			  for (int j=0; j<nyc; j++)
				  for (int k=0; k<nzc; k++) {
					  Bxc[nxc-1-i][j][k] = Bxc[nxc-1-i-NUM_CELL_BC][j][k];
					  Byc[nxc-1-i][j][k] = Byc[nxc-1-i-NUM_CELL_BC][j][k];
					  Bzc[nxc-1-i][j][k] = Bzc[nxc-1-i-NUM_CELL_BC][j][k];
				  }
	  }
    else if (bcEMfaceXright == 1) {
    	for (int i=0; i<NUM_CELL_BC; i++)
    	        for (int j=0; j<nyc; j++)
    	          for (int k=0; k<nzc; k++) {
    	            Bxc[nxc-1-i][j][k] = Bxc_rs[nxc-1-i][j][k];
    	            Byc[nxc-1-i][j][k] = Byc_rs[nxc-1-i][j][k];
    	            Bzc[nxc-1-i][j][k] = Bzc_rs[nxc-1-i][j][k];
    	          }
    }
    else if (bcEMfaceXright == 2) {
      for (int i=0; i<NUM_CELL_BC; i++)
        for (int j=0; j<nyc; j++) 
          for (int k=0; k<nzc; k++) {
            Bxc[nxc-1-i][j][k] = tmpBx;
            Byc[nxc-1-i][j][k] = tmpBy;
            Bzc[nxc-1-i][j][k] = tmpBz;
          }
    } else if (bcEMfaceXright == 3) {

    	for (int j=0; j<nyc; j++)
    	          for (int k=0; k<nzc; k++) {
    	        	 for (int i=0; i<NUM_CELL_BC; i++)
    				vect[i] = Bxc[nxc-2-i][j][k];
    			estimate_coef(vect, NUM_CELL_BC, lin_reg0, lin_reg1);
    			Bxc[nxc-1][j][k] = lin_reg0 - lin_reg1;


    			for (int i=0; i<NUM_CELL_BC; i++)
    				vect[i] = Byc[nxc-2-i][j][k];
    			estimate_coef(vect, NUM_CELL_BC, lin_reg0, lin_reg1);
    			Byc[nxc-1][j][k] = lin_reg0 - lin_reg1;

    			for (int i=0; i<NUM_CELL_BC; i++)
    				vect[i] = Bzc[nxc-2-i][j][k];
    			estimate_coef(vect, NUM_CELL_BC, lin_reg0, lin_reg1);
    			Bzc[nxc-1][j][k] = lin_reg0 - lin_reg1;

    		}
     } else {
      fprintf(stderr,"ERROR: Boundary X-Xr contitions (%d) not implemented for the fields\n", bcEMfaceXright);
      MPI_Abort(MPI_COMM_WORLD, -1);
    }
  }

  // Y left
  if (vct->getYleft_neighbor() == MPI_PROC_NULL) {
    if (bcEMfaceYleft == 0) {
   	 for (int j=0; j<NUM_CELL_BC; j++)
   	        for (int i=0; i<nxc; i++)
   	          for (int k=0; k<nzc; k++) {
   	            Bxc[i][j][k] = Bxc[i][j + NUM_CELL_BC][k];
   	            Byc[i][j][k] = Byc[i][j + NUM_CELL_BC][k];
   	            Bzc[i][j][k] = Bzc[i][j + NUM_CELL_BC][k];
   	          }
    }
    else if (bcEMfaceYleft == 1) {
    	 for (int j=0; j<NUM_CELL_BC; j++)
    	        for (int i=0; i<nxc; i++)
    	          for (int k=0; k<nzc; k++) {
    	            Bxc[i][j][k] = Bxc_rs[i][j][k];
    	            Byc[i][j][k] = Byc_rs[i][j][k];
    	            Bzc[i][j][k] = Bzc_rs[i][j][k];
    	          }
    }
    else if (bcEMfaceYleft == 2) {
      for (int j=0; j<NUM_CELL_BC; j++)
        for (int i=0; i<nxc; i++) 
          for (int k=0; k<nzc; k++) {
            Bxc[i][j][k] = tmpBx;
            Byc[i][j][k] = tmpBy;
            Bzc[i][j][k] = tmpBz;
          }

    } else if (bcEMfaceYleft == 3) {

    	for (int i=0; i<nxc; i++)
    		for (int k=0; k<nzc; k++){
    			for (int j=0; j<NUM_CELL_BC; j++)
    				vect[j] = Bxc[i][j+1][k];
    			estimate_coef(vect, NUM_CELL_BC, lin_reg0, lin_reg1);
    			//cout << "Bxc: "<< lin_reg0 <<"  "<< lin_reg1<<endl;
    			Bxc[i][0][k] = lin_reg0 - lin_reg1;

    			for (int j=0; j<NUM_CELL_BC; j++)
    				vect[j] = Byc[i][j+1][k];
    			estimate_coef(vect, NUM_CELL_BC, lin_reg0, lin_reg1);
    			//cout << "Byc: "<< lin_reg0 <<"  "<< lin_reg1<<endl;
    			Byc[i][0][k] = lin_reg0 - lin_reg1;

    			for (int j=0; j<NUM_CELL_BC; j++)
    				vect[j] = Bzc[i][j+1][k];
    			estimate_coef(vect, NUM_CELL_BC, lin_reg0, lin_reg1);
    			//cout << "Bzc: "<< lin_reg0 <<"  "<< lin_reg1<<endl;
    			Bzc[i][0][k] = lin_reg0 - lin_reg1;

            }
      }else {
      fprintf(stderr,"ERROR: Boundary B-Yl contitions (%d) not implemented for the fields\n", bcEMfaceYleft);
      MPI_Abort(MPI_COMM_WORLD, -1);
    }
  }

  // Y right
  if (vct->getYright_neighbor() == MPI_PROC_NULL) {
    if (bcEMfaceYright == 0) {
      	 for (int j=0; j<NUM_CELL_BC; j++)
       	        for (int i=0; i<nxc; i++)
       	          for (int k=0; k<nzc; k++) {
       	            Bxc[i][nyc-1-j][k] = Bxc[i][nyc-1-j- NUM_CELL_BC][k];
       	            Byc[i][nyc-1-j][k] = Byc[i][nyc-1-j- NUM_CELL_BC][k];
       	            Bzc[i][nyc-1-j][k] = Bzc[i][nyc-1-j- NUM_CELL_BC][k];
       	          }
    }
    else if (bcEMfaceYright == 1) {
    	 for (int j=0; j<NUM_CELL_BC; j++)
    	        for (int i=0; i<nxc; i++)
    	          for (int k=0; k<nzc; k++) {
    	            Bxc[i][nyc-1-j][k] = Bxc_rs[i][nyc-1-j][k];
    	            Byc[i][nyc-1-j][k] = Byc_rs[i][nyc-1-j][k];
    	            Bzc[i][nyc-1-j][k] = Bzc_rs[i][nyc-1-j][k];
    	          }
    }
    else if (bcEMfaceYright == 2) {
      for (int j=0; j<NUM_CELL_BC; j++)
        for (int i=0; i<nxc; i++) 
          for (int k=0; k<nzc; k++) {
            Bxc[i][nyc-1-j][k] = tmpBx;
            Byc[i][nyc-1-j][k] = tmpBy;
            Bzc[i][nyc-1-j][k] = tmpBz;
          }

    } else if (bcEMfaceYright == 3) {

    	for (int i=0; i<nxc; i++)
    		for (int k=0; k<nzc; k++){
    			for (int j=0; j<NUM_CELL_BC; j++)
    				vect[j] = Bxc[i][nyc-2-j][k];
    			estimate_coef(vect, NUM_CELL_BC, lin_reg0, lin_reg1);
    			Bxc[i][nyc-1][k] = lin_reg0 - lin_reg1;


    			for (int j=0; j<NUM_CELL_BC; j++)
    				vect[j] = Byc[i][nyc-2-j][k];
    			estimate_coef(vect, NUM_CELL_BC, lin_reg0, lin_reg1);
    			Byc[i][nyc-1][k] = lin_reg0 - lin_reg1;

    			for (int j=0; j<NUM_CELL_BC; j++)
    				vect[j] = Bzc[i][nyc-2-j][k];
    			estimate_coef(vect, NUM_CELL_BC, lin_reg0, lin_reg1);
    			Bzc[i][nyc-1][k] = lin_reg0 - lin_reg1;

    		}
     } else {
      fprintf(stderr,"ERROR: Boundary B-Yr contitions (%d) not implemented for the fields\n", bcEMfaceYright);
      MPI_Abort(MPI_COMM_WORLD, -1);
    }
  }

  // Z left
  if (vct->getZleft_neighbor() == MPI_PROC_NULL) {
    if (bcEMfaceZleft == 0) {
    	for (int k=0; k<NUM_CELL_BC; k++)
    	        for (int i=0; i<nxc; i++)
    	          for (int j=0; j<nyc; j++) {
    	            Bxc[i][j][k] = Bxc[i][j][k+NUM_CELL_BC];
    	            Byc[i][j][k] = Byc[i][j][k+NUM_CELL_BC];
    	            Bzc[i][j][k] = Bzc[i][j][k+NUM_CELL_BC];
    	          }
    }
    else if (bcEMfaceZleft == 1) {
    	for (int k=0; k<NUM_CELL_BC; k++)
    	        for (int i=0; i<nxc; i++)
    	          for (int j=0; j<nyc; j++) {
    	            Bxc[i][j][k] = Bxc_rs[i][j][k];
    	            Byc[i][j][k] = Byc_rs[i][j][k];
    	            Bzc[i][j][k] = Bzc_rs[i][j][k];
    	          }
    }
    else if (bcEMfaceZleft == 2) {
      for (int k=0; k<NUM_CELL_BC; k++)
        for (int i=0; i<nxc; i++) 
          for (int j=0; j<nyc; j++) {
            Bxc[i][j][k] = tmpBx;
            Byc[i][j][k] = tmpBy;
            Bzc[i][j][k] = tmpBz;
          }
    } else if (bcEMfaceZleft == 3) {
        for (int i=0; i<nxc; i++)
          for (int j=0; j<nyc; j++){
        	  for (int k=0; k<NUM_CELL_BC; k++)
    				vect[k] = Bxc[i][j][k+1];
    			estimate_coef(vect, NUM_CELL_BC, lin_reg0, lin_reg1);
    			Bxc[i][j][0] = lin_reg0 - lin_reg1;

    			for (int k=0; k<NUM_CELL_BC; k++)
    				vect[k] = Byc[i][j][k+1];
    			estimate_coef(vect, NUM_CELL_BC, lin_reg0, lin_reg1);
    			Byc[i][j][0] = lin_reg0 - lin_reg1;

    			for (int k=0; k<NUM_CELL_BC; k++)
    				vect[k] = Bzc[i][j][k+1];
    			estimate_coef(vect, NUM_CELL_BC, lin_reg0, lin_reg1);
    			Bzc[i][j][0] = lin_reg0 - lin_reg1;

            }
      }
    else {
      fprintf(stderr,"ERROR: Boundary B-Zl contitions (%d) not implemented for the fields\n", bcEMfaceZleft);
      MPI_Abort(MPI_COMM_WORLD, -1);
    }
  }

  // Z right
  if (vct->getZright_neighbor() == MPI_PROC_NULL) {
    if (bcEMfaceZright == 0) {
    	for (int k=0; k<NUM_CELL_BC; k++)
    	        for (int i=0; i<nxc; i++)
    	          for (int j=0; j<nyc; j++) {
    	            Bxc[i][j][nzc-1-k] = Bxc[i][j][nzc-1-k-NUM_CELL_BC];
    	            Byc[i][j][nzc-1-k] = Byc[i][j][nzc-1-k-NUM_CELL_BC];
    	            Bzc[i][j][nzc-1-k] = Bzc[i][j][nzc-1-k-NUM_CELL_BC];
    	          }
    }
    else if (bcEMfaceZright == 1) {
    	for (int k=0; k<NUM_CELL_BC; k++)
    	        for (int i=0; i<nxc; i++)
    	          for (int j=0; j<nyc; j++) {
    	            Bxc[i][j][nzc-1-k] = Bxc_rs[i][j][nzc-1-k];
    	            Byc[i][j][nzc-1-k] = Byc_rs[i][j][nzc-1-k];
    	            Bzc[i][j][nzc-1-k] = Bzc_rs[i][j][nzc-1-k];
    	          }
    }
    else if (bcEMfaceZright == 2) {
      for (int k=0; k<NUM_CELL_BC; k++)
        for (int i=0; i<nxc; i++) 
          for (int j=0; j<nyc; j++) {
            Bxc[i][j][nzc-1-k] = tmpBx;
            Byc[i][j][nzc-1-k] = tmpBy;
            Bzc[i][j][nzc-1-k] = tmpBz;
          }
    } else if (bcEMfaceZright == 3) {
        for (int i=0; i<nxc; i++)
           for (int j=0; j<nyc; j++){
        	   for (int k=0; k<NUM_CELL_BC; k++)
    				vect[k] = Bxc[i][j][nzc-2-k];
    			estimate_coef(vect, NUM_CELL_BC, lin_reg0, lin_reg1);
    			Bxc[i][j][nzc-1] = lin_reg0 - lin_reg1;


    			for (int k=0; k<NUM_CELL_BC; k++)
    				vect[k] = Byc[i][j][nzc-2-k];
    			estimate_coef(vect, NUM_CELL_BC, lin_reg0, lin_reg1);
    			Byc[i][j][nzc-1] = lin_reg0 - lin_reg1;

    			for (int k=0; k<NUM_CELL_BC; k++)
    				vect[k] = Bzc[i][j][nzc-2-k];
    			estimate_coef(vect, NUM_CELL_BC, lin_reg0, lin_reg1);
    			Bzc[i][j][nzc-1] = lin_reg0 - lin_reg1;
    		}
      }
    else {
      fprintf(stderr,"ERROR: Boundary B-Zr contitions (%d) not implemented for the fields\n", bcEMfaceZright);
      MPI_Abort(MPI_COMM_WORLD, -1);
    }
  }
}