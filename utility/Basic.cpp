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
#include "ipicdefs.h"
#include "Basic.h"
#include "EllipticF.h"
#include "Alloc.h"
#include "TimeTasks.h"
#include "errors.h"

/** method to calculate the parallel dot product with vect1, vect2 having the ghost cells*/
double dotP(const double *vect1, const double *vect2, int n,MPI_Comm* comm) {
  double result = 0;
  double local_result = 0;
  for (int i = 0; i < n; i++)
    local_result += vect1[i] * vect2[i];
  MPI_Allreduce(&local_result, &result, 1, MPI_DOUBLE, MPI_SUM, *comm);
  return (result);

}
/** method to calculate dot product */
double dot(const double *vect1, const double *vect2, int n) {
  double result = 0;
  for (int i = 0; i < n; i++)
    result += vect1[i] * vect2[i];
  return (result);
}


//? ========================= L2 norm ========================= ?//

double norm2(const double *vect, int nx) 
{
    double result = 0;
    
    for (int i = 0; i < nx; i++)
        result += vect[i] * vect[i];
    
        return (result);
}

double norm2(const double *const*vect, int nx, int ny) 
{
    double result = 0;
    
    for (int i = 0; i < nx; i++)
        for (int j = 0; j < ny; j++)
            result += vect[i][j] * vect[i][j];
    
            return (result);
}
double norm2(const arr3_double vect, int nx, int ny) 
{
    double result = 0;
    
    for (int i = 0; i < nx; i++)
        for (int j = 0; j < ny; j++)
            result += vect.get(i,j,0) * vect.get(i,j,0);
    
            return (result);
}

double norm2(const arr3_double vect, int nx, int ny, int nz) 
{
    double result = 0;
    
    for (int i = 0; i < nx; i++)
        for (int j = 0; j < ny; j++)
            for (int k = 0; k < nz; j++)
                result += vect.get(i,j,k) * vect.get(i,j,k);
    
            return (result);
}

double norm2(double ***vect, int nx, int ny) 
{
    double result = 0;
    
    for (int i = 0; i < nx; i++)
        for (int j = 0; j < ny; j++)
            result += vect[i][j][0] * vect[i][j][0];

    return (result);
}

double norm2(double ***vect, int nx, int ny, int nz) 
{
    double result = 0;
    
    for (int i = 0; i < nx; i++)
        for (int j = 0; j < ny; j++)
            for (int k = 0; k < nz; k++)
                result += vect[i][j][k] * vect[i][j][k];

    return (result);
  }

//* Compute parallel norm of a vector on different processors with the ghost cell
double normP(const double *vect, int n,MPI_Comm* comm) 
{
    double result = 0.0;
    double local_result = 0.0;
    
    for (int i = 0; i < n; i++)
        local_result += vect[i] * vect[i];
    
    MPI_Allreduce(&local_result, &result, 1, MPI_DOUBLE, MPI_SUM, *comm);
    
    return (sqrt(result));
}

double norm2P(double ***vect, int nx, int ny, int nz) 
{
    double result = 0;
    double local_result = 0;
    for (int i = 0; i < nx; i++)
        for (int j = 0; j < ny; j++)
            for (int k = 0; k < nz; k++)
                local_result += vect[i][j][k] * vect[i][j][k];

    MPI_Allreduce(&local_result, &result, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return (result);
}

//? =========================================================== ?//

/** method to calculate the difference of two vectors*/
void sub(double *res, const double *vect1, const double *vect2, int n) {
  for (int i = 0; i < n; i++)
    res[i] = vect1[i] - vect2[i];
}
/** method to calculate the sum of two vectors vector1 = vector1 + vector2*/
void sum(double *vect1, const double *vect2, int n) {
  for (int i = 0; i < n; i++)
    vect1[i] += vect2[i];
}
/** method to calculate the sum of two vectors vector1 = vector1 + vector2*/
void sum(arr3_double vect1, const arr3_double vect2, int nx, int ny, int nz) {
  for (int i = 0; i < nx; i++)
    for (int j = 0; j < ny; j++)
      for (int k = 0; k < nz; k++)
        vect1.fetch(i,j,k) += vect2.get(i,j,k);
}

/** method to calculate the sum of two vectors vector1 = vector1 + vector2*/
void sum(arr3_double vect1, const arr3_double vect2, int nx, int ny) {
  for (int i = 0; i < nx; i++)
    for (int j = 0; j < ny; j++)
      vect1.fetch(i,j,0) += vect2.get(i,j,0);
}

/** method to calculate the sum of two vectors vector1 = vector1 + vector2*/
void sum(arr3_double vect1, const arr4_double vect2, int nx, int ny, int nz, int ns) {
  for (int i = 0; i < nx; i++)
    for (int j = 0; j < ny; j++)
      for (int k = 0; k < nz; k++)
        vect1.fetch(i,j,k) += vect2.get(ns,i,j,k);
}

/** method to calculate the sum of two vectors vector1 = vector1 + vector2*/
void sum(arr3_double vect1, const arr4_double vect2, int nx, int ny, int ns) {
  for (int i = 0; i < nx; i++)
    for (int j = 0; j < ny; j++)
      vect1.fetch(i,j,0) += vect2.get(ns,i,j,0);
}
/** method to calculate the subtraction of two vectors vector1 = vector1 - vector2*/
void sub(arr3_double vect1, const arr3_double vect2, int nx, int ny, int nz) {
  for (int i = 0; i < nx; i++)
    for (int j = 0; j < ny; j++)
      for (int k = 0; k < nz; k++)
        vect1.fetch(i,j,k) -= vect2.get(i,j,k);
}

/** method to calculate the subtraction of two vectors vector1 = vector1 - vector2*/
void sub(arr3_double vect1, const arr3_double vect2, int nx, int ny) {
  for (int i = 0; i < nx; i++)
    for (int j = 0; j < ny; j++)
      vect1.fetch(i,j,0) -= vect2.get(i,j,0);
}


/** method to sum 4 vectors vector1 = alfa*vector1 + beta*vector2 + gamma*vector3 + delta*vector4 */
void sum4(arr3_double vect1, double alfa, const arr3_double vect2, double beta, const arr3_double vect3, double gamma, const arr3_double vect4, double delta, const arr3_double vect5, int nx, int ny, int nz) {
  for (int i = 0; i < nx; i++)
    for (int j = 0; j < ny; j++)
      for (int k = 0; k < nz; k++)
        vect1.fetch(i,j,k) = alfa * (vect2.get(i,j,k) + beta * vect3.get(i,j,k) + gamma * vect4.get(i,j,k) + delta * vect5.get(i,j,k));

}
/** method to calculate the scalar-vector product */
void scale(double *vect, double alfa, int n) {
  for (int i = 0; i < n; i++)
    vect[i] *= alfa;
}

//* vector = alfa * vector
void scale(arr3_double vect, double alfa, int nx, int ny) 
{
    for (int i = 0; i < nx; i++)
        for (int j = 0; j < ny; j++)
            vect.fetch(i,j,0) *= alfa;
}

void scale(arr3_double vect, double alfa, int nx, int ny, int nz) 
{
    for (int i = 0; i < nx; i++)
        for (int j = 0; j < ny; j++)
            for (int k = 0; k < nz; k++)
                vect.fetch(i,j,k) *= alfa;
}

//* vector_1 = alfa * vector_2
void scale(arr3_double vect1, const arr3_double vect2, double alfa, int nx, int ny, int nz)
{
    for (int i = 0; i < nx; i++)
        for (int j = 0; j < ny; j++)
            for (int k = 0; k < nz; k++)
                vect1.fetch(i,j,k) = vect2.get(i,j,k) * alfa;
}

void scale(arr4_double vect1, const arr3_double vect2, double alfa, int ns, int nx, int ny, int nz)
{
    for (int is = 0; is < ns; is++)
        for (int i = 0; i < nx; i++)
            for (int j = 0; j < ny; j++)
                for (int k = 0; k < nz; k++)
                    vect1.fetch(is,i,j,k) = vect2.get(i,j,k) * alfa;
}

void scale(arr3_double vect1, const arr3_double vect2, double alfa, int nx, int ny) 
{
    for (int i = 0; i < nx; i++)
        for (int j = 0; j < ny; j++)
            vect1.fetch(i,j,0) = vect2.get(i,j,0) * alfa;
}

void scale(double *vect1, const double *vect2, double alfa, int n) 
{
    for (int i = 0; i < n; i++)
        vect1[i] = vect2[i] * alfa;
}

//* vector3 = vector1 + alfa*vector2
void addscale(double alfa, arr3_double vect1, arr3_double vect2, const arr3_double vect3, int nx, int ny, int nz)
{
    for (int i = 0; i < nx; i++)
	    for (int j = 0; j < ny; j++)
	        for (int k = 0; k < nz; k++)
	            vect3.fetch(i,j,k) = vect1.get(i,j,k) + alfa * vect2.get(i,j,k);
}

//* vector1 = vector1 + alfa*vector2
void addscale(double alfa, arr4_double vect1, const arr3_double vect2, int ns, int nx, int ny, int nz)
{
    for (int is = 0; is < ns; is++)
        for (int i = 0; i < nx; i++)
            for (int j = 0; j < ny; j++)
                for (int k = 0; k < nz; k++)
                    vect1.fetch(is,i,j,k) = vect1.get(is,i,j,k) + alfa * vect2.get(i,j,k);
}

void addscale(double alfa, arr4_double vect1, const arr4_double vect2, int ns, int nx, int ny, int nz)
{
    for (int is = 0; is < ns; is++)
        for (int i = 0; i < nx; i++)
            for (int j = 0; j < ny; j++)
                for (int k = 0; k < nz; k++)
                    vect1.fetch(is,i,j,k) = vect1.get(is,i,j,k) + alfa * vect2.get(is,i,j,k);
}

void addscale(double alfa, arr3_double vect1, const arr3_double vect2, int nx, int ny, int nz)
{
    for (int i = 0; i < nx; i++)
        for (int j = 0; j < ny; j++)
            for (int k = 0; k < nz; k++)
                vect1.fetch(i,j,k) = vect1.get(i,j,k) + alfa * vect2.get(i,j,k);
}

void addscale(double alfa, double vect1[][2][2], double vect2[][2][2], int nx, int ny, int nz)
{
    for (int i = 0; i < nx; i++)
        for (int j = 0; j < ny; j++)
            for (int k = 0; k < nz; k++)
                vect1[i][j][k] = vect1[i][j][k] + alfa * vect2[i][j][k];
}

void addscale(double alfa, arr3_double vect1, const arr3_double vect2, int nx, int ny) 
{
    for (int i = 0; i < nx; i++)
        for (int j = 0; j < ny; j++)
            vect1.fetch(i,j,0) += alfa * vect2.get(i,j,0);
}

void addscale(double alfa, double *vect1, const double *vect2, int n) 
{
    for (int i = 0; i < n; i++)
        vect1[i] += alfa * vect2[i];
}

//* vector1 = beta*vector1 + alfa*vector2 
void addscale(double alfa, double beta, double *vect1, const double *vect2, int n) 
{
    for (int i = 0; i < n; i++)
        vect1[i] = vect1[i] * beta + alfa * vect2[i];
}

void addscale(double alfa, double beta, arr3_double vect1, const arr3_double vect2, int nx, int ny, int nz)
{
    for (int i = 0; i < nx; i++)
        for (int j = 0; j < ny; j++)
            for (int k = 0; k < nz; k++)
                vect1.fetch(i,j,k) = beta * vect1.get(i,j,k) + alfa * vect2.get(i,j,k);
}

void addscale(double alfa, double beta, arr3_double vect1, const arr3_double vect2, int nx, int ny)
{
    for (int i = 0; i < nx; i++)
        for (int j = 0; j < ny; j++)
            vect1.fetch(i,j,0) = beta * vect1.get(i,j,0) + alfa * vect2.get(i,j,0);
}

/** method to calculate vector1 = alfa*vector2 + beta*vector3 */
void scaleandsum(arr3_double vect1, double alfa, double beta, const arr3_double vect2, const arr3_double vect3, int nx, int ny, int nz) {
  for (int i = 0; i < nx; i++)
    for (int j = 0; j < ny; j++)
      for (int k = 0; k < nz; k++)
        vect1.fetch(i,j,k) = alfa * vect2.get(i,j,k) + beta * vect3.get(i,j,k);
}
/** method to calculate vector1 = alfa*vector2 + beta*vector3 with vector2 depending on species*/
void scaleandsum(arr3_double vect1, double alfa, double beta, const arr4_double vect2, const arr3_double vect3, int ns, int nx, int ny, int nz) {
  for (int i = 0; i < nx; i++)
    for (int j = 0; j < ny; j++)
      for (int k = 0; k < nz; k++)
        vect1.fetch(i,j,k) = alfa * vect2.get(ns,i,j,k) + beta * vect3.get(i,j,k);
}
/** method to calculate vector1 = alfa*vector2*vector3 with vector2 depending on species*/
void prod(arr3_double vect1, double alfa, const arr4_double vect2, int ns, const arr3_double vect3, int nx, int ny, int nz) {
  for (int i = 0; i < nx; i++)
    for (int j = 0; j < ny; j++)
      for (int k = 0; k < nz; k++)
        vect1.fetch(i,j,k) = alfa * vect2.get(ns,i,j,k) * vect3.get(i,j,k);

}
/** method to calculate vect1 = vect2/alfa */
void div(arr3_double vect1, double alfa, const arr3_double vect2, int nx, int ny, int nz) {
  for (int i = 0; i < nx; i++)
    for (int j = 0; j < ny; j++)
      for (int k = 0; k < nz; k++)
        vect1.fetch(i,j,k) = vect2.get(i,j,k) / alfa;

}
void prod6(arr3_double vect1, const arr3_double vect2, const arr3_double vect3, const arr3_double vect4, const arr3_double vect5, const arr3_double vect6, const arr3_double vect7, int nx, int ny, int nz) {
  for (int i = 0; i < nx; i++)
    for (int j = 0; j < ny; j++)
      for (int k = 0; k < nz; k++)
        vect1.fetch(i,j,k) = vect2.get(i,j,k) * vect3.get(i,j,k) + vect4.get(i,j,k) * vect5.get(i,j,k) + vect6.get(i,j,k) * vect7.get(i,j,k);
}
/** method used for calculating PI */
void proddiv(arr3_double vect1, const arr3_double vect2, double alfa, const arr3_double vect3, const arr3_double vect4, const arr3_double vect5, const arr3_double vect6, double beta, const arr3_double vect7, const arr3_double vect8, double gamma, const arr3_double vect9, int nx, int ny, int nz) {
  for (int i = 0; i < nx; i++)
    for (int j = 0; j < ny; j++)
      for (int k = 0; k < nz; k++)
        vect1.fetch(i,j,k) = (vect2.get(i,j,k) + alfa * (vect3.get(i,j,k) * vect4.get(i,j,k) - vect5.get(i,j,k) * vect6.get(i,j,k)) + beta * vect7.get(i,j,k) * vect8.get(i,j,k)) / (1 + gamma * vect9.get(i,j,k));

  // questo mi convince veramente poco!!!!!!!!!!!!!! CAZZO!!!!!!!!!!!!!!!!!!
  // ***vect1++ = (***vect2++ + alfa*((***vect3++)*(***vect4++) - (***vect5++)*(***vect6++)) + beta*(***vect7++)*(***vect8++))/(1+gamma*(***vect9++));
}
/** method to calculate the opposite of a vector */
void neg(arr3_double vect, int nx, int ny, int nz) {
  for (int i = 0; i < nx; i++)
    for (int j = 0; j < ny; j++)
      for (int k = 0; k < nz; k++)
        vect.fetch(i,j,k) = -vect.get(i,j,k);
}

/** method to calculate the opposite of a vector */
void neg(arr3_double vect, int nx, int ny) {
  for (int i = 0; i < nx; i++)
    for (int j = 0; j < ny; j++)
      vect.fetch(i,j,0) = -vect.get(i,j,0);
}
/** method to calculate the opposite of a vector */
void neg(arr3_double vect, int nx) {
  for (int i = 0; i < nx; i++)
    vect.fetch(i,0,0) = -vect.get(i,0,0);
}
/** method to calculate the opposite of a vector */
void neg(double *vect, int n) 
{
    for (int i = 0; i < n; i++)
        vect[i] = -vect[i];
}

//* vect1(i, j, k) = vect2(i, j, k)
void eq(arr3_double vect1, const arr3_double vect2, int nx, int ny, int nz) 
{
    for (int i = 0; i < nx; i++)
        for (int j = 0; j < ny; j++)
            for (int k = 0; k < nz; k++)
                vect1.fetch(i, j, k) = vect2.get(i, j, k);
}

//* vect1(i, j, 0) = vect2(i, j, 0)
void eq(arr3_double vect1, const arr3_double vect2, int nx, int ny) 
{
    for (int i = 0; i < nx; i++)
        for (int j = 0; j < ny; j++)
            vect1.fetch(i, j, 0) = vect2.get(i, j, 0);
}

//* vect1(is, i, j, 0) = vect2(is, i, j, 0)
void eq(arr4_double vect1, const arr3_double vect2, int nx, int ny, int is)
{
    for (int i = 0; i < nx; i++)
        for (int j = 0; j < ny; j++)
            vect1.fetch(is, i, j, 0) = vect2.get(i, j, 0);
}

//* vect1(is, i, j, k) = vect2(is, i, j, k)
void eq(arr4_double vect1, const arr3_double vect2, int nx, int ny, int nz, int is) 
{
    for (int i = 0; i < nx; i++)
        for (int j = 0; j < ny; j++)
            for (int k = 0; k < nz; k++)
                vect1.fetch(is,i,j,k) = vect2.get(i,j,k);
}

//* Set a vector (arr3_double/arr4_double) to a value
void eqValue(double value, arr3_double vect, int nx, int ny, int nz) 
{
    for (int i = 0; i < nx; i++)
        for (int j = 0; j < ny; j++)
            for (int k = 0; k < nz; k++)
                vect.fetch(i, j, k) = value;
}

void eqValue(double value, arr3_double vect, int nx, int ny) 
{
    for (int i = 0; i < nx; i++)
        for (int j = 0; j < ny; j++)
            vect.fetch(i, j, 0) = value;
}

void eqValue(double value, arr3_double vect, int nx) 
{
    for (int i = 0; i < nx; i++)
        vect.fetch(i, 0, 0) = value;
}

void eqValue(double value, arr4_double vect, int ns, int nx, int ny, int nz) 
{
    for (int is = 0; is < ns; is++)
        for (int i = 0; i < nx; i++)
            for (int j = 0; j < ny; j++)
                for (int k = 0; k < nz; k++)
                    vect.fetch(is, i, j, k) = value;
}

void eqValue(double value, arr4_double vect, int ns, int nx, int ny) 
{
    for (int s = 0; s < ns; s++)
        for (int i = 0; i < nx; i++)
            for (int j = 0; j < ny; j++)
                vect.fetch(s, i, j, 0) = value;
}

void eqValue(double value, arr4_double vect, int ns, int nx) 
{
    for (int s = 0; s < ns; s++)
        for (int i = 0; i < nx; i++)
            vect.fetch(s, i, 0, 0) = value;
}

void eqValue(double value, double *vect, int n) 
{
    for (int i = 0; i < n; i++)
        vect[i] = value;
}

/** method to put a column in a matrix 2D */
void putColumn(double **Matrix, double *vect, int column, int n) {
  for (int i = 0; i < n; i++)
    Matrix[i][column] = vect[i];

}
/** method to get a column in a matrix 2D */
void getColumn(double *vect, double **Matrix, int column, int n) {
  for (int i = 0; i < n; i++)
    vect[i] = Matrix[i][column];
}
/** method to get rid of the ghost cells */
void getRidGhost(double **out, double **in, int nx, int ny) {
  for (int i = 1; i < nx - 1; i++)
    for (int j = 1; j < ny - 1; j++)
      out[i - 1][j - 1] = in[i][j];
}

void loopX(double *b, double z, double x, double y, double a, double zc, double xc, double yc, double m){

  double r = sqrt((x-xc)*(x-xc)+(y-yc)*(y-yc)+(z-zc)*(z-zc));
  double theta = acos((z-zc+1e-10)/(r+1e-10));
  double phi = atan2(y-yc,x-xc);
  //double Rho = r * sin(theta);
  double Rho = sqrt((x-xc)*(x-xc)+(y-yc)*(y-yc));

  double Alpha = Rho/a;
  double Beta = (z-zc)/a;
  double Gamma = (z-zc+1e-10)/(Rho+1e-10);

  double Q = ((1 + Alpha)*(1 + Alpha) + Beta*Beta);
  double k = sqrt(4*Alpha/Q);
  double B0 = m / (2*a); //m * (C_LIGHT * MU0)/(2*a*a*a*M_PI);

  int err = 0;

  double Bz = B0*(EllipticE(k,err)*(1-Alpha*Alpha-Beta*Beta)/(Q-4*Alpha)+EllipticF(k,err))/(M_PI*sqrt(Q));
  double BRho = B0*Gamma*(EllipticE(k,err)*(1+Alpha*Alpha+Beta*Beta)/(Q-4*Alpha)-EllipticF(k,err))/(M_PI*sqrt(Q));

  if (err)
    eprintf("Err came back :%d", err);

  if ( isnan(BRho) )
    BRho = 0;
  if ( isnan(Bz) )
    Bz = 0;

  double Bx = BRho * cos(phi);
  double By = BRho * sin(phi);

  //for debugging
  /*cout << "\n\nAt (" << x << "," << y << "," << z << "), the field is :" << endl;
    cout << "Bx: " << Bx << " T" << endl;
    cout << "By: " << By << " T" << endl;
    cout << "Bz: " << Bz << " T" << endl;
    cout << "BRho: " << BRho << " T" << endl;*/

  b[1] = Bx;
  b[2] = By;
  b[0] = Bz;
}

void loopY(double *b, double y, double z, double x, double a, double yc, double zc, double xc, double m){

  double r = sqrt((x-xc)*(x-xc)+(y-yc)*(y-yc)+(z-zc)*(z-zc));
  double theta = acos((z-zc+1e-10)/(r+1e-10));
  double phi = atan2(y-yc,x-xc);
  //double Rho = r * sin(theta);
  double Rho = sqrt((x-xc)*(x-xc)+(y-yc)*(y-yc));

  double Alpha = Rho/a;
  double Beta = (z-zc)/a;
  double Gamma = (z-zc+1e-10)/(Rho+1e-10);

  double Q = ((1 + Alpha)*(1 + Alpha) + Beta*Beta);
  double k = sqrt(4*Alpha/Q);
  double B0 = m / (2*a); //m * (C_LIGHT * MU0)/(2*a*a*a*M_PI);

  int err = 0;

  double Bz = B0*(EllipticE(k,err)*(1-Alpha*Alpha-Beta*Beta)/(Q-4*Alpha)+EllipticF(k,err))/(M_PI*sqrt(Q));
  double BRho = B0*Gamma*(EllipticE(k,err)*(1+Alpha*Alpha+Beta*Beta)/(Q-4*Alpha)-EllipticF(k,err))/(M_PI*sqrt(Q));

  if (err)
    eprintf("Err came back :%d", err);

  if ( isnan(BRho) )
    BRho = 0;
  if ( isnan(Bz) )
    Bz = 0;

  double Bx = BRho * cos(phi);
  double By = BRho * sin(phi);

  //for debugging
  /*cout << "\n\nAt (" << x << "," << y << "," << z << "), the field is :" << endl;
    cout << "Bx: " << Bx << " T" << endl;
    cout << "By: " << By << " T" << endl;
    cout << "Bz: " << Bz << " T" << endl;
    cout << "BRho: " << BRho << " T" << endl;*/

  b[2] = Bx;
  b[0] = By;
  b[1] = Bz;
}

void loopZ(double *b, double x, double y, double z, double a, double xc, double yc, double zc, double m){

  double r = sqrt((x-xc)*(x-xc)+(y-yc)*(y-yc)+(z-zc)*(z-zc));
  double theta = acos((z-zc+1e-10)/(r+1e-10));
  double phi = atan2(y-yc,x-xc);

  double Rho = sqrt((x-xc)*(x-xc)+(y-yc)*(y-yc));

  double Alpha = Rho/a;
  double Beta = (z-zc)/a;
  double Gamma = (z-zc+1e-10)/(Rho+1e-10);

  double Q = ((1 + Alpha)*(1 + Alpha) + Beta*Beta);
  double k = sqrt(4*Alpha/Q);
  double B0 = m / (2*a); //m * (C_LIGHT * MU0)/(2*a*a*a*M_PI);

  int err = 0;

  double Bz = B0*(EllipticE(k,err)*(1-Alpha*Alpha-Beta*Beta)/(Q-4*Alpha)+EllipticF(k,err))/(M_PI*sqrt(Q));
  double BRho = B0*Gamma*(EllipticE(k,err)*(1+Alpha*Alpha+Beta*Beta)/(Q-4*Alpha)-EllipticF(k,err))/(M_PI*sqrt(Q));

  if (err)
    eprintf("Err came back :%d", err);

  if ( isnan(BRho) )
    BRho = 0;
  if ( isnan(Bz) )
    Bz = 0;

  double Bx = BRho * cos(phi);
  double By = BRho * sin(phi);

  b[0] = Bx;
  b[1] = By;
  b[2] = Bz;
}

