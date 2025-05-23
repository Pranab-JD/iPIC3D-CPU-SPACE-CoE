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
#include "GMRES.h"
#include "Basic.h"
#include "parallel.h"
#include "errors.h"
#include "Alloc.h"
#include "TimeTasks.h"
#include "EMfields3D.h"
#include "VCtopology3D.h"

using namespace std;

void GMRES(FIELD_IMAGE FunctionImage, double *xkrylov, int xkrylovlen,  const double *b, int m, int max_iter, double tol, Field * field)
{
    if (m > xkrylovlen) 
    {
        // m need not be the same for all processes, we cannot restrict this test to the main process,
        // (although we could probably restrict it to the process with the highest cartesian rank).
        eprintf("In GMRES the dimension of Krylov space(m) can't be > (length of krylov vector)/(# processors)\n");
    }

    bool GMRESVERBOSE = false;
    double initial_error, rho_tol;
    
    double *r = new double[xkrylovlen];
    double *im = new double[xkrylovlen];

    double *s = new double[m + 1];
    double *cs = new double[m + 1];
    double *sn = new double[m + 1];
    double *y = new double[m + 3];
    
    eqValue(0.0, s, m + 1);
    eqValue(0.0, cs, m + 1);
    eqValue(0.0, sn, m + 1);
    eqValue(0.0, y, m + 3);

    //* allocate H (Hessenberg matrix) for storing the results from decomposition
    double **H = newArr2(double, m + 1, m);
    for (int ii = 0; ii < m + 1; ii++)
        for (int jj = 0; jj < m; jj++)
            H[ii][jj] = 0;

    //* allocate V
    double **V = newArr2(double, m+1, xkrylovlen);
    for (int ii = 0; ii < m+1; ii++)
        for (int jj = 0; jj < xkrylovlen; jj++)
            V[ii][jj] = 0;

    if (GMRESVERBOSE && is_output_thread()) 
    {
        printf( "------------------------------------\n"
                "-             GMRES                -\n"
                "------------------------------------\n\n");
    }

    MPI_Comm fieldcomm = (field->get_vct()).getFieldComm();

    double normb = normP(b, xkrylovlen, &fieldcomm);
    if (normb == 0.0) normb = 1.0;

    //? GMRes iterations
    int itr = 0;
    for (itr = 0; itr < max_iter; itr++)
    {
        //* r = b - A*x
        (field->*FunctionImage) (im, xkrylov);
        sub(r, b, im, xkrylovlen);
        initial_error = normP(r, xkrylovlen, &fieldcomm);

        if (itr == 0) 
        {
            // if (is_output_thread())
                // cout << "Initial residual = " << initial_error << "; norm b vector (source) = " << normb << endl;

            rho_tol = initial_error * tol;

            if ((initial_error / normb) <= tol) 
            {
                if (is_output_thread())
                    printf("GMRES converged without iterations: initial error < tolerance\n");
                break;
            }
        }

        scale(V[0], r, (1.0 / initial_error), xkrylovlen);
        eqValue(0.0, s, m + 1);
        s[0] = initial_error;
        int k = 0;

        while (rho_tol < initial_error && k < m)
        {
            // w= A*V(:,k)
            double *w = V[k+1];
            (field->*FunctionImage) (w, V[k]);    

            // new code to make a single MPI_Allreduce call
            for (int j = 0; j <= k; j++)
                y[j] = dot(w, V[j], xkrylovlen);

            y[k+1] = norm2(w, xkrylovlen);
            
            MPI_Allreduce(MPI_IN_PLACE, y, (k+2), MPI_DOUBLE, MPI_SUM, fieldcomm);

            for (int j = 0; j <= k; j++)
            {
                H[j][k] = y[j];
                addscale(-H[j][k], V[k+1], V[j], xkrylovlen);
            }

            H[k+1][k] = normP(V[k+1], xkrylovlen,&fieldcomm);

            double av = sqrt(y[k+1]);
            // why are we testing floating point numbers
            // for equality?  Is this supposed to say
            //if (av < delta * fabs(H[k + 1][k]))
            const double delta=0.001;
            if (av + delta * H[k + 1][k] == av)
            {
                for (int j = 0; j <= k; j++) 
                {
                    const double htmp = dotP(w, V[j], xkrylovlen,&fieldcomm);
                    H[j][k] = H[j][k] + htmp;
                    addscale(-htmp, w, V[j], xkrylovlen);
                }
                H[k + 1][k] = normP(w, xkrylovlen,&fieldcomm);
            }

            // normalize the new vector
            scale(w, (1.0 / H[k + 1][k]), xkrylovlen);

            if (0 < k) 
            {
                for (int j = 0; j < k; j++)
                    ApplyPlaneRotation(H[j + 1][k], H[j][k], cs[j], sn[j]);

                getColumn(y, H, k, m + 1);
            }

            const double mu = sqrt(H[k][k] * H[k][k] + H[k + 1][k] * H[k + 1][k]);
            cs[k] = H[k][k] / mu;
            sn[k] = -H[k + 1][k] / mu;
            H[k][k] = cs[k] * H[k][k] - sn[k] * H[k + 1][k];
            H[k + 1][k] = 0.0;

            ApplyPlaneRotation(s[k + 1], s[k], cs[k], sn[k]);
            initial_error = fabs(s[k]);
            k++;
        }

        k--;
        y[k] = s[k] / H[k][k];

        for (int i = k - 1; i >= 0; i--) 
        {
            double tmp = 0.0;
            for (int l = i + 1; l <= k; l++)
                tmp += H[i][l] * y[l];
            
            y[i] = (s[i] - tmp) / H[i][i];
        }

        for (int j = 0; j < k; j++)
        {
            const double yj = y[j];
            double* Vj = V[j];
            
            for (int i = 0; i < xkrylovlen; i++)
                xkrylov[i] += yj * Vj[i];
        }

        if (initial_error <= rho_tol) 
        {
            if (is_output_thread())
                printf("GMRES converged at restart %d; iteration %d with error: %g\n", itr, k,  initial_error / rho_tol * tol);
            
            break;
        }

        if (is_output_thread() && GMRESVERBOSE)
            printf("Restart: %d error: %g\n", itr,  initial_error / rho_tol * tol);

    }

    if(itr == max_iter && is_output_thread())
        printf("GMRES not converged !! Final error: %g\n", initial_error / rho_tol * tol);

    delete[]r;
    delete[]im;
    delete[]s;
    delete[]cs;
    delete[]sn;
    delete[]y;
    delArr2(H, m + 1);
    delArr2(V, m + 1);
    return;
}

void ApplyPlaneRotation(double &dx, double &dy, double &cs, double &sn) 
{
    double temp = cs * dx + sn * dy;
    dy = -sn * dx + cs * dy;
    dx = temp;
}
