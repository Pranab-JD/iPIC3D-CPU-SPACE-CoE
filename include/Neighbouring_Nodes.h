#ifndef NEIGHBOURINGNODES_H
#define NEIGHBOURINGNODES_H

//!  Class to handle which nodes have to be computed when the mass matrix is calculated !//

class NeighbouringNodes 
{
    public:
    NeighbouringNodes() 
    {
        i[0 ] = 0; j[0 ] = 0; k[0 ] = 0;  // i  ,j  ,k

        i[1 ] = 1; j[1 ] = 0; k[1 ] = 0;  // i+1,j  ,k      -> i-1,j  ,k
        i[2 ] = 0; j[2 ] = 1; k[2 ] = 0;  // i  ,j+1,k      -> i  ,j-1,k
        i[3 ] = 0; j[3 ] = 0; k[3 ] = 1;  // i  ,j  ,k+1    -> i  ,j  ,k-1

        i[4 ] = 1; j[4 ] = 1; k[4 ] = 0;  // i+1,j-1,k      ->i-1,j+1,k
        i[5 ] = 1; j[5 ] =-1; k[5 ] = 0;  // i+1,j-1,k      ->i-1,j+1,k
        i[6 ] = 1; j[6 ] = 0; k[6 ] = 1;  // i+1,j  ,k+1    ->i-1,j  ,k-1
        i[7 ] = 1; j[7 ] = 0; k[7 ] =-1;  // i+1,j  ,k-1    ->i-1,j  ,k+1
        i[8 ] = 0; j[8 ] = 1; k[8 ] = 1;  // i  ,j+1,k+1    ->i  ,j-1,k-1
        i[9 ] = 0; j[9 ] = 1; k[9 ] =-1;  // i  ,j+1,k-1    ->i  ,j-1,k+1

        i[10] = 1; j[10] =-1; k[10] = 1;  // i+1,j-1,k+1    ->i-1,j+1,k-1
        i[11] = 1; j[11] = 1; k[11] =-1;  // i+1,j+1,k-1    ->i-1,j-1,k+1
        i[12] =-1; j[12] = 1; k[12] = 1;  // i-1,j+1,k+1    ->i+1,j-1,k-1

        i[13] = 1; j[13] = 1; k[13] = 1;  // i+1,j+1,k+1    ->i+1,j+1,k+1
    };

    int getX(int ind) 
    { 
        if (ind <= 13) 
            return i[ind];
        else 
            return -i[ind-13];
    }
    
    int getY(int ind) 
    { 
        if (ind <= 13) 
            return j[ind];
        else 
            return -j[ind-13];
    }
    
    int getZ(int ind) 
    { 
        if (ind <= 13) 
            return k[ind];
        else 
            return -k[ind-13];
    }
        

    private:
    int i[14];
    int j[14];
    int k[14];
};

#endif
