/* 
 * iPIC3D was originally developed by Stefano Markidis and Giovanni Lapenta. 
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

/*!************************************************************************* EMfields3D.h - ElectroMagnetic fields definition ------------------- begin : May 2008 copyright : KU Leuven developers : Stefano Markidis, Giovanni Lapenta ************************************************************************* */

#ifndef EMfields3D_H
#define EMfields3D_H

#include <iostream>
#include "asserts.h"
#include "ipicfwd.h"
#include "Alloc.h"
#include "Basic.h"
#include "Neighbouring_Nodes.h"
#include <memory>
#include <iomanip>

using namespace std;
/*! Electromagnetic fields and sources defined for each local grid, and for an implicit maxwell's solver @date May 2008 @par Copyright: (C) 2008 KUL @author Stefano Markidis, Giovanni Lapenta. @version 3.0 */

// dimension of vectors used in fieldForPcls
const int DFIELD_3or4 = 4; // 4 pads with garbage but is needed for alignment

class Particles3Dcomm;
class Moments10;
class ECSIM_Moments13;
class EMfields3D                // :public Field
{
public:
    /*! constructor */
    EMfields3D(Collective * col, Grid * grid, VirtualTopology3D *vct);
    /*! destructor */
    ~EMfields3D();

    void setAllzero();

    //! ======================================================================================================= !//

    //? ---------- Initial field distributions (Non Relativistic) ---------- ?//

    //* Initialise electromagnetic fields with constant values
    void init();

    //* Initialise beam
    void initBEAM(double x_center, double y_center, double z_center, double radius);
    
    //* Initialise GEM challenge 
    void initGEM();

    void initOriginalGEM();
    
    //* Initialise double Harris sheets for magnetic reconnection
    void init_double_Harris();

    void init_double_Harris_hump();
    
    //* Initialise GEM challenge with dipole-like tail without perturbation
    void initGEMDipoleLikeTailNoPert();
    
    //* Initialise GEM challenge with no Perturbation
    void initGEMnoPert();

    //* Initialise from BATSRUS
    #ifdef BATSRUS
        void initBATSRUS();
    #endif

    //* Random initial fields
    void initRandomField();
    
    //* Initialise force free field (JxB=0)
    void initForceFree();
    
    //* Initialise rotated magnetic field
    void initEM_rotate(double B, double theta);
    
    //* Add a perturbation to charge density
    void AddPerturbationRho(double deltaBoB, double kx, double ky, double Bx_mod, double By_mod, double Bz_mod, double ne_mod, double ne_phase, double ni_mod, double ni_phase, double B0, Grid * grid);
    
    //* Add perturbation to the EM field
    void AddPerturbation(double deltaBoB, double kx, double ky, double Ex_mod, double Ex_phase, double Ey_mod, double Ey_phase, double Ez_mod, double Ez_phase, double Bx_mod, double Bx_phase, double By_mod, double By_phase, double Bz_mod, double Bz_phase, double B0, Grid * grid);
    
    //* Initialise a combination of magnetic dipoles
    void initDipole();
    void initDipole2D();
    
    //* Initialise magnetic nulls
    void initNullPoints();
    
    //* Initialise Taylor-Green flow
    void initTaylorGreen();

    //* Initialise fields for shear velocity in fluid finite Larmor radius (FLR) equilibrium (Cerri et al. 2013)
    void init_KHI_FLR(); 

    //? ---------- Initial particle distributions (Relativistic) ---------- ?//

    //? Quasi-1D ion-electron shock (Relativistic and Non relativistic)
    void initShock1D();

    //? Relativistic double Harris for pair plasma: Maxwellian background, drifting particles in the sheets
    void init_Relativistic_Double_Harris_pairs();

    //? Relativistic double Harris for ion-electron plasma: Maxwellian background, drifting particles in the sheets
    void init_Relativistic_Double_Harris_ion_electron();

    //! ======================================================================================================= !//

    /*! Calculate Electric field using the implicit Maxwell solver */
    void calculateE();
    /*! Image of Poisson Solver (for SOLVER) */
    void PoissonImage(double *image, double *vector);
    /*! Image of Maxwell Solver (for Solver) */
    void MaxwellImage(double *im, double *vector);
    /*! Maxwell source term (for SOLVER) */
    void MaxwellSource(double *bkrylov);
    /*! Impose a constant charge inside a spherical zone of the domain */
    void ConstantChargePlanet(double R, double x_center, double y_center, double z_center);
    void ConstantChargePlanet2DPlaneXZ(double R, double x_center, double z_center);
    /*! Impose a constant charge in the OpenBC boundaries */
    void ConstantChargeOpenBC();
    /*! Impose a constant charge in the OpenBC boundaries */
    void ConstantChargeOpenBCv2();
    /*! Calculate Magnetic field with the implicit solver: calculate B defined on nodes With E(n+ theta) computed, the magnetic field is evaluated from Faraday's law */
    void calculateB();

    //? ---------- Boundary Conditions ---------- ?//
    
    //* B boundary for GEM (cell centres) - this assumes non-periodic boundaries along Y *//
    void fixBcGEM();
    
    //* B boundary for GEM (nodes) - this assumes non-periodic boundaries along Y *//
    void fixBnGEM();
    
    //* B boundary for forcefree *//
    void fixBforcefree();

    //* Boundary conditions for magnetic field *//
    // void fixBC_B();

    //? ----------------------------------------- ?//

    /*! Calculate rho hat, Jx hat, Jy hat, Jz hat */
    void calculateHatFunctions();

    void C2NB();

    //* Compute the product of mass matrix with vector "V = (Vx, Vy, Vz)"
    void mass_matrix_times_vector(double* MEx, double* MEy, double* MEz, const_arr3_double vectX, const_arr3_double vectY, const_arr3_double vectZ, int i, int j, int k);

    //* Energy-conserving smoothing
    void energy_conserve_smooth(arr3_double data_X, arr3_double data_Y, arr3_double data_Z, int nx, int ny, int nz);
    void energy_conserve_smooth_direction(double*** data, int nx, int ny, int nz, int dir);

    /*! communicate ghost for densities and interp rho from node to center */
    void interpDensitiesN2C();
    /*! set to 0 all the densities fields */
    void setZeroDensities();
    /*! set to 0 primary moments */
    void setZeroPrimaryMoments();
    /*! set to 0 all densities derived from primary moments */
    void setZeroDerivedMoments();
    //! Set all elements of mass matrix to 0.0 !//
    void setZeroMassMatrix();
    /*! Sum rhon and J over species */
    void sumOverSpecies();
        /*! Sum rhon over species */
    // void sumOverSpeciesRho();
    /*! Sum current over different species */
    // void sumOverSpeciesJ();
    void sumOverSpecies_supplementary();
    void interpolateCenterSpecies(int is); 
    /*! Smoothing after the interpolation* */
    void smooth(arr3_double vector, int type);
    /*! SPECIES: Smoothing after the interpolation for species fields* */
    void smooth(double value, arr4_double vector, int is, int type);
    /*! smooth the electric field */
    void smoothE();

    /*! copy the field data to the array used to move the particles */
    void set_fieldForPcls();
    
    //* Communicate ghost cells for grid -> particles interpolation - IMM
    void communicateGhostP2G(int is);

    //* Communicate ghost cells for grid -> particles interpolation - ECSIM
    void communicateGhostP2G_ecsim(int is);
    void communicateGhostP2G_mass_matrix();

    //* Communicate ghost cells for grid -> particles interpolation - ECSIM output only
    void communicateGhostP2G_supplementary_moments(int is);

    /*! sum moments (interp_P2G) versions */
    void sumMoments(const Particles3Dcomm* part);
    void sumMoments_AoS(const Particles3Dcomm* part);
    void sumMoments_AoS_intr(const Particles3Dcomm* part);
    void sumMoments_vectorized(const Particles3Dcomm* part);
    void sumMoments_vectorized_AoS(const Particles3Dcomm* part);
    void sumMomentsOld(const Particles3Dcomm& pcls);
    /*! add accumulated moments to the moments for a given species */
    //void addToSpeciesMoments(const TenMoments & in, int is);

    //* ECSIM/RelSIM moments
    void add_Rho(double weight[8], int X, int Y, int Z, int is);
    
    void add_Jxh(double weight[8], int X, int Y, int Z, int is);
    void add_Jyh(double weight[8], int X, int Y, int Z, int is);
    void add_Jzh(double weight[8], int X, int Y, int Z, int is);

    void add_Mass(double value[3][3], int X, int Y, int Z, int ind);

    //* ECSIM/RelSIM supplementary moments
    void add_Jx(double weight[8], int X, int Y, int Z, int is);
    void add_Jy(double weight[8], int X, int Y, int Z, int is);
    void add_Jz(double weight[8], int X, int Y, int Z, int is);

    void add_N(double weight[8], int X, int Y, int Z, int is);

    void add_Pxx(double weight[8], int X, int Y, int Z, int is);
    void add_Pxy(double weight[8], int X, int Y, int Z, int is);
    void add_Pxz(double weight[8], int X, int Y, int Z, int is);
    void add_Pyy(double weight[8], int X, int Y, int Z, int is);
    void add_Pyz(double weight[8], int X, int Y, int Z, int is);
    void add_Pzz(double weight[8], int X, int Y, int Z, int is);

    void add_E_flux_x(double weight[8], int X, int Y, int Z, int is);
    void add_E_flux_y(double weight[8], int X, int Y, int Z, int is);
    void add_E_flux_z(double weight[8], int X, int Y, int Z, int is);

    void add_Qxxx(double weight[8], int X, int Y, int Z, int is);
    void add_Qyyy(double weight[8], int X, int Y, int Z, int is);
    void add_Qzzz(double weight[8], int X, int Y, int Z, int is);
    void add_Qxyz(double weight[8], int X, int Y, int Z, int is);
    void add_Qxxy(double weight[8], int X, int Y, int Z, int is);
    void add_Qxxz(double weight[8], int X, int Y, int Z, int is);
    void add_Qyyz(double weight[8], int X, int Y, int Z, int is);
    void add_Qxyy(double weight[8], int X, int Y, int Z, int is);
    void add_Qxzz(double weight[8], int X, int Y, int Z, int is);
    void add_Qyzz(double weight[8], int X, int Y, int Z, int is);
    
    /*! adjust densities on boundaries that are not periodic */
    void adjustNonPeriodicDensities(int is);

    /*! Perfect conductor boundary conditions LEFT wall */
    void perfectConductorLeft(arr3_double imageX, arr3_double imageY, arr3_double imageZ,
                              const_arr3_double vectorX, const_arr3_double vectorY, const_arr3_double vectorZ,
                              int dir);
    /*! Perfect conductor boundary conditions RIGHT wall */
    void perfectConductorRight(arr3_double imageX, arr3_double imageY, arr3_double imageZ,
                               const_arr3_double vectorX, const_arr3_double vectorY, const_arr3_double vectorZ,
                               int dir);
    /*! Perfect conductor boundary conditions for source LEFT wall */
    void perfectConductorLeftS(arr3_double vectorX, arr3_double vectorY, arr3_double vectorZ, int dir);
    /*! Perfect conductor boundary conditions for source RIGHT wall */
    void perfectConductorRightS(arr3_double vectorX, arr3_double vectorY, arr3_double vectorZ, int dir);

    /*! Calculate the sysceptibility tensor on the boundary */
    void sustensorRightX(double **susxx, double **susyx, double **suszx);
    void sustensorLeftX (double **susxx, double **susyx, double **suszx);
    void sustensorRightY(double **susxy, double **susyy, double **suszy);
    void sustensorLeftY (double **susxy, double **susyy, double **suszy);
    void sustensorRightZ(double **susxz, double **susyz, double **suszz);
    void sustensorLeftZ (double **susxz, double **susyz, double **suszz);

    //? Potential array
    arr3_double getPHI() {return PHI;}

    //* Field components
    const_arr4_pfloat get_fieldForPcls() { return fieldForPcls; }

    //? Electric Field (nodes)
    double getEx(int X, int Y, int Z) const { return Ex.get(X,Y,Z);  }
    double getEy(int X, int Y, int Z) const { return Ey.get(X,Y,Z);  }
    double getEz(int X, int Y, int Z) const { return Ez.get(X,Y,Z);  }
    arr3_double getEx() { return Ex;  }
    arr3_double getEy() { return Ey;  }
    arr3_double getEz() { return Ez;  }

    double getEx_ext(int X, int Y, int Z) const { return Ex_ext.get(X,Y,Z); }
    double getEy_ext(int X, int Y, int Z) const { return Ey_ext.get(X,Y,Z); }
    double getEz_ext(int X, int Y, int Z) const { return Ez_ext.get(X,Y,Z); }
    arr3_double getEx_ext() { return Ex_ext; }
    arr3_double getEy_ext() { return Ey_ext; }
    arr3_double getEz_ext() { return Ez_ext; }
    
    //? Magnetic field (nodes)
    double getBx(int X, int Y, int Z) const { return Bxn.get(X,Y,Z); }
    double getBy(int X, int Y, int Z) const { return Byn.get(X,Y,Z); }
    double getBz(int X, int Y, int Z) const { return Bzn.get(X,Y,Z); }
    arr3_double getBx() { return Bxn; }
    arr3_double getBy() { return Byn; }
    arr3_double getBz() { return Bzn; }

    double getBx_ext(int X, int Y, int Z) const { return Bx_ext.get(X,Y,Z); }
    double getBy_ext(int X, int Y, int Z) const { return By_ext.get(X,Y,Z); }
    double getBz_ext(int X, int Y, int Z) const { return Bz_ext.get(X,Y,Z); }
    arr3_double getBx_ext() { return Bx_ext; }
    arr3_double getBy_ext() { return By_ext; }
    arr3_double getBz_ext() { return Bz_ext; }

    double getBxc_ext(int X, int Y, int Z) const { return Bxc_ext.get(X,Y,Z); }
    double getByc_ext(int X, int Y, int Z) const { return Byc_ext.get(X,Y,Z); }
    double getBzc_ext(int X, int Y, int Z) const { return Bzc_ext.get(X,Y,Z); }
    arr3_double getBxc_ext() { return Bxc_ext; }
    arr3_double getByc_ext() { return Byc_ext; }
    arr3_double getBzc_ext() { return Bzc_ext; }

    double getBxc(int X, int Y, int Z) const { return Bxc.get(X,Y,Z); }
    double getByc(int X, int Y, int Z) const { return Byc.get(X,Y,Z); }
    double getBzc(int X, int Y, int Z) const { return Bzc.get(X,Y,Z); }
    arr3_double getBxc() { return Bxc; };
    arr3_double getByc() { return Byc; };
    arr3_double getBzc() { return Bzc; };

    double getBxTot(int X, int Y, int Z) const { return Bxn.get(X,Y,Z) + Bx_ext.get(X,Y,Z); }
    double getByTot(int X, int Y, int Z) const { return Byn.get(X,Y,Z) + By_ext.get(X,Y,Z); }
    double getBzTot(int X, int Y, int Z) const { return Bzn.get(X,Y,Z) + Bz_ext.get(X,Y,Z); }
    arr3_double getBxTot() { addscale(1.0,Bxn,Bx_ext,Bx_tot,nxn,nyn,nzn); return Bx_tot; }
    arr3_double getByTot() { addscale(1.0,Byn,By_ext,By_tot,nxn,nyn,nzn); return By_tot; }
    arr3_double getBzTot() { addscale(1.0,Bzn,Bz_ext,Bz_tot,nxn,nyn,nzn); return Bz_tot; }

    //* Densities (s --> of each species)
    arr3_double getRHOn() { return rhon; }
    double getRHOn(int X, int Y, int Z) const { return rhon.get(X, Y, Z); }
    
    arr4_double getRHOns() { return rhons; }
    double getRHOns(int X, int Y, int Z, int is) const { return rhons.get(is, X, Y, Z); }

    arr4_double getRHOcs() { return rhocs; }
    
    double getRHOcs(int X, int Y, int Z, int is) const { return rhocs.get(is, X, Y, Z); }
    arr3_double getRHOc_avg() { return rhoc_avg; }
    double getRHOc_avg(int X, int Y, int Z) const { return rhoc_avg.get(X, Y, Z); }

    //* Current (s --> of each species)
    double getJxs(int X,int Y,int Z,int is) const { return Jxs.get(is,X,Y,Z); }
    double getJys(int X,int Y,int Z,int is) const { return Jys.get(is,X,Y,Z); }
    double getJzs(int X,int Y,int Z,int is) const { return Jzs.get(is,X,Y,Z); }
    arr4_double getJxs() { return Jxs; }
    arr4_double getJys() { return Jys; }
    arr4_double getJzs() { return Jzs; }

    //* Current (overall)
    arr3_double getJx() { return Jx; }
    arr3_double getJy() { return Jy; }
    arr3_double getJz() { return Jz; }
    double getJx(int X,int Y,int Z) const { return Jx.get(X,Y,Z); }
    double getJy(int X,int Y,int Z) const { return Jy.get(X,Y,Z); }
    double getJz(int X,int Y,int Z) const { return Jz.get(X,Y,Z); }

    arr3_double getJxh() { return Jxh; }
    arr3_double getJyh() { return Jyh; }
    arr3_double getJzh() { return Jzh; }

    arr3_double getJx_ext() { return Jx_ext; }
    arr3_double getJy_ext() { return Jy_ext; }
    arr3_double getJz_ext() { return Jz_ext; }

    //* Pressure Tensor
    arr4_double getpXXsn() { return pXXsn; }
    arr4_double getpXYsn() { return pXYsn; }
    arr4_double getpXZsn() { return pXZsn; }
    arr4_double getpYYsn() { return pYYsn; }
    arr4_double getpYZsn() { return pYZsn; }
    arr4_double getpZZsn() { return pZZsn; }

    double getpXXsn(int X, int Y, int Z, int is) const { return pXXsn.get(is,X,Y,Z); }
    double getpXYsn(int X, int Y, int Z, int is) const { return pXYsn.get(is,X,Y,Z); }
    double getpXZsn(int X, int Y, int Z, int is) const { return pXZsn.get(is,X,Y,Z); }
    double getpYYsn(int X, int Y, int Z, int is) const { return pYYsn.get(is,X,Y,Z); }
    double getpYZsn(int X, int Y, int Z, int is) const { return pYZsn.get(is,X,Y,Z); }
    double getpZZsn(int X, int Y, int Z, int is) const { return pZZsn.get(is,X,Y,Z); }

    // double getJx(int X, int Y, int Z) const { return Jx.get(X,Y,Z); }
    // double getJy(int X, int Y, int Z) const { return Jy.get(X,Y,Z); }
    // double getJz(int X, int Y, int Z) const { return Jz.get(X,Y,Z); }

    //* Energy Flux Density
    arr4_double getEFxs() { return E_flux_xs; }
    arr4_double getEFys() { return E_flux_ys; }
    arr4_double getEFzs() { return E_flux_zs; }

    double getEFxs(int X, int Y, int Z, int is) const { return E_flux_xs.get(is,X,Y,Z); }
    double getEFys(int X, int Y, int Z, int is) const { return E_flux_ys.get(is,X,Y,Z); }
    double getEFzs(int X, int Y, int Z, int is) const { return E_flux_zs.get(is,X,Y,Z); }
    

    //* Heat Flux Tensor 
    double ****getQxxxs() { return (Qxxxs); }
    double ****getQyyys() { return (Qyyys); }
    double ****getQzzzs() { return (Qzzzs); }
    double ****getQxyzs() { return (Qxyzs); }
    double ****getQxxys() { return (Qxxys); }
    double ****getQxxzs() { return (Qxxzs); }
    double ****getQxyys() { return (Qxyys); }
    double ****getQxzzs() { return (Qxzzs); }
    double ****getQyzzs() { return (Qyzzs); }
    double ****getQyyzs() { return (Qyyzs); }


    //* Divergences
    arr3_double getDivE() { return divE; }
    arr3_double getDivB() { return divB; }
    arr3_double getDivAverage() { return divE_average; }

    //* Residual of DivE on cell centers *//
    double getResDiv(int X, int Y, int Z, int is) const { return residual_divergence.get(is, X, Y, Z); }

    void divergence_E(double ma);
    void divergence_B();
    void timeAveragedRho(double ma);
    void timeAveragedDivE(double ma);

    double get_E_field_energy();        //* Electric field energy
    double get_Ex_field_energy();       //* Electric field energy along X
    double get_Ey_field_energy();       //* Electric field energy along Y
    double get_Ez_field_energy();       //* Electric field energy along Z
    double get_B_field_energy();        //* Magnetic (internal) field energy along X
    double get_Bx_field_energy();       //* Magnetic (internal) field energy along Y
    double get_By_field_energy();       //* Magnetic (internal) field energy along Z
    double get_Bz_field_energy();       //* Magnetic (internal) field energy
    double get_Bext_energy();           //* External magnetic field energy
    double get_bulk_energy(int is);       //* Bulk kinetic energy

    void setZeroCurrent();
    void setZeroRho();

    double LOG_COSH(double x) 
    {
        double res;
        if (fabs(x) > 18.5) 
            res = fabs(x) - log(2.0);
        else 
            res = log(cosh(x));
      
        return res;
    }

    /*! fetch array for summing moments of thread i */
    Moments10& fetch_moments10Array(int i)
    {
        assert_le(0,i);
        assert_lt(i,sizeMomentsArray);
        return *(moments10Array[i]);
    }

    ECSIM_Moments13& fetch_moments13Array(int i)
    {
        assert_le(0, i);
        assert_lt(i, sizeMomentsArray);
        return *(ecsim_moments13Array[i]);
    }

    int get_sizeMomentsArray() { return sizeMomentsArray; }

    /*! print electromagnetic fields info */
    void print(void) const;
    
    //get MPI Derived Datatype
    MPI_Datatype getYZFacetype(bool isCenterFlag){return isCenterFlag ?yzFacetypeC : yzFacetypeN;}
    MPI_Datatype getXZFacetype(bool isCenterFlag){return isCenterFlag ?xzFacetypeC : xzFacetypeN;}
    MPI_Datatype getXYFacetype(bool isCenterFlag){return isCenterFlag ?xyFacetypeC : xyFacetypeN;}
    MPI_Datatype getXEdgetype(bool isCenterFlag){return  isCenterFlag ?xEdgetypeC : xEdgetypeN;}
    MPI_Datatype getYEdgetype(bool isCenterFlag){return  isCenterFlag ?yEdgetypeC : yEdgetypeN;}
    MPI_Datatype getZEdgetype(bool isCenterFlag){return  isCenterFlag ?zEdgetypeC : zEdgetypeN;}
    MPI_Datatype getXEdgetype2(bool isCenterFlag){return  isCenterFlag ?xEdgetypeC2 : xEdgetypeN2;}
    MPI_Datatype getYEdgetype2(bool isCenterFlag){return  isCenterFlag ?yEdgetypeC2 : yEdgetypeN2;}
    MPI_Datatype getZEdgetype2(bool isCenterFlag){return  isCenterFlag ?zEdgetypeC2 : zEdgetypeN2;}
    MPI_Datatype getCornertype(bool isCenterFlag){return  isCenterFlag ?cornertypeC : cornertypeN;}

    MPI_Datatype getProcview(){return  procview;}
    MPI_Datatype getXYZeType(){return xyzcomp;}
    MPI_Datatype getProcviewXYZ(){return  procviewXYZ;}
    MPI_Datatype getGhostType(){return  ghosttype;}

    void freeDataType();
    bool isLittleEndian(){return lEndFlag;};

public: // accessors
    const Collective& get_col()const{return _col;}
    const Grid& get_grid()const{return _grid;};
    const VirtualTopology3D& get_vct()const{return _vct;}

    //* ********************************* VARIABLES ********************************* *//
    
private:
    //? access to global data
    const Collective& _col;
    const Grid& _grid;
    const VirtualTopology3D&_vct;
    
    /*! light speed */
    double c;
    /* 4*PI for normalization */
    double FourPI;
    /*! time step */
    double dt;
    /*! decentering parameter */
    double th;
    
    /*! Smoothing value */
    bool Smooth;
    int smooth_cycle;         // Frequency of smoothing (after every "smooth_cycle" time cycles)
    int num_smoothings;       // Number of times of smoothing at a given time cycle

    //* Custom input parameters
    double *input_param; int nparam;
    
    int zeroCurrent; double delt; int ns; double delta;
    
    //* Magnetic field
    double B0x, B0y, B0z, B1x, B1y, B1z;
    
    //* Charge to mass ratio
    double *qom;
    
    //* Boundary electron speed
    double ue0, ve0, we0;

    //! Mass matrix
    double *mass_matrix;

    //* Number of cells including 2 (guard cells)
    int nxc, nxn, nyc, nyn, nzc, nzn;

    //* Local grid boundaries coordinate
    double xStart, xEnd, yStart, yEnd, zStart, zEnd;

    //* Grid spacing
    double dx, dy, dz, invVOL;
    
    //* Simulation box length
    double Lx, Ly, Lz;
    
    //* Source
    double x_center, y_center, z_center, L_square;

    //? Electric field component used to move particles organized for rapid access
    //! This is the information transferred from cluster to booster
    array4_pfloat fieldForPcls;

    //? Electric field (defined at nodes)
    array3_double Ex, Ey, Ez;

    //? Implicit electric field (defined at nodes)
    array3_double Exth, Eyth, Ezth;

    //? Magnetic field (defined at nodes)
    array3_double Bxn, Byn, Bzn;

    //? Magnetic field (defined at cell centres)
    array3_double Bxc, Byc, Bzc;

    //? Current density (defined at nodes)
    array3_double Jx, Jy, Jz;

    //? Implicit current density 
    array3_double Jxh, Jyh, Jzh;

    //? External magnetic field (defined at cell centres)
    array3_double Bxc_ext, Byc_ext, Bzc_ext;

    //? External magnetic field (defined at nodes)
    array3_double Bx_ext, By_ext, Bz_ext;

    //? Total magnetic field (defined at nodes)
    array3_double Bx_tot, By_tot, Bz_tot;

    //? External current (defined at nodes)
    array3_double Jx_ext, Jy_ext, Jz_ext;

    //? External electric field (defined at nodes)
    array3_double Ex_ext, Ey_ext, Ez_ext;

    //! Mass matrix components (defined at nodes)
    array4_double Mxx, Mxy, Mxz, Myx, Myy, Myz, Mzx, Mzy, Mzz;

    //? Density for each species (defined at nodes and centres, respectively)
    array4_double rhons, rhocs;         
    
    array3_double rhoc_avg;             //* Time averaged density (defined at cell centres)    
    array4_double rhocs_avg;            //* Time averaged density for each species (defined at cell centres)
    
    //? Current densities for each species (defined at nodes)
    array4_double Jxs, Jys, Jzs, Jxhs, Jyhs, Jzhs;
    
    //! Supplementary moments
    //? Energy flux for each species (defined at nodes)
    array4_double E_flux_xs, E_flux_ys, E_flux_zs;

    //? Number of particles for each species (defined at nodes)
    array4_double Nns;

    bool SaveHeatFluxTensor;

    //? Heat Flux Tensor (defined at nodes)
    double**** Qxxxs; double**** Qyyys; double**** Qzzzs;  
    double**** Qxxys; double**** Qxzzs; double**** Qxyys; double**** Qxyzs; 
    double**** Qyzzs; double**** Qyyzs; double**** Qxxzs;

    //? Pressure Tensor (defined at nodes)
    array4_double pXXsn, pXYsn, pXZsn, pYYsn, pYZsn, pZZsn;

    //? Density (defined at nodes and centres, respectively)
    array3_double rhon, rhoc;            
    
    //? Implicit density (defined at cell centres)
    // array3_double rhoh;

    //? Electric potential (defined at cell centres)
    array3_double PHI, Phic;  

    //? Used in computing divergence
    array3_double divC, divB, divE, divE_average;
    array4_double residual_divergence;

    //? ***************** TEMPORARY ARRAYS ***************** ?//

    //* Used in MaxwellSource()
    array3_double tempC, tempXC, tempYC, tempZC, tempXC2, tempYC2, tempZC2;
    array3_double tempX, tempY, tempZ, temp2X, temp2Y, temp2Z, temp3X, temp3Y, temp3Z, smooth_temp;
    array3_double tempXN, tempYN, tempZN;

    //* Temporary arrays for MaxwellImage() *//
    array3_double imageX, imageY, imageZ, Dx, Dy, Dz, vectX, vectY, vectZ;

    //* Arrays for summing moments *//
    int sizeMomentsArray;
    Moments10 **moments10Array;
    ECSIM_Moments13 **ecsim_moments13Array;

    //* Object of class to handle which nodes have to be computed when the mass matrix is calculated
    NeighbouringNodes NeNo;

    /*! Field Boundary Condition
      0 = Dirichlet Boundary Condition: specifies the
          value on the boundary of the domain
      1 = Neumann Boundary Condition: specifies the value of
          derivative on the boundary of the domain
      2 = Periodic boundary condition */

    // boundary conditions for electrostatic potential
    //
    int bcPHIfaceXright;
    int bcPHIfaceXleft;
    int bcPHIfaceYright;
    int bcPHIfaceYleft;
    int bcPHIfaceZright;
    int bcPHIfaceZleft;

    /*! Boundary condition for electric field 0 = perfect conductor 1 = magnetic mirror */
    // boundary conditions for EM field
    int bcEMfaceXright;
    int bcEMfaceXleft;
    int bcEMfaceYright;
    int bcEMfaceYleft;
    int bcEMfaceZright;
    int bcEMfaceZleft;

    /*! GEM Challenge background ion */
    double *rhoINIT;
    /*! Drift of the species */
    bool *DriftSpecies;
    
    int restart_status;

    double GMREStol;                            //* GMRES tolerance criterium for stopping iterations

    double Fext;
    double Fzro;
    bool Fpext;
    
    double u_bulk, v_bulk, w_bulk;              //* Bulk velocity along X, Y, Z, respectively for the Lagrangian reference frame

    //MPI Derived Datatype for Center Halo Exchange
    MPI_Datatype yzFacetypeC;
    MPI_Datatype xzFacetypeC;
    MPI_Datatype xyFacetypeC;
    MPI_Datatype xEdgetypeC;
    MPI_Datatype yEdgetypeC;
    MPI_Datatype zEdgetypeC;
    MPI_Datatype xEdgetypeC2;
    MPI_Datatype yEdgetypeC2;
    MPI_Datatype zEdgetypeC2;
    MPI_Datatype cornertypeC;

    //MPI Derived Datatype for Node Halo Exchange
    MPI_Datatype yzFacetypeN;
    MPI_Datatype xzFacetypeN;
    MPI_Datatype xyFacetypeN;
    MPI_Datatype xEdgetypeN;
    MPI_Datatype yEdgetypeN;
    MPI_Datatype zEdgetypeN;
    MPI_Datatype xEdgetypeN2;
    MPI_Datatype yEdgetypeN2;
    MPI_Datatype zEdgetypeN2;
    MPI_Datatype cornertypeN;
    
    //for VTK output
    MPI_Datatype  procviewXYZ,xyzcomp,procview,ghosttype;
    bool lEndFlag;
    
    void OpenBoundaryInflowB(arr3_double vectorX, arr3_double vectorY, arr3_double vectorZ, int nx, int ny, int nz);
    void OpenBoundaryInflowE(arr3_double vectorX, arr3_double vectorY, arr3_double vectorZ, int nx, int ny, int nz);
    void OpenBoundaryInflowEImage(arr3_double imageX, arr3_double imageY, arr3_double imageZ, 
                                 const_arr3_double vectorX, const_arr3_double vectorY, const_arr3_double vectorZ,
                                 int nx, int ny, int nz);
};

//* Add an amount of charge density to charge density field at node X,Y,Z
inline void EMfields3D::add_Rho(double weight[8], int X, int Y, int Z, int is) 
{
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            for (int k = 0; k < 2; k++)
                rhons[is][X - i][Y - j][Z - k] += weight[i * 4 + j * 2 + k] * invVOL;
}

//* Add an amount of current density to current density field at node X,Y,Z
inline void EMfields3D::add_Jxh(double weight[8], int X, int Y, int Z, int is)
{
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            for (int k = 0; k < 2; k++)
                Jxhs[is][X - i][Y - j][Z - k] += weight[i * 4 + j * 2 + k];
}
inline void EMfields3D::add_Jyh(double weight[8], int X, int Y, int Z, int is)
{
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            for (int k = 0; k < 2; k++)
                Jyhs[is][X - i][Y - j][Z - k] += weight[i * 4 + j * 2 + k];
}
inline void EMfields3D::add_Jzh(double weight[8], int X, int Y, int Z, int is)
{
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            for (int k = 0; k < 2; k++)
                Jzhs[is][X - i][Y - j][Z - k] += weight[i * 4 + j * 2 + k];
}

inline void EMfields3D::add_Jx(double weight[8], int X, int Y, int Z, int is) 
{
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            for (int k = 0; k < 2; k++) 
                Jxs[is][X - i][Y - j][Z - k] += weight[i * 4 + j * 2 + k] * invVOL;
}

inline void EMfields3D::add_Jy(double weight[8], int X, int Y, int Z, int is) 
{
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            for (int k = 0; k < 2; k++) 
                Jys[is][X - i][Y - j][Z - k] += weight[i * 4 + j * 2 + k] * invVOL;
}

inline void EMfields3D::add_Jz(double weight[8], int X, int Y, int Z, int is) 
{
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            for (int k = 0; k < 2; k++) 
                Jzs[is][X - i][Y - j][Z - k] += weight[i * 4 + j * 2 + k] * invVOL;
}

inline void EMfields3D::add_N(double weight[8], int X, int Y, int Z, int is) 
{
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            for (int k = 0; k < 2; k++)
                Nns[is][X - i][Y - j][Z - k] += weight[i * 4 + j * 2 + k];
}

inline void EMfields3D::add_Pxx(double weight[8], int X, int Y, int Z, int is) 
{
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            for (int k = 0; k < 2; k++)
                pXXsn[is][X - i][Y - j][Z - k] += weight[i * 4 + j * 2 + k] * invVOL;
}

inline void EMfields3D::add_Pxy(double weight[8], int X, int Y, int Z, int is) 
{
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            for (int k = 0; k < 2; k++)
                pXYsn[is][X - i][Y - j][Z - k] += weight[i * 4 + j * 2 + k] * invVOL;
}

inline void EMfields3D::add_Pxz(double weight[8], int X, int Y, int Z, int is) 
{
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            for (int k = 0; k < 2; k++)
                pXZsn[is][X - i][Y - j][Z - k] += weight[i * 4 + j * 2 + k] * invVOL;
}

inline void EMfields3D::add_Pyy(double weight[8], int X, int Y, int Z, int is) 
{
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            for (int k = 0; k < 2; k++)
                pYYsn[is][X - i][Y - j][Z - k] += weight[i * 4 + j * 2 + k] * invVOL;
}

inline void EMfields3D::add_Pyz(double weight[8], int X, int Y, int Z, int is)
{
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            for (int k = 0; k < 2; k++)
                pYZsn[is][X - i][Y - j][Z - k] += weight[i * 4 + j * 2 + k] * invVOL;
}

inline void EMfields3D::add_Pzz(double weight[8], int X, int Y, int Z, int is) 
{
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            for (int k = 0; k < 2; k++)
                pZZsn[is][X - i][Y - j][Z - k] += weight[i * 4 + j * 2 + k] * invVOL;
}

inline void EMfields3D::add_E_flux_x(double weight[8], int X, int Y, int Z, int is) 
{
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            for (int k = 0; k < 2; k++) 
                E_flux_xs[is][X - i][Y - j][Z - k] += weight[i * 4 + j * 2 + k] * invVOL;
}

inline void EMfields3D::add_E_flux_y(double weight[8], int X, int Y, int Z, int is) 
{
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            for (int k = 0; k < 2; k++) 
                E_flux_ys[is][X - i][Y - j][Z - k] += weight[i * 4 + j * 2 + k] * invVOL;
}

inline void EMfields3D::add_E_flux_z(double weight[8], int X, int Y, int Z, int is) 
{
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            for (int k = 0; k < 2; k++) 
                E_flux_zs[is][X - i][Y - j][Z - k] += weight[i * 4 + j * 2 + k] * invVOL;
}

inline void EMfields3D::add_Qxxx(double weight[8], int X, int Y, int Z, int is) 
{
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            for (int k = 0; k < 2; k++) 
                Qxxxs[is][X - i][Y - j][Z - k] += weight[i * 4 + j * 2 + k] * invVOL;
}

inline void EMfields3D::add_Qyyy(double weight[8], int X, int Y, int Z, int is) 
{
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            for (int k = 0; k < 2; k++) 
                Qyyys[is][X - i][Y - j][Z - k] += weight[i * 4 + j * 2 + k] * invVOL;
}

inline void EMfields3D::add_Qzzz(double weight[8], int X, int Y, int Z, int is) 
{
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            for (int k = 0; k < 2; k++) 
                Qzzzs[is][X - i][Y - j][Z - k] += weight[i * 4 + j * 2 + k] * invVOL;
}

inline void EMfields3D::add_Qxyz(double weight[8], int X, int Y, int Z, int is) 
{
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            for (int k = 0; k < 2; k++) 
                Qxyzs[is][X - i][Y - j][Z - k] += weight[i * 4 + j * 2 + k] * invVOL;
}

inline void EMfields3D::add_Qxxy(double weight[8], int X, int Y, int Z, int is) 
{
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            for (int k = 0; k < 2; k++) 
                Qxxys[is][X - i][Y - j][Z - k] += weight[i * 4 + j * 2 + k] * invVOL;
}

inline void EMfields3D::add_Qxxz(double weight[8], int X, int Y, int Z, int is) 
{
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            for (int k = 0; k < 2; k++) 
                Qxxzs[is][X - i][Y - j][Z - k] += weight[i * 4 + j * 2 + k] * invVOL;
}

inline void EMfields3D::add_Qxyy(double weight[8], int X, int Y, int Z, int is) 
{
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            for (int k = 0; k < 2; k++) 
                Qxyys[is][X - i][Y - j][Z - k] += weight[i * 4 + j * 2 + k] * invVOL;
}

inline void EMfields3D::add_Qxzz(double weight[8], int X, int Y, int Z, int is) 
{
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            for (int k = 0; k < 2; k++) 
                Qxzzs[is][X - i][Y - j][Z - k] += weight[i * 4 + j * 2 + k] * invVOL;
}

inline void EMfields3D::add_Qyzz(double weight[8], int X, int Y, int Z, int is) 
{
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            for (int k = 0; k < 2; k++) 
                Qyzzs[is][X - i][Y - j][Z - k] += weight[i * 4 + j * 2 + k] * invVOL;
}

inline void EMfields3D::add_Qyyz(double weight[8], int X, int Y, int Z, int is) 
{
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            for (int k = 0; k < 2; k++) 
                Qyyzs[is][X - i][Y - j][Z - k] += weight[i * 4 + j * 2 + k] * invVOL;
}

//* Add an amount of current density to mass matrix field at node X,Y *//
inline void EMfields3D::add_Mass(double value[3][3], int X, int Y, int Z, int ind) 
{
    Mxx[ind][X][Y][Z] += value[0][0];
    Mxy[ind][X][Y][Z] += value[0][1];
    Mxz[ind][X][Y][Z] += value[0][2];
    Myx[ind][X][Y][Z] += value[1][0];
    Myy[ind][X][Y][Z] += value[1][1];
    Myz[ind][X][Y][Z] += value[1][2];
    Mzx[ind][X][Y][Z] += value[2][0];
    Mzy[ind][X][Y][Z] += value[2][1];
    Mzz[ind][X][Y][Z] += value[2][2];
}

inline void get_field_components_for_cell(const double* field_components[8], const_arr4_double fieldForPcls, int cx, int cy, int cz)
{
    // interface to the right of cell
    const int ix = cx + 1;
    const int iy = cy + 1;
    const int iz = cz + 1;

    arr3_double_get field0 = fieldForPcls[ix];
    arr3_double_get field1 = fieldForPcls[cx];
    arr2_double_get field00 = field0[iy];
    arr2_double_get field01 = field0[cy];
    arr2_double_get field10 = field1[iy];
    arr2_double_get field11 = field1[cy];
    field_components[0] = field00[iz]; // field000 
    field_components[1] = field00[cz]; // field001 
    field_components[2] = field01[iz]; // field010 
    field_components[3] = field01[cz]; // field011 
    field_components[4] = field10[iz]; // field100 
    field_components[5] = field10[cz]; // field101 
    field_components[6] = field11[iz]; // field110 
    field_components[7] = field11[cz]; // field111 
}

typedef EMfields3D Field;

#endif
