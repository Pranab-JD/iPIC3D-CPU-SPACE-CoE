//* This file needs to be included in main/iPIC3Dlib.cpp
//*
//* The functions used to define initial particles and field config need to be 
//* included in 'include/Particles3D.h' and 'include/EMfields3D.h', respectively.

#pragma once

#include "../include.hpp"

#ifndef NO_HDF5
#endif

using std::cout;
using std::endl;
using namespace iPic3D;

//TODO: Only 2D turbulence can be seeded (for now); 3D turbulence to be included soon

//? ========================================================================== ?//

//! Particles are initialised with a uniform spatial distribution and a Maxellian velocity distribution

//? ========================================================================== ?//

void EMfields3D::init_Turbulence_Decay()
{
    const Grid *grid = &get_grid();
    const Collective *col = &get_col();
    const VirtualTopology3D *vct = &get_vct();

    //* Custom input parameters
    const long long turbulence_seed        = static_cast<long long>(input_param[0]);    //* "Random" turbulence seed (only for phases)
    const double    turbulence_amplitude   = input_param[1];                            //* Per-component RMS amplitude of turbulent perturbation, normalised to B_BG
    const int       kmin                   = int(input_param[2]);                       //* Minimum integer turbulent mode number
    const int       kmax                   = int(input_param[3]);                       //* Maximum integer turbulent mode number
    const int       turbulence_plane       = int(input_param[4]);                       //* Turbulence plane: 0 -> xy, 1 -> yz, 2 -> zx
    const double    spectral_index         = std::max(0.0, input_param[5]);             //* Power-law index; E_B(k) ~ k^(-p)

    if (input_param[5] < 0.0 && vct->getCartesian_rank() == 0)
    {
        cout << "WARNING: negative spectral_index is clamped to 0 " << endl;
        cout << "   No particular slope is imposed! " << endl;
    }

    //* ------------------------------------------------------------------
    //* Turbulent magnetic perturbation (2D planar, divergence-free)
    //* ------------------------------------------------------------------
    //* The perturbation is built from a single out-of-plane vector-potential
    //* component, so that delta_B = curl(A) is divergence-free by construction.
    //* The plane selects which potential component is active:
    //*    turbulence_plane = 0 -> xy : A_z(x,y) -> delta_Bx, delta_By ; delta_Bz = 0
    //*    turbulence_plane = 1 -> yz : A_x(y,z) -> delta_By, delta_Bz ; delta_Bx = 0
    //*    turbulence_plane = 2 -> zx : A_y(z,x) -> delta_Bz, delta_Bx ; delta_By = 0
    //*
    //* Spectral method (derived from spectral_index):
    //*    spectral_index >  0 -> power law: weight ~ k^(-(spectral_index+3)/2), E_B(k) ~ k^(-spectral_index)
    //*    spectral_index == 0 -> random: random Gaussian amplitude on delta_B per mode, no slope imposed
    //*
    //* Amplitude: each populated in-plane component is independently rescaled so
    //* that its RMS equals turbulence_amplitude * B_BG.

    struct TurbMode
    {
        double kx; double ky; double kz;
        double phase; double potential_amplitude;
    };

    std::vector<TurbMode> modes;

    //* Which two directions the in-plane modes vary in, set by the plane
    bool vary_x = false; bool vary_y = false; bool vary_z = false;

    if      (turbulence_plane == 0) { vary_x = true;  vary_y = true; }    //* xy : A_z(x,y)
    else if (turbulence_plane == 1) { vary_y = true;  vary_z = true; }    //* yz : A_x(y,z)
    else if (turbulence_plane == 2) { vary_z = true;  vary_x = true; }    //* zx : A_y(z,x)
    else
    {
        if (vct->getCartesian_rank() == 0)
            cout << "Incorrect turbulence plane. Choose 0 for xy, 1 for yz, or 2 for zx" << endl;
        abort();
    }

    if (turbulence_amplitude > 0.0 && kmax >= kmin)
    {
        std::mt19937_64 rng(turbulence_seed);
        std::uniform_real_distribution<double> phase_dist(0.0, 2.0*M_PI);
        std::normal_distribution<double>       gauss_dist(0.0, 1.0);

        //* Fundamental wavenumbers of the box in each direction
        const double k0x = 2.0*M_PI/Lx;
        const double k0y = 2.0*M_PI/Ly;
        const double k0z = 2.0*M_PI/Lz;

        //* Reference wavenumber for the power-law weight (smallest active fundamental)
        double k0_ref = 0.0;
        if (vary_x) k0_ref = (k0_ref == 0.0) ? k0x : std::min(k0_ref, k0x);
        if (vary_y) k0_ref = (k0_ref == 0.0) ? k0y : std::min(k0_ref, k0y);
        if (vary_z) k0_ref = (k0_ref == 0.0) ? k0z : std::min(k0_ref, k0z);

        //* Power-law weight exponent on the potential (2D, d = 2):
        //*     E_B(k) ~ k^(d-1) * k^2 |A_k|^2 = k^(d+1) |A_k|^2 = k^(-p)
        //*     =>  |A_k| ~ k^(-(p+3)/2)
        const double weight_exponent = -0.5*(spectral_index + 3.0);

        //* Finalise one mode given its integer mode numbers and random factor
        auto append_mode = [&](int nx, int ny, int nz, double random_factor)
        {
            double kx = vary_x ? double(nx)*k0x : 0.0;
            double ky = vary_y ? double(ny)*k0y : 0.0;
            double kz = vary_z ? double(nz)*k0z : 0.0;

            double k_mag = sqrt(kx*kx + ky*ky + kz*kz);

            if (k_mag == 0.0)   return;

            //* POWER LAW (spectral_index > 0): weight potential by k^(-(p+3)/2) -> E_B(k) ~ k^(-p)
            //* RANDOM    (spectral_index == 0): random amplitude on delta_B directly. Since
            //*             delta_B = curl(A) carries a factor of k, a field amplitude that is
            //*             purely the random factor needs potential ~ random/k.
            double potential_weight = (spectral_index > 0.0) ? pow(k_mag/k0_ref, weight_exponent) : 1.0/k_mag;

            TurbMode mode;
            mode.kx                  = kx;
            mode.ky                  = ky;
            mode.kz                  = kz;
            mode.phase               = phase_dist(rng);
            mode.potential_amplitude = random_factor * potential_weight;
            modes.push_back(mode);
        };

        //* Excite every integer mode in the band [kmin, kmax]
        //* Iterate over a half-space of mode numbers to avoid 
        //* double-counting +(n) and -(n) (the same real mode)

        const int range_x = vary_x ? kmax : 0;
        const int range_y = vary_y ? kmax : 0;
        const int range_z = vary_z ? kmax : 0;

        for (int nx = 0; nx <= range_x; nx++)
            for (int ny = -range_y; ny <= range_y; ny++)
                for (int nz = -range_z; nz <= range_z; nz++)
                {
                    //* Half-space selection: first non-zero leading component
                    //* positive; skip the mirror image and the zero mode.
                    if (nx < 0) continue;
                    if (nx == 0 && ny < 0) continue;
                    if (nx == 0 && ny == 0 && nz <= 0) continue;

                    double n_mag = sqrt(double(nx*nx + ny*ny + nz*nz));

                    if (n_mag < double(kmin)) continue;
                    if (n_mag > double(kmax)) continue;

                    double random_factor = (spectral_index > 0.0) ? 1.0 : gauss_dist(rng);

                    append_mode(nx, ny, nz, random_factor);
                }
    }

    //? Curl of vector potential, summed over modes. Only the two in-plane
    //? field components are non-zero for each plane; the third is zero.
    
    //* xy: A = A_z z_hat -> delta_Bx =  dA_z/dy , delta_By = -dA_z/dx
    auto turb_Bx = [&](double x, double y, double z)
    {
        double dBx = 0.0;
        for (int m = 0; m < int(modes.size()); m++)
        {
            double arg = modes[m].kx*x + modes[m].ky*y + modes[m].kz*z + modes[m].phase;
            double s   = sin(arg);
            if      (turbulence_plane == 0) dBx += -modes[m].ky * modes[m].potential_amplitude * s;     //* xy:  dA_z/dy
            else if (turbulence_plane == 2) dBx +=  modes[m].kz * modes[m].potential_amplitude * s;     //* zx: -dA_y/dz
            //* yz: delta_Bx = 0
        }
        return dBx;
    };

    //* yz: A = A_x x_hat -> delta_By =  dA_x/dz , delta_Bz = -dA_x/dy
    auto turb_By = [&](double x, double y, double z)
    {
        double dBy = 0.0;
        for (int m = 0; m < int(modes.size()); m++)
        {
            double arg = modes[m].kx*x + modes[m].ky*y + modes[m].kz*z + modes[m].phase;
            double s   = sin(arg);
            if      (turbulence_plane == 0) dBy +=  modes[m].kx * modes[m].potential_amplitude * s;     //* xy: -dA_z/dx
            else if (turbulence_plane == 1) dBy += -modes[m].kz * modes[m].potential_amplitude * s;     //* yz:  dA_x/dz
            //* zx: delta_By = 0
        }
        return dBy;
    };

    //* zx: A = A_y y_hat -> delta_Bz =  dA_y/dx , delta_Bx = -dA_y/dz
    auto turb_Bz = [&](double x, double y, double z)
    {
        double dBz = 0.0;
        for (int m = 0; m < int(modes.size()); m++)
        {
            double arg = modes[m].kx*x + modes[m].ky*y + modes[m].kz*z + modes[m].phase;
            double s   = sin(arg);
            if      (turbulence_plane == 1) dBz +=  modes[m].ky * modes[m].potential_amplitude * s;     //* yz: -dA_x/dy
            else if (turbulence_plane == 2) dBz += -modes[m].kx * modes[m].potential_amplitude * s;     //* zx:  dA_y/dx
            //* xy: delta_Bz = 0
        }
        return dBz;
    };

    //! New initial setup
    if (col->getRestart_status() == 0)
    {
        if (vct->getCartesian_rank() == 0)
        {
            const char* plane_name = (turbulence_plane == 0) ? "xy" : (turbulence_plane == 1) ? "yz" : "zx";

            cout << "-----------------------------------------------------------"   << endl;
            cout << "                   Decaying Turbulence"                        << endl;
            cout << "-----------------------------------------------------------"   << endl << endl;

            cout << "Turbulent perturbation amplitude (dB/B0)           = " << turbulence_amplitude             << endl;
            cout << "Number of turbulent modes                          = " << modes.size()                     << endl;
            cout << "Turbulent mode range                               = " << kmin << " to " << kmax           << endl;
            cout << "Spectral method                                    = " << (spectral_index > 0.0 ? "power law" : "random (no slope)") << endl;
            if (spectral_index > 0.0)
                cout << "Spectral index p in E_B(k) ~ k^(-p)            = " << spectral_index << endl;
            cout << "Turbulence plane                                   = " << plane_name << " (delta_Bz " << (turbulence_plane == 0 ? "= 0" : "!= 0") << ")" << endl;
            cout << "Turbulence seed                                    = " << turbulence_seed                  << endl << endl;

            cout << "-----------------------------------------------------------"   << endl;
        }

        //* Measure the RMS of each turbulent component so that each can be
        //* normalised independently to turbulence_amplitude * B_BG.
        double local_sumsq_Bx = 0.0;
        double local_sumsq_By = 0.0;
        double local_sumsq_Bz = 0.0;
        double local_count    = 0.0;
        double xN, yN, zN;

        for (int i = 1; i < nxn-1; i++)
            for (int j = 1; j < nyn-1; j++)
                for (int k = 1; k < nzn-1; k++)
                {
                    xN = grid->getXN(i, j, k);
                    yN = grid->getYN(i, j, k);
                    zN = grid->getZN(i, j, k);

                    double dBx = turb_Bx(xN, yN, zN);
                    double dBy = turb_By(xN, yN, zN);
                    double dBz = turb_Bz(xN, yN, zN);

                    local_sumsq_Bx += dBx*dBx;
                    local_sumsq_By += dBy*dBy;
                    local_sumsq_Bz += dBz*dBz;
                    local_count    += 1.0;
                }

        double global_sumsq_Bx = 0.0;
        double global_sumsq_By = 0.0;
        double global_sumsq_Bz = 0.0;
        double global_count    = 0.0;

        MPI_Allreduce(&local_sumsq_Bx, &global_sumsq_Bx, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&local_sumsq_By, &global_sumsq_By, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&local_sumsq_Bz, &global_sumsq_Bz, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&local_count,    &global_count,    1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        //* Independent per-component scale factors
        double scale_Bx = 0.0;
        double scale_By = 0.0;
        double scale_Bz = 0.0;

        if (turbulence_amplitude > 0.0 && global_count > 0.0)
        {
            const double B_ref = sqrt(B0x*B0x + B0y*B0y + B0z*B0z);
            double target_rms = (B_ref > 0.0) ? turbulence_amplitude * B_ref : turbulence_amplitude;

            if (global_sumsq_Bx > 0.0) scale_Bx = target_rms / sqrt(global_sumsq_Bx/global_count);
            if (global_sumsq_By > 0.0) scale_By = target_rms / sqrt(global_sumsq_By/global_count);
            if (global_sumsq_Bz > 0.0) scale_Bz = target_rms / sqrt(global_sumsq_Bz/global_count);
        }

        //* Total B and E fields
        for (int i = 1; i < nxn-1; i++)
            for (int j = 1; j < nyn-1; j++)
                for (int k = 1; k < nzn-1; k++)
                {
                    xN = grid->getXN(i, j, k);
                    yN = grid->getYN(i, j, k);
                    zN = grid->getZN(i, j, k);

                    //* Initialise B on nodes
                    Bxn[i][j][k] = B0x;
                    Byn[i][j][k] = B0y;
                    Bzn[i][j][k] = B0z;

                    //* Add divergence-free turbulence; each component scaled to the exact amplitude
                    Bxn[i][j][k] += scale_Bx * turb_Bx(xN, yN, zN);
                    Byn[i][j][k] += scale_By * turb_By(xN, yN, zN);
                    Bzn[i][j][k] += scale_Bz * turb_Bz(xN, yN, zN);

                    //* Initialise E on nodes
                    Ex[i][j][k] = 0.0;
                    Ey[i][j][k] = 0.0;
                    Ez[i][j][k] = 0.0;

                    //* Initialize rho on nodes
                    for (int is = 0; is < ns; is++)
                        rhons[is][i][j][k] = rhoINIT[is] / FourPI;
                }


        //* Initialise B and rho on cell centers
        grid->interpN2C(Bxc, Bxn);
        grid->interpN2C(Byc, Byn);
        grid->interpN2C(Bzc, Bzn);

        for (int is = 0; is < ns; is++)
            grid->interpN2C(rhocs, is, rhons);

        //* Communicate ghost data on cell centres
        communicateCenterBC(nxc, nyc, nzc, Bxc, col->bcBx[0],col->bcBx[1],col->bcBx[2],col->bcBx[3],col->bcBx[4],col->bcBx[5], vct, this);
        communicateCenterBC(nxc, nyc, nzc, Byc, col->bcBy[0],col->bcBy[1],col->bcBy[2],col->bcBy[3],col->bcBy[4],col->bcBy[5], vct, this);
        communicateCenterBC(nxc, nyc, nzc, Bzc, col->bcBz[0],col->bcBz[1],col->bcBz[2],col->bcBz[3],col->bcBz[4],col->bcBz[5], vct, this);
    }

    else if (col->getRestart_status() == 1 || col->getRestart_status() == 2)
    {
        //! Read data from restart files
        init_fields_restart();
    }

    else
    {
        if (vct->getCartesian_rank() == 0)
        {
            cout << "Incorrect restart status!" << endl;
            cout << "   restart_status = 0 ---> NO RESTART!" << endl;
            cout << "   restart_status = 1 or 2 ---> RESTART!" << endl;
        }
        abort();
    }
}

//? ========================================================================== ?//