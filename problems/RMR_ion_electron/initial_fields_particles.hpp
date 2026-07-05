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


//? ========================================================================== ?//

//? Relativistic double Harris for ion-electron plasma: Maxwellian background, drifting particles in the sheets
void Particles3D::Relativistic_Double_Harris_ion_electron(Field * EMf) 
{
    //* Custom input parameters for relativistic reconnection
    const double sigma                  = input_param[0];       //* Magnetisation parameter
    const double CS_density             = input_param[1];       //* Ratio of current sheet density to upstream density (this is "alpha" in Fabio's paper; Eqs 52 and 53)
    const double CS_thickness           = input_param[2];       //* Half-thickness of current sheet (free parameter)
    
    //! New initial setup
    if (col->getRestart_status() == 0)
    {
        //* Initialise random generator with different seed on different processor
        long long seed = (vct->getCartesian_rank() + 1)*20 + ns;
        srand(seed);
        srand48(seed);

        assert_eq(_pcls.size(), 0);

        //* Background (BG) or upstream particles
        double thermal_spread_BG_electrons  = col->getUth(0);                           //* Thermal spread of electrons
        double thermal_spread_BG_ions       = col->getUth(1);                           //* Thermal spread of ions
        double rho_BG                       = col->getRHOinit(ns)/(4.0*M_PI);           //* Density (rho_BG = n * mc^2)
        double B_BG                         = sqrt(sigma*4.0*M_PI*rho_BG);              //* sigma = B^2/(4*pi*rho_electrons)
        
        //* Current sheet (CS) particles
        double rho_CS                       = CS_density*rho_BG;                                    //* Density (rho_CS = CS_density * n * mc^2)
        double drift_velocity               = B_BG/(8.0*M_PI*rho_CS*CS_thickness/c);                //* v = B*c/(8 * pi * rho_CS * CS_thickness); Eq 52
        double lorentz_factor_CS            = 1.0/sqrt(1.0 - drift_velocity*drift_velocity);        //* Lorentz factor of the relativistic drifting particles
        double thermal_spread_CS_ions       = B_BG*B_BG*lorentz_factor_CS/(16.0*M_PI*rho_CS);       //* Thermal spread of ions (B^2 * Gamma/(16 * pi * CS_density * n * mc^2)); Eq 53
        double thermal_spread_CS_electrons  = thermal_spread_CS_ions * fabs(col->getQOM(0));        //* Thermal spread of electrons (Ratio of thermal spread = mass ratio)
        
        const double q_factor = (qom/fabs(qom)) * grid->getVOL()/npcel;
        
        for (int i = 1; i < grid->getNXC() - 1; i++)
            for (int j = 1; j < grid->getNYC() - 1; j++)
                for (int k = 1; k < grid->getNZC() - 1; k++)
                    for (int ii = 0; ii < npcelx; ii++)
                        for (int jj = 0; jj < npcely; jj++)
                            for (int kk = 0; kk < npcelz; kk++) 
                            {
                                const double x = (ii + 0.5) * (dx / npcelx) + grid->getXN(i, j, k);
                                const double y = (jj + 0.5) * (dy / npcely) + grid->getYN(i, j, k);
                                const double z = (kk + 0.5) * (dz / npcelz) + grid->getZN(i, j, k);

                                //* Velocities and charges of particles
                                double u, v, w, q;
                            
                                //* Distinguish between background and drifting species
                                if (ns < 2) 
                                {
                                    //? Background species (these are the particles that get accelerated)
                                    q = q_factor * rho_BG;
                                    
                                    //* Velocity of relativistic nondrifting Maxwellian
                                    if (qom < 0.0) 
                                        sample_Maxwell_Juttner(u, v, w, thermal_spread_BG_electrons, 1.0, 0);
                                    else        
                                        sample_Maxwell_Juttner(u, v, w, thermal_spread_BG_ions, 1.0, 0);   
                                }
                                else 
                                {
                                    //? Current sheet species (necessary to initialise a current sheet)
                                    double fs;

                                    if (y < (Ly/2.0))   
                                        fs = sech_square((y - (Ly/4.0))/CS_thickness);
                                    else              
                                        fs = sech_square((y - (3.0*Ly/4.0))/CS_thickness);
                            
                                    //* Skip the particle if its weight is too small
                                    if (fabs(fs) < 1.e-8) continue;

                                    q = q_factor * rho_CS * fs;

                                    //* Velocity of relativistic drifting (along the Z-axis) Maxwellian
                                    if (qom < 0.0) 
                                        sample_Maxwell_Juttner(u, v, w, thermal_spread_CS_electrons, lorentz_factor_CS, -3);    //* Negative charges (e.g., electrons)
                                    else
                                        sample_Maxwell_Juttner(u, v, w, thermal_spread_CS_ions, lorentz_factor_CS, 3);          //* Positive charges (e.g., ions)
                                    
                                    //* Flip sign of drift velocity for particles in the second layer
                                    if (y > (Ly/2.0))
                                    {
                                        u = -u; 
                                        v = -v; 
                                        w = -w;
                                    }
                                }

                                create_new_particle(u, v, w, q, x, y, z);
                            }

        fixPosition();
    }

    //! If col->getRestart_status() == 0 or 1, particle restart data is automatically read from the restart#.hdf files
}

//? ========================================================================== ?//

void EMfields3D::init_Relativistic_Double_Harris_ion_electron()
{
    const Grid *grid = &get_grid();
    const Collective *col = &get_col();
    const VirtualTopology3D *vct = &get_vct();

    //* Custom input parameters
    const double sigma                  = input_param[0];           //* Magnetisation parameter
    const double CS_density             = input_param[1];           //* Ratio of current sheet density to upstream density (this is "alpha" in Fabio's paper; Eqs 52 and 53)
    const double CS_thickness           = input_param[2];           //* Half-thickness of current sheet (free parameter)
    const double guide_field            = input_param[3];           //* Ratio of guide field to in-plane magnetic field
    const double turbulence_amplitude   = input_param[4];           //* Per-component RMS amplitude of turbulent perturbation, normalised to B_BG
    const int    nmodes                 = int(input_param[5]);      //* Number of turbulent Fourier modes (only used if use_shell_fill = 0)
    const int    kmin                   = int(input_param[6]);      //* Minimum integer turbulent mode number
    const int    kmax                   = int(input_param[7]);      //* Maximum integer turbulent mode number
    const int    use_shell_fill         = int(input_param[9]);      //* Mode selection: 0 -> random sampling, 1 -> deterministic shell fill
    const int    turbulence_plane       = int(input_param[10]);     //* Turbulence plane: 0 -> xy, 1 -> yz, 2 -> zx

    //* Power-law index p in E_B(k) ~ k^(-p). A negative input is clamped to 0.
    //*    spectral_index >  0 -> power law, E_B(k) ~ k^(-spectral_index)
    //*    spectral_index == 0 -> random Gaussian amplitude on delta_B, no particular slope
    const double spectral_index         = (input_param[8] > 0.0) ? input_param[8] : 0.0;

    const long long turbulence_seed     = 12345LL;                                          //* "Random" turbulence seed

    //* Background (BG) or upstream particles
    double thermal_spread_BG_electrons  = col->getUth(0);                                   //* Thermal spread of electrons
    double thermal_spread_BG_ions       = col->getUth(1);                                   //* Thermal spread of ions
    double rho_BG                       = rhoINIT[0]/(4.0*M_PI);                            //* Density (rho_BG = n * mc^2)
    double B_BG                         = sqrt(sigma*4.0*M_PI*rho_BG);                      //* sigma = B^2/(4*pi*rho_electrons)

    //* Current sheet (CS) particles
    double rho_CS                       = CS_density*rho_BG;                                //* Density (rho_CS = CS_density * n * mc^2)
    double drift_velocity               = B_BG/(8.0*M_PI*rho_CS*CS_thickness/c);            //* v = B*c/(8 * pi * rho_CS * CS_thickness); Eq 52
    double lorentz_factor_CS            = 1.0/sqrt(1.0 - drift_velocity*drift_velocity);    //* Lorentz factor of the relativistic drifting particles
    double thermal_spread_CS_ions       = B_BG*B_BG*lorentz_factor_CS/(16.0*M_PI*rho_CS);   //* Thermal spread of ions (B^2 * Gamma/(16 * pi * CS_density * n * mc^2)); Eq 53
    double thermal_spread_CS_electrons  = thermal_spread_CS_ions * fabs(col->getQOM(0));    //* Thermal spread of electrons (Ratio of thermal spread = mass ratio)

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
    //*    spectral_index >  0 -> power law: weight ~ k^(-(spectral_index+3)/2),
    //*                           giving shell-integrated E_B(k) ~ k^(-spectral_index)
    //*    spectral_index == 0 -> random: random Gaussian amplitude on delta_B per
    //*                           mode, no slope imposed
    //*
    //* Amplitude: each populated in-plane component is independently rescaled so
    //* that its RMS equals turbulence_amplitude * B_BG.

    struct TurbMode
    {
        double kx;
        double ky;
        double kz;
        double phase;
        double potential_amplitude;
    };

    std::vector<TurbMode> modes;

    //* Which two directions the in-plane modes vary in, set by the plane
    bool vary_x = false;
    bool vary_y = false;
    bool vary_z = false;

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

        if (use_shell_fill != 0)
        {
            //* Deterministic shell fill: every integer mode in the band. 
            //* Iterate over a half-space of mode numbers to avoid double-counting 
            //* +(n) and -(n) (the same real mode). In the power-law method the 
            //* amplitude is the spectral weight (random factor = 1, phase random); in
            //* the random method each mode gets an independent Gaussian random factor.
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
        else
        {
            //* Random mode sampling: draw "nmodes" modes. Each active direction gets a random 
            //* integer mode number in [kmin, kmax] and a random sign; the inactive direction
            //* stays zero. A Gaussian factor multiplies the (possibly unit) weight.
            std::uniform_int_distribution<int> mode_dist(kmin, kmax);
            std::uniform_int_distribution<int> sign_dist(0, 1);

            for (int m = 0; m < nmodes; m++)
            {
                int nx = 0;
                int ny = 0;
                int nz = 0;

                if (vary_x)
                {
                    nx = mode_dist(rng);
                    if (sign_dist(rng) == 0) nx = -nx;
                }
                if (vary_y)
                {
                    ny = mode_dist(rng);
                    if (sign_dist(rng) == 0) ny = -ny;
                }
                if (vary_z)
                {
                    nz = mode_dist(rng);
                    if (sign_dist(rng) == 0) nz = -nz;
                }

                append_mode(nx, ny, nz, gauss_dist(rng));
            }
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
            cout << "Relativistic double Harris sheet for ion-electron plasma"      << endl;
            cout << "-----------------------------------------------------------"   << endl << endl;

            cout << "Ratio of CS density to upstream density            = " << CS_density                       << endl;
            cout << "Ratio of guide magnetic field to background field  = " << guide_field                      << endl;
            cout << "Turbulent perturbation amplitude (dB/B0)           = " << turbulence_amplitude             << endl;
            cout << "Number of turbulent modes                          = " << modes.size()                     << endl;
            cout << "Turbulent mode range                               = " << kmin << " to " << kmax           << endl;
            cout << "Spectral method                                    = " << (spectral_index > 0.0 ? "power law" : "random (no slope)") << endl;
            if (spectral_index > 0.0)
            cout << "Spectral index p in E_B(k) ~ k^(-p)                = " << spectral_index                   << endl;
            cout << "Mode-selection method                              = " << (use_shell_fill != 0 ? "shell fill" : "random sampling") << endl;
            cout << "Turbulence plane                                   = " << plane_name << " (delta_Bz " << (turbulence_plane == 0 ? "= 0" : "!= 0") << ")" << endl;
            cout << "Turbulence seed                                    = " << turbulence_seed                  << endl << endl;

            cout << "BACKGROUND/UPSTREAM:"                                                                      << endl;
            cout << "   Magnetisation parameter (ions)                  = " << sigma                            << endl;
            cout << "   Plasma density                                  = " << 2.0*rho_BG*thermal_spread_BG_ions/(B_BG*B_BG/2.0/FourPI)  << endl;
            cout << "   Thermal spread of ions                          = " << thermal_spread_BG_ions           << endl;
            cout << "   Thermal spread of electrons                     = " << thermal_spread_BG_electrons      << endl;
            cout << "   Lorentz factor of electrons                     = " << 3*thermal_spread_BG_electrons    << endl << endl;

            cout << "CURRENT SHEET:"                                                                            << endl;
            cout << "   Thermal spread of drifting ions                 = " << thermal_spread_CS_ions           << endl;
            cout << "   Thermal spread of drifting electrons            = " << thermal_spread_CS_electrons      << endl;
            cout << "   Lorentz factor of drifting particles            = " << lorentz_factor_CS                << endl;

            cout << "-----------------------------------------------------------"   << endl;
        }

        //* Params for setting up current sheet
        double xN, yN, zN, yh, fBx;

        //* Measure the RMS of each turbulent component so that each can be
        //* normalised independently to turbulence_amplitude * B_BG.
        double local_sumsq_Bx = 0.0;
        double local_sumsq_By = 0.0;
        double local_sumsq_Bz = 0.0;
        double local_count    = 0.0;

        for (int i = 1; i < nxc-1; i++)
            for (int j = 1; j < nyc-1; j++)
                for (int k = 1; k < nzc-1; k++)
                {
                    xN = grid->getXC(i, j, k);
                    yN = grid->getYC(i, j, k);
                    zN = grid->getZC(i, j, k);

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
            double target_rms = turbulence_amplitude * B_BG;
            if (global_sumsq_Bx > 0.0) scale_Bx = target_rms / sqrt(global_sumsq_Bx/global_count);
            if (global_sumsq_By > 0.0) scale_By = target_rms / sqrt(global_sumsq_By/global_count);
            if (global_sumsq_Bz > 0.0) scale_Bz = target_rms / sqrt(global_sumsq_Bz/global_count);
        }

        //* Total B and E fields
        for (int i = 1; i < nxc-1; i++)
            for (int j = 1; j < nyc-1; j++)
                for (int k = 1; k < nzc-1; k++)
                {
                    xN = grid->getXC(i, j, k);
                    yN = grid->getYC(i, j, k);
                    zN = grid->getZC(i, j, k);

                    if (yN <= (Ly/2.0))
                    {
                        yh  = yN-(Ly/4.0);
                        fBx = -1.0;
                    }
                    else
                    {
                        yh  = yN-(3.0*Ly/4.0);
                        fBx = 1.0;
                    }

                    //* Reconnecting/reversing field
                    Bxc[i][j][k] = fBx * B_BG * tanh(yh/CS_thickness);

                    //* Normal-to-current-sheet field
                    Byc[i][j][k] = 0.0;

                    //* Guide field
                    Bzc[i][j][k] = B_BG*guide_field;

                    //* Add divergence-free turbulence; each component scaled to the exact amplitude
                    Bxc[i][j][k] += scale_Bx * turb_Bx(xN, yN, zN);
                    Byc[i][j][k] += scale_By * turb_By(xN, yN, zN);
                    Bzc[i][j][k] += scale_Bz * turb_Bz(xN, yN, zN);
                }

        for (int i = 0; i < nxn; i++)
            for (int j = 0; j < nyn; j++)
                for (int k = 0; k < nzn; k++)
                {
                    //* Initialise E on nodes
                    Ex[i][j][k] = 0.0;
                    Ey[i][j][k] = 0.0;
                    Ez[i][j][k] = 0.0;
                }

        //* Communicate ghost data at cell centres
        communicateCenterBC(nxc, nyc, nzc, Bxc, col->bcBx[0],col->bcBx[1],col->bcBx[2],col->bcBx[3],col->bcBx[4],col->bcBx[5], vct, this);
        communicateCenterBC(nxc, nyc, nzc, Byc, col->bcBy[0],col->bcBy[1],col->bcBy[2],col->bcBy[3],col->bcBy[4],col->bcBy[5], vct, this);
        communicateCenterBC(nxc, nyc, nzc, Bzc, col->bcBz[0],col->bcBz[1],col->bcBz[2],col->bcBz[3],col->bcBz[4],col->bcBz[5], vct, this);

        //* Initialise B on nodes
        grid->interpC2N(Bxn, Bxc);
        grid->interpC2N(Byn, Byc);
        grid->interpC2N(Bzn, Bzc);

        //* Communicate ghost data on nodes
        communicateNodeBC(nxn, nyn, nzn, Bxn, col->bcBx[0],col->bcBx[1],col->bcBx[2],col->bcBx[3],col->bcBx[4],col->bcBx[5], vct, this);
        communicateNodeBC(nxn, nyn, nzn, Byn, col->bcBy[0],col->bcBy[1],col->bcBy[2],col->bcBy[3],col->bcBy[4],col->bcBy[5], vct, this);
        communicateNodeBC(nxn, nyn, nzn, Bzn, col->bcBz[0],col->bcBz[1],col->bcBz[2],col->bcBz[3],col->bcBz[4],col->bcBz[5], vct, this);
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