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

/*******************************************************************************************
  PSKOutput.h  -  Framework classes for PARSEK output
  -------------------
developers: D. Burgess, June/July 2006
 ********************************************************************************************/
#ifndef _PSK_OUTPUT_H_
#define _PSK_OUTPUT_H_

#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>
#include <list>

#include "errors.h"
#include "PSKException.h"
#include "Particles3Dcomm.h"
#include "Field.h"
#include "Collective.h"
#include "VCtopology3D.h"
#include "MPIdata.h"
#include "ipicdefs.h"

using std::string;
using std::stringstream;
using std::ofstream;
using std::cout;
using std::endl;

/**
 * 
 * Framework classes for PARSEK output
 * @date June/July 2006
 * @author David Burgess
 * @version 2.0
 *
 */

namespace PSK {

  /** class for handling IO exception*/
  class OutputException:public Exception {
  public:
  OutputException(const std::string & err_str, const std::string fn_str = "", int sys_errno = 0):Exception(err_str, fn_str, sys_errno) {
      _type_str += "::OutputException";
  } OutputException(const char *err_str, const char *fn_str = "", int sys_errno = 0):Exception(err_str, fn_str, sys_errno) {
      _type_str += "::OutputException";
  }};

  /** 
   *
   * !\brief Class for dimensionality of rectangular arrays
   *
   */
  class Dimens {

    std::vector < int >_dimens;
    friend class OutputAdaptor;

  public:
    Dimens(void) {;
    } Dimens(const Dimens & dimens):_dimens(dimens._dimens) {;
    }
    Dimens(const std::vector < int >&dimens):_dimens(dimens) {;
    }
    Dimens(int d1):_dimens(1) {
      _dimens[0] = d1;
    }
    Dimens(int d1, int d2):_dimens(2) {
      _dimens[0] = d1;
      _dimens[1] = d2;
    }
    Dimens(int d1, int d2, int d3):_dimens(3) {
      _dimens[0] = d1;
      _dimens[1] = d2;
      _dimens[2] = d3;
    }
    Dimens(int d1, int d2, int d3, int d4):_dimens(4) {
      _dimens[0] = d1;
      _dimens[1] = d2;
      _dimens[2] = d3;
      _dimens[3] = d4;
    }

    int size(void) const {
      return _dimens.size();
    } int operator[] (const int i) const {
      return _dimens[i];
    } int nels(void) const {
      int n = 1;
      for (int i = 0; i < _dimens.size(); ++i)
        n *= _dimens[i];
      return n;
    }
  };

  /**
   *  !\brief Virtual base class  
   *
   */
  class OutputAdaptor {

  public:
    OutputAdaptor(void) {;
    } virtual void open(const std::string & outf) {
      eprintf("Function not implemented");
      eprintf("Function not implemented");
    }
    virtual void close(void) {
      eprintf("Function not implemented");
    }

    // write int functions
    virtual void write(const std::string & objname, int i) {
      eprintf("Function not implemented");
    }
    virtual void write(const std::string & objname, const Dimens dimens, const int *i_array) {
      eprintf("Function not implemented");
    }
    virtual void write(const std::string & objname, const Dimens dimens, const longid *i_array) {
      eprintf("Function not implemented");
    }
    virtual void write(const std::string & objname, const Dimens dimens, const std::vector < int >&i_array) {
      eprintf("Function not implemented");
    }

    // write float functions
    virtual void write(const std::string & objname, float f) {
      eprintf("Function not implemented");
    }
    virtual void write(const std::string & objname, const Dimens dimens, const float *f_array) {
      eprintf("Function not implemented");
    }
    virtual void write(const std::string & objname, const Dimens dimens, const std::vector < float >&f_array) {
      eprintf("Function not implemented");
    }

    // write double functions
    virtual void write(const std::string & objname, double d) {
      eprintf("Function not implemented");
    }
    virtual void write(const std::string & objname, const Dimens dimens, const double *d_array) {
      eprintf("Function not implemented");
    }
    virtual void write(const std::string & objname, const Dimens dimens, const std::vector < double >&d_array) {
      eprintf("Function not implemented");
    }

  };
  /** bse class for output agent */
  class OutputAgentBase {

  public:
    OutputAgentBase(void) {;} 
    OutputAgentBase(const OutputAgentBase & a) {;}

    virtual void output_fields(const std::string & tag, int cycle) = 0;
    virtual void output_particles(const std::string & tag, int cycle) = 0;
    virtual void output_particles_DS(const std::string & tag, int cycle, int sample) = 0;
    virtual void output(const std::string & tag, int cycle) = 0;
    virtual void open(const std::string & outf) = 0;
    virtual void close(void) = 0;
  };

  /** \brief Base class for OutputAgents using template for output adaptor */
template < class Toa > class OutputAgent:public OutputAgentBase {

  protected:
    Toa output_adaptor;

  public:
    OutputAgent(void) {;}
    OutputAgent(const OutputAgent & a) {;}

    virtual void output(const std::string & tag, int cycle) = 0;
    virtual void output_fields(const std::string & tag, int cycle) = 0;
    virtual void output_particles(const std::string & tag, int cycle) = 0;
    virtual void output_particles_DS(const std::string & tag, int cycle, int sample) = 0;

    void open(const std::string & outf) {
      output_adaptor.open(outf);
    }
    void open_append(const std::string & outf) {
      output_adaptor.open_append(outf);
    }

    void close(void) {
      output_adaptor.close();
    }
    // grid
    Grid *mygrid;
  };

  /** \brief Container (list) class for OutputAgents */
  template < class Toa > class OutputManager {
    std::list < OutputAgentBase * >agents_list;

  public:
    OutputManager(void) {;
    }

    void push_back(OutputAgentBase * a_p) {
      agents_list.push_back(a_p);
    }

    void output(const std::string & tag, int cycle) 
    {
        typename std::list < OutputAgentBase * >::iterator p = agents_list.begin();
        while (p != agents_list.end())
            (*p++)->output(tag, cycle);
    }

    void output_fields(const std::string & tag, int cycle) 
    {
        typename std::list < OutputAgentBase * >::iterator p = agents_list.begin();
        while (p != agents_list.end())
            (*p++)->output_fields(tag, cycle);
    }

    void output_particles(const std::string & tag, int cycle) 
    {
        typename std::list < OutputAgentBase * >::iterator p = agents_list.begin();
        while (p != agents_list.end())
            (*p++)->output_particles(tag, cycle);
    }

    void output_particles_DS(const std::string & tag, int cycle, int sample) 
    {
        typename std::list < OutputAgentBase * >::iterator p = agents_list.begin();
        while (p != agents_list.end())
            (*p++)->output_particles_DS(tag, cycle, sample);
    }

  };

  class coutOutputAdaptor:public OutputAdaptor {

  public:
    coutOutputAdaptor(void) {;
    } void open(const std::string & outf) {
      std::cout << "coutPSKOutputAdaptor open() file: " << outf << "\n";
    }
    void close(void) {
      std::cout << "coutPSKOutputAdaptor close()\n";
    }


    //* write int functions
    void write(const std::string & objname, int i) {
      std::cout << "coutPSKOutputAdaptor write int: <" << objname << "> : " << i << "\n";
    }

    void write(const std::string & objname, const Dimens dimens, const int *i_array) {
      std::cout << "coutPSKOutputAdaptor write int* array: <" << objname << "> : " << "\n";
    }
    void write(const std::string & objname, const Dimens dimens, const std::vector < int >&i_array) {
      std::cout << "coutPSKOutputAdaptor write vector<int> array: <" << objname << "> : " << "\n";
    }


    //* write float functions
    void write(const std::string & objname, float f) {
      std::cout << "coutPSKOutputAdaptor write float: <" << objname << "> : " << f << "\n";
    }
    void write(const std::string & objname, const Dimens dimens, const float *f_array) {
      std::cout << "coutPSKOutputAdaptor write float* array: <" << objname << "> : " << "\n";
    }
    void write(const std::string & objname, const Dimens dimens, const std::vector < float >&f_array) {
      std::cout << "coutPSKOutputAdaptor write vector<float> array: <" << objname << "> : " << "\n";
    }

    //* write double functions
    void write(const std::string & objname, double d) {
      std::cout << "coutPSKOutputAdaptor write double: <" << objname << "> : " << d << "\n";
    }
    void write(const std::string & objname, const Dimens dimens, const double *d_array) {
      std::cout << "coutPSKOutputAdaptor write double* array: <" << objname << "> : " << "\n";
    }
    void write(const std::string & objname, const Dimens dimens, const std::vector < double >&d_array) {
      std::cout << "coutPSKOutputAdaptor write vector<double> array: <" << objname << "> : " << "\n";
    }

  };

}                               // end namespace PSK


template < class Toa > class myOutputAgent:public PSK::OutputAgent < Toa > 
{
    Field *_field;
    Grid *_grid;
    VCtopology3D *_vct;
    Collective *_col;
    int ns;
    std::vector < Particles * >_part;

    public:
    // static void convert_to_single_precision(const arr3_double&, float*, int, int, int);
    myOutputAgent(void) {;
    }

    void set_simulation_pointers(Field * field, Grid * grid, VCtopology3D * vct, Collective * col) {
        _field = field;
        _grid = grid;
        _vct = vct;
        _col = col;
    }

    void set_simulation_pointers_part(Particles * part) {
        _part.push_back(part);
    }

    void write_single_node(const std::string& path, const double* d_array) 
    {
        const int nx = _grid->getNXN() - 2;
        const int ny = _grid->getNYN() - 2;
        const int nz = _grid->getNZN() - 2;
        const size_t size = static_cast<size_t>(nx) * ny * nz;

        std::vector<float> f_array(size);
        for (size_t i = 0; i < size; ++i) 
            f_array[i] = static_cast<float>(d_array[i]);

        this->output_adaptor.write(path, PSK::Dimens(nx, ny, nz), f_array.data());
    };

    void write_single_cell_centre(const std::string& path, const double* d_array) 
    {
        const int nx = _grid->getNXC() - 2;
        const int ny = _grid->getNYC() - 2;
        const int nz = _grid->getNZC() - 2;
        const size_t size = static_cast<size_t>(nx) * ny * nz;

        std::vector<float> f_array(size);
        for (size_t i = 0; i < size; ++i) 
            f_array[i] = static_cast<float>(d_array[i]);

        this->output_adaptor.write(path, PSK::Dimens(nx, ny, nz), f_array.data());
    };

    void convert_to_single_precision(double*** input_array, float* output_array, int nx, int ny, int nz)
    {
        for (int iz = 0; iz < nz; ++iz) 
            for (int iy = 0; iy < ny; ++iy) 
                for (int ix = 0; ix < nx; ++ix) 
                {
                    size_t flat_index = static_cast<size_t>(ix) + nx * (iy + ny * iz);
                    output_array[flat_index] = static_cast<float>(input_array[ix][iy][iz]);
                }
    }

    //  void set_simulation_pointers_testpart(Particles * part) {
    //    _testpart.push_back(part);
    //  }

    //! ============================================================================================================ !//

    void output(const std::string & tag, int cycle) 
    {
        stringstream ss;
		stringstream cc;
		stringstream ii;
		ss << MPIdata::instance().get_rank();
		cc << cycle;
		const int ns = _col->getNs();
		
		if (tag.find("last_cycle", 0) != string::npos)
			this->output_adaptor.write("/last_cycle", cycle);
		
		if (tag.find("collective", 0) != string::npos) 
		{
			this->output_adaptor.write("/collective/Lx", _col->getLx());
			this->output_adaptor.write("/collective/Ly", _col->getLy());
			this->output_adaptor.write("/collective/Lz", _col->getLz());
			this->output_adaptor.write("/collective/x_center", _col->getx_center());
			this->output_adaptor.write("/collective/y_center", _col->gety_center());
			this->output_adaptor.write("/collective/z_center", _col->getz_center());
			this->output_adaptor.write("/collective/L_square", _col->getL_square());
			this->output_adaptor.write("/collective/Bx0", _col->getB0x());
			this->output_adaptor.write("/collective/By0", _col->getB0y());
			this->output_adaptor.write("/collective/Bz0", _col->getB0z());
			this->output_adaptor.write("/collective/Nxc", _col->getNxc());
			this->output_adaptor.write("/collective/Nyc", _col->getNyc());
			this->output_adaptor.write("/collective/Nzc", _col->getNzc());
			this->output_adaptor.write("/collective/Dx", _col->getDx());
			this->output_adaptor.write("/collective/Dy", _col->getDy());
			this->output_adaptor.write("/collective/Dz", _col->getDz());
			this->output_adaptor.write("/collective/Dt", _col->getDt());
			this->output_adaptor.write("/collective/Th", _col->getTh());
			this->output_adaptor.write("/collective/Ncycles", _col->getNcycles());
			this->output_adaptor.write("/collective/Ns", _col->getNs());
			this->output_adaptor.write("/collective/NsTestPart", _col->getNsTestPart());
			this->output_adaptor.write("/collective/c", _col->getC());
			this->output_adaptor.write("/collective/Smooth", _col->getSmooth());

			this->output_adaptor.write("/collective/bc/PfaceXright", _col->getBcPfaceXright());
			this->output_adaptor.write("/collective/bc/PfaceXleft", _col->getBcPfaceXleft());
			this->output_adaptor.write("/collective/bc/PfaceYright", _col->getBcPfaceYright());
			this->output_adaptor.write("/collective/bc/PfaceYleft", _col->getBcPfaceYleft());
			this->output_adaptor.write("/collective/bc/PfaceZright", _col->getBcPfaceZright());
			this->output_adaptor.write("/collective/bc/PfaceZleft", _col->getBcPfaceZleft());

			this->output_adaptor.write("/collective/bc/PHIfaceXright", _col->getBcPHIfaceXright());
			this->output_adaptor.write("/collective/bc/PHIfaceXleft", _col->getBcPHIfaceXleft());
			this->output_adaptor.write("/collective/bc/PHIfaceYright", _col->getBcPHIfaceYright());
			this->output_adaptor.write("/collective/bc/PHIfaceYleft", _col->getBcPHIfaceYleft());
			this->output_adaptor.write("/collective/bc/PHIfaceZright", _col->getBcPHIfaceZright());
			this->output_adaptor.write("/collective/bc/PHIfaceZleft", _col->getBcPHIfaceZleft());

			this->output_adaptor.write("/collective/bc/EMfaceXright", _col->getBcEMfaceXright());
			this->output_adaptor.write("/collective/bc/EMfaceXleft", _col->getBcEMfaceXleft());
			this->output_adaptor.write("/collective/bc/EMfaceYright", _col->getBcEMfaceYright());
			this->output_adaptor.write("/collective/bc/EMfaceYleft", _col->getBcEMfaceYleft());
			this->output_adaptor.write("/collective/bc/EMfaceZright", _col->getBcEMfaceZright());
			this->output_adaptor.write("/collective/bc/EMfaceZleft", _col->getBcEMfaceZleft());

			for (int i = 0; i < ns; ++i) 
			{
				stringstream ii;
				ii << i;
				this->output_adaptor.write("/collective/species_" + ii.str() + "/Npcelx", _col->getNpcelx(i));
				this->output_adaptor.write("/collective/species_" + ii.str() + "/Npcely", _col->getNpcely(i));
				this->output_adaptor.write("/collective/species_" + ii.str() + "/Npcelz", _col->getNpcelz(i));
				this->output_adaptor.write("/collective/species_" + ii.str() + "/qom", _col->getQOM(i));
				this->output_adaptor.write("/collective/species_" + ii.str() + "/uth", _col->getUth(i));
				this->output_adaptor.write("/collective/species_" + ii.str() + "/vth", _col->getVth(i));
				this->output_adaptor.write("/collective/species_" + ii.str() + "/wth", _col->getWth(i));
				this->output_adaptor.write("/collective/species_" + ii.str() + "/u0", _col->getU0(i));
				this->output_adaptor.write("/collective/species_" + ii.str() + "/v0", _col->getV0(i));
				this->output_adaptor.write("/collective/species_" + ii.str() + "/w0", _col->getW0(i));
			};

			const int nstestpart = _col->getNsTestPart();
			for (int i = 0; i < nstestpart; ++i) 
			{
				stringstream ii;
				ii << (i+ns);
				this->output_adaptor.write("/collective/testspecies_" + ii.str() + "/Npcelx", _col->getNpcelx(i+ns));
				this->output_adaptor.write("/collective/testspecies_" + ii.str() + "/Npcely", _col->getNpcely(i+ns));
				this->output_adaptor.write("/collective/testspecies_" + ii.str() + "/Npcelz", _col->getNpcelz(i+ns));
				this->output_adaptor.write("/collective/testspecies_" + ii.str() + "/qom", 	  _col->getQOM(i+ns));
				this->output_adaptor.write("/collective/testspecies_" + ii.str() + "/pitch_angle", _col->getPitchAngle(i));
				this->output_adaptor.write("/collective/testspecies_" + ii.str() + "/energy", 	   _col->getEnergy(i));
			};

		}

		if (tag.find("total_topology", 0) != string::npos) 
		{
			this->output_adaptor.write("/topology/XLEN", _vct->getXLEN());
			this->output_adaptor.write("/topology/YLEN", _vct->getYLEN());
			this->output_adaptor.write("/topology/ZLEN", _vct->getZLEN());
			this->output_adaptor.write("/topology/Nprocs", _vct->getNprocs());
			this->output_adaptor.write("/topology/periodicX", _vct->getPERIODICX());
			this->output_adaptor.write("/topology/periodicY", _vct->getPERIODICY());
			this->output_adaptor.write("/topology/periodicZ", _vct->getPERIODICZ());
		}

		if (tag.find("proc_topology", 0) != string::npos) 
		{
			int *coord = new int[3];
			coord[0] = _vct->getCoordinates(0);
			coord[1] = _vct->getCoordinates(1);
			coord[2] = _vct->getCoordinates(2);
			this->output_adaptor.write("/topology/cartesian_coord", PSK::Dimens(3), coord);
			this->output_adaptor.write("/topology/cartesian_rank", _vct->getCartesian_rank());
			this->output_adaptor.write("/topology/Xleft_neighbor", _vct->getXleft_neighbor());
			this->output_adaptor.write("/topology/Xright_neighbor", _vct->getXright_neighbor());
			this->output_adaptor.write("/topology/Yleft_neighbor", _vct->getYleft_neighbor());
			this->output_adaptor.write("/topology/Yright_neighbor", _vct->getYright_neighbor());
			this->output_adaptor.write("/topology/Zleft_neighbor", _vct->getZleft_neighbor());
			this->output_adaptor.write("/topology/Zright_neighbor", _vct->getZright_neighbor());
			delete[]coord;
		}
    }

        /****************************************************************************
        //? Tags for writing for files 

            collective
            total_topology 
            proc_topology
        
        //? Fields
            # Ball              --> Bx, By, and Bz
            # Bx (By, Bz)       --> Bx (By or Bz)
            # Eall              --> Ex, Ey, and Ez
            # Ex (Ey, Ez)       --> Ex (Ey or Ez)
            # phi               --> scalar vector

        //? Moments
            # Jall              --> Jx, Jy, and Jz (total current)
            # Jx (Jy, Jz)       --> Jx (Jy or Jz; total current along X, Y, or Z)
            # Jsall             --> Jxs, Jys, and Jzs (current densities for each species)
            # Jxs (Jys, Jzs)    --> Jxs (Jys or Jzs; current densities along X, Y, or Z)
            # rhos              --> charge densities for each species
            # pressure          --> pressure tensor for each species

        //? Particles
            # position          --> particle positions (X, Y, Z)
            # velocity          --> particle velocities (U, V, W)
            # q                 --> particle charges
            # ID                --> particle ID (note: TrackParticleID has to be set true in Collective)
                
        //? Energies
            # k_energy          --> Kinetic energy for each species
            # B_energy          --> Magnetic field energy
            # E_energy          --> Electric field energy

        ****************************************************************************/

    //! ============================================================================================================ !//

    //! Moment and Field Output
    void output_fields(const string & tag, int cycle) 
	{
		stringstream ss;
		stringstream cc;
		stringstream ii;
		ss << MPIdata::instance().get_rank();
		cc << cycle;
		const int ns = _col->getNs();
		
		if (tag.find("last_cycle", 0) != string::npos)
			this->output_adaptor.write("/last_cycle", cycle);

		//! ************************* Fields ************************* !//

        //TODO: Atm, writing in single precision involves writing the desired data to an array in double, converting that array to single, and writing this data
        //TODO: This may be further optimised - PJD
        double*** temp_double; float* temp_single;     //* Temporary array to store data in double & single precision, respectively

		//* B field (defined at nodes) is written without ghost cells
		if (tag.find("B", 0) != string::npos) 
		{
            if (_col->get_output_data_precision() == "double")
            {
                //* Double precision
                this->output_adaptor.write("/fields/Bx/cycle_" + cc.str(), PSK::Dimens(_grid->getNXN() - 2, _grid->getNYN() - 2, _grid->getNZN() - 2), _field->getBx());
                this->output_adaptor.write("/fields/By/cycle_" + cc.str(), PSK::Dimens(_grid->getNXN() - 2, _grid->getNYN() - 2, _grid->getNZN() - 2), _field->getBy());
                this->output_adaptor.write("/fields/Bz/cycle_" + cc.str(), PSK::Dimens(_grid->getNXN() - 2, _grid->getNYN() - 2, _grid->getNZN() - 2), _field->getBz());
            }
            // else if (_col->get_output_data_precision() == "single")
            // {
            //     cout << "WRITING OUTPUT DATA IN SINGLE PRECISION" << endl;

            //     for (int i = 0; i < _grid->getNXN(); i++)
            //         for (int j = 0; j < _grid->getNYN(); j++)
            //             for (int k = 0; k < _grid->getNYN(); k++)
            //                temp_double[i][j][k] = _field->getBx(i, j, k);
                        
            //     convert_to_single_precision(temp_double, temp_single, _grid->getNXN(), _grid->getNYN(), _grid->getNZN());
            //     this->output_adaptor.write("/fields/Bx/cycle_" + cc.str(), PSK::Dimens(_grid->getNXN() - 2, _grid->getNYN() - 2, _grid->getNZN() - 2), temp_single);

            //     for (int i = 0; i < _grid->getNXN(); i++)
            //         for (int j = 0; j < _grid->getNYN(); j++)
            //             for (int k = 0; k < _grid->getNYN(); k++)
            //                temp_double[i][j][k] = _field->getBy(i, j, k);
                        
            //     convert_to_single_precision(temp_double, temp_single, _grid->getNXN(), _grid->getNYN(), _grid->getNZN());
            //     this->output_adaptor.write("/fields/By/cycle_" + cc.str(), PSK::Dimens(_grid->getNXN() - 2, _grid->getNYN() - 2, _grid->getNZN() - 2), temp_single);

            //     for (int i = 0; i < _grid->getNXN(); i++)
            //         for (int j = 0; j < _grid->getNYN(); j++)
            //             for (int k = 0; k < _grid->getNYN(); k++)
            //                temp_double[i][j][k] = _field->getBz(i, j, k);
                        
            //     convert_to_single_precision(temp_double, temp_single, _grid->getNXN(), _grid->getNYN(), _grid->getNZN());
            //     this->output_adaptor.write("/fields/Bz/cycle_" + cc.str(), PSK::Dimens(_grid->getNXN() - 2, _grid->getNYN() - 2, _grid->getNZN() - 2), temp_single);
            // }
		}
		else if (tag.find("Bx", 0) != string::npos)
			this->output_adaptor.write("/fields/Bx/cycle_" + cc.str(), PSK::Dimens(_grid->getNXN() - 2, _grid->getNYN() - 2, _grid->getNZN() - 2), _field->getBx());
		
		else if (tag.find("By", 0) != string::npos)
			this->output_adaptor.write("/fields/By/cycle_" + cc.str(), PSK::Dimens(_grid->getNXN() - 2, _grid->getNYN() - 2, _grid->getNZN() - 2), _field->getBy());

		else if (tag.find("Bz", 0) != string::npos)
			this->output_adaptor.write("/fields/Bz/cycle_" + cc.str(), PSK::Dimens(_grid->getNXN() - 2, _grid->getNYN() - 2, _grid->getNZN() - 2), _field->getBz());

        //* B field (defined at cell centres) is written without ghost cells
		if (tag.find("B_c", 0) != string::npos) 
		{
			this->output_adaptor.write("/fields/Bxc/cycle_" + cc.str(), PSK::Dimens(_grid->getNXC() - 2, _grid->getNYC() - 2, _grid->getNZC() - 2), _field->getBxc());
			this->output_adaptor.write("/fields/Byc/cycle_" + cc.str(), PSK::Dimens(_grid->getNXC() - 2, _grid->getNYC() - 2, _grid->getNZC() - 2), _field->getByc());
			this->output_adaptor.write("/fields/Bzc/cycle_" + cc.str(), PSK::Dimens(_grid->getNXC() - 2, _grid->getNYC() - 2, _grid->getNZC() - 2), _field->getBzc());
		}

        if (tag.find("B_ext", 0) != string::npos) 
        {
            this->output_adaptor.write("/fields/Bx_ext/cycle_" + cc.str(), PSK::Dimens(_grid->getNXN() - 2, _grid->getNYN() - 2, _grid->getNZN() - 2), _field->getBx_ext());
            this->output_adaptor.write("/fields/By_ext/cycle_" + cc.str(), PSK::Dimens(_grid->getNXN() - 2, _grid->getNYN() - 2, _grid->getNZN() - 2), _field->getBy_ext());
            this->output_adaptor.write("/fields/Bz_ext/cycle_" + cc.str(), PSK::Dimens(_grid->getNXN() - 2, _grid->getNYN() - 2, _grid->getNZN() - 2), _field->getBz_ext());
        }


    	//* E field (defined at nodes) is written without ghost cells
		if (tag.find("E", 0) != string::npos) 
		{
            if (_col->get_output_data_precision() == "double")
            {
                //* Double precision
                this->output_adaptor.write("/fields/Ex/cycle_" + cc.str(), PSK::Dimens(_grid->getNXN() - 2, _grid->getNYN() - 2, _grid->getNZN() - 2), _field->getEx());
                this->output_adaptor.write("/fields/Ey/cycle_" + cc.str(), PSK::Dimens(_grid->getNXN() - 2, _grid->getNYN() - 2, _grid->getNZN() - 2), _field->getEy());
                this->output_adaptor.write("/fields/Ez/cycle_" + cc.str(), PSK::Dimens(_grid->getNXN() - 2, _grid->getNYN() - 2, _grid->getNZN() - 2), _field->getEz());
            }
            // else if (_col->get_output_data_precision() == "single")
            // {
            //     cout << "WRITING OUTPUT DATA IN SINGLE PRECISION" << endl;

            //     for (int i = 0; i < _grid->getNXN(); i++)
            //         for (int j = 0; j < _grid->getNYN(); j++)
            //             for (int k = 0; k < _grid->getNYN(); k++)
            //                temp_double[i][j][k] = _field->getEx(i, j, k);
                        
            //     convert_to_single_precision(temp_double, temp_single, _grid->getNXN(), _grid->getNYN(), _grid->getNZN());
            //     this->output_adaptor.write("/fields/Bx/cycle_" + cc.str(), PSK::Dimens(_grid->getNXN() - 2, _grid->getNYN() - 2, _grid->getNZN() - 2), temp_single);

            //     for (int i = 0; i < _grid->getNXN(); i++)
            //         for (int j = 0; j < _grid->getNYN(); j++)
            //             for (int k = 0; k < _grid->getNYN(); k++)
            //                temp_double[i][j][k] = _field->getEy(i, j, k);
                        
            //     convert_to_single_precision(temp_double, temp_single, _grid->getNXN(), _grid->getNYN(), _grid->getNZN());
            //     this->output_adaptor.write("/fields/By/cycle_" + cc.str(), PSK::Dimens(_grid->getNXN() - 2, _grid->getNYN() - 2, _grid->getNZN() - 2), temp_single);

            //     for (int i = 0; i < _grid->getNXN(); i++)
            //         for (int j = 0; j < _grid->getNYN(); j++)
            //             for (int k = 0; k < _grid->getNYN(); k++)
            //                temp_double[i][j][k] = _field->getEz(i, j, k);
                        
            //     convert_to_single_precision(temp_double, temp_single, _grid->getNXN(), _grid->getNYN(), _grid->getNZN());
            //     this->output_adaptor.write("/fields/Bz/cycle_" + cc.str(), PSK::Dimens(_grid->getNXN() - 2, _grid->getNYN() - 2, _grid->getNZN() - 2), temp_single);
            // }
		}
		else if (tag.find("Ex", 0) != string::npos)
			this->output_adaptor.write("/fields/Ex/cycle_" + cc.str(), PSK::Dimens(_grid->getNXN() - 2, _grid->getNYN() - 2, _grid->getNZN() - 2), _field->getEx());

		else if (tag.find("Ey", 0) != string::npos)
			this->output_adaptor.write("/fields/Ey/cycle_" + cc.str(), PSK::Dimens(_grid->getNXN() - 2, _grid->getNYN() - 2, _grid->getNZN() - 2), _field->getEy());

		else if (tag.find("Ez", 0) != string::npos)
			this->output_adaptor.write("/fields/Ez/cycle_" + cc.str(), PSK::Dimens(_grid->getNXN() - 2, _grid->getNYN() - 2, _grid->getNZN() - 2), _field->getEz());


		//* PHI (defined at cell centres) is written without ghost cells 
		if (tag.find("phi", 0) != string::npos)
			this->output_adaptor.write("/potentials/phi/cycle_" + cc.str(), PSK::Dimens(_grid->getNXC() - 2, _grid->getNYC() - 2, _grid->getNZC() - 2), _field->getPHI());


		//! ************************* Moments ************************* !//

		//* J (total current, defined at nodes) is written without ghost cells
		if (tag.find("J", 0) != string::npos)
		{
			// _field->sumOverSpeciesJ();
			this->output_adaptor.write("/moments/Jx/cycle_" + cc.str(), PSK::Dimens(_grid->getNXN() - 2, _grid->getNYN() - 2, _grid->getNZN() - 2), _field->getJxh());
			this->output_adaptor.write("/moments/Jy/cycle_" + cc.str(), PSK::Dimens(_grid->getNXN() - 2, _grid->getNYN() - 2, _grid->getNZN() - 2), _field->getJyh());
			this->output_adaptor.write("/moments/Jz/cycle_" + cc.str(), PSK::Dimens(_grid->getNXN() - 2, _grid->getNYN() - 2, _grid->getNZN() - 2), _field->getJzh());
		}
		else if (tag.find("Jx", 0) != string::npos) 
		{
			// _field->sumOverSpeciesJ();
			this->output_adaptor.write("/moments/Jx/cycle_" + cc.str(), PSK::Dimens(_grid->getNXN() - 2, _grid->getNYN() - 2, _grid->getNZN() - 2), _field->getJxh());
		}
		else if (tag.find("Jy", 0) != string::npos) 
		{
			// _field->sumOverSpeciesJ();
			this->output_adaptor.write("/moments/Jy/cycle_" + cc.str(), PSK::Dimens(_grid->getNXN() - 2, _grid->getNYN() - 2, _grid->getNZN() - 2), _field->getJyh());
		}
		else if (tag.find("Jz", 0) != string::npos) 
		{
			// _field->sumOverSpeciesJ();
			this->output_adaptor.write("/moments/Jz/cycle_" + cc.str(), PSK::Dimens(_grid->getNXN() - 2, _grid->getNYN() - 2, _grid->getNZN() - 2), _field->getJzh());
		}

		//* Js (current for each species, defined at nodes) is written without ghost cells
		if (tag.find("J_s", 0) != string::npos) 
		{
			for (int i = 0; i < ns; ++i)
			{
				stringstream ii;
				ii << i;
				this->output_adaptor.write("/moments/species_" + ii.str() + "/Jx/cycle_" + cc.str(), PSK::Dimens(_grid->getNXN() - 2, _grid->getNYN() - 2, _grid->getNZN() - 2), i, _field->getJxs());
				this->output_adaptor.write("/moments/species_" + ii.str() + "/Jy/cycle_" + cc.str(), PSK::Dimens(_grid->getNXN() - 2, _grid->getNYN() - 2, _grid->getNZN() - 2), i, _field->getJys());
				this->output_adaptor.write("/moments/species_" + ii.str() + "/Jz/cycle_" + cc.str(), PSK::Dimens(_grid->getNXN() - 2, _grid->getNYN() - 2, _grid->getNZN() - 2), i, _field->getJzs());
			}
		}

		//* rhos (charge density for each species, defined at nodes) is written without ghost cells
		if (tag.find("rho_s", 0) != string::npos) 
		{
			for (int i = 0; i < ns; ++i)
			{
				stringstream ii;
				ii << i;
				this->output_adaptor.write("/moments/species_" + ii.str() + "/rho/cycle_" + cc.str(), PSK::Dimens(_grid->getNXN() - 2, _grid->getNYN() - 2, _grid->getNZN() - 2), i, _field->getRHOns());
			}
		}

        //* rhos (overall charge density) is written without ghost cells
        if (tag.find("rho", 0) != string::npos) 
        {
            this->output_adaptor.write("/moments/rho/cycle_" + cc.str(), PSK::Dimens(_grid->getNXN() - 2, _grid->getNYN() - 2, _grid->getNZN() - 2), _field->getRHOn());
        }

		//* Pressure tensor (for each species, defined at nodes) is written without ghost cells
		if (tag.find("pressure", 0) != string::npos) 
		{
			for (int i = 0; i < ns; ++i) 
			{
				stringstream ii;
				ii << i;
				this->output_adaptor.write("/moments/species_" + ii.str() + "/pXX/cycle_" + cc.str(), PSK::Dimens(_grid->getNXN() - 2, _grid->getNYN() - 2, _grid->getNZN() - 2), i, _field->getpXXsn());
				this->output_adaptor.write("/moments/species_" + ii.str() + "/pXY/cycle_" + cc.str(), PSK::Dimens(_grid->getNXN() - 2, _grid->getNYN() - 2, _grid->getNZN() - 2), i, _field->getpXYsn());
				this->output_adaptor.write("/moments/species_" + ii.str() + "/pXZ/cycle_" + cc.str(), PSK::Dimens(_grid->getNXN() - 2, _grid->getNYN() - 2, _grid->getNZN() - 2), i, _field->getpXZsn());
				this->output_adaptor.write("/moments/species_" + ii.str() + "/pYY/cycle_" + cc.str(), PSK::Dimens(_grid->getNXN() - 2, _grid->getNYN() - 2, _grid->getNZN() - 2), i, _field->getpYYsn());
				this->output_adaptor.write("/moments/species_" + ii.str() + "/pYZ/cycle_" + cc.str(), PSK::Dimens(_grid->getNXN() - 2, _grid->getNYN() - 2, _grid->getNZN() - 2), i, _field->getpYZsn());
				this->output_adaptor.write("/moments/species_" + ii.str() + "/pZZ/cycle_" + cc.str(), PSK::Dimens(_grid->getNXN() - 2, _grid->getNYN() - 2, _grid->getNZN() - 2), i, _field->getpZZsn());
			}
		}

        //* Energy flux density (for each species, defined at nodes) is written without ghost cells
		if (tag.find("E_flux", 0) != string::npos)
		{
			for (int i = 0; i < ns; ++i) 
			{
				stringstream ii;
				ii << i;
				this->output_adaptor.write("/moments/species_" + ii.str() + "/EFx/cycle_" + cc.str(), PSK::Dimens(_grid->getNXN() - 2, _grid->getNYN() - 2, _grid->getNZN() - 2), i, _field->getEFxs());
				this->output_adaptor.write("/moments/species_" + ii.str() + "/EFy/cycle_" + cc.str(), PSK::Dimens(_grid->getNXN() - 2, _grid->getNYN() - 2, _grid->getNZN() - 2), i, _field->getEFys());
				this->output_adaptor.write("/moments/species_" + ii.str() + "/EFz/cycle_" + cc.str(), PSK::Dimens(_grid->getNXN() - 2, _grid->getNYN() - 2, _grid->getNZN() - 2), i, _field->getEFzs());
            }
        }

        //* Heat flux tensor (for each species, defined at nodes) is written without ghost cells
		if (tag.find("H_flux", 0) != string::npos) 
		{
			for (int i = 0; i < ns; ++i) 
			{
                // this->output_adaptor.write("/moments/species_" + ii.str() + "/Qxxx/cycle_" + cc.str(), PSK::Dimens(_grid->getNXN() - 2, _grid->getNYN() - 2, _grid->getNZN() - 2), i, _field->getQxxxs());
                // this->output_adaptor.write("/moments/species_" + ii.str() + "/Qxxy/cycle_" + cc.str(), PSK::Dimens(_grid->getNXN() - 2, _grid->getNYN() - 2, _grid->getNZN() - 2), i, _field->getQxxys());
                // this->output_adaptor.write("/moments/species_" + ii.str() + "/Qxyy/cycle_" + cc.str(), PSK::Dimens(_grid->getNXN() - 2, _grid->getNYN() - 2, _grid->getNZN() - 2), i, _field->getQxyys());
                // this->output_adaptor.write("/moments/species_" + ii.str() + "/Qxzz/cycle_" + cc.str(), PSK::Dimens(_grid->getNXN() - 2, _grid->getNYN() - 2, _grid->getNZN() - 2), i, _field->getQxzzs());
                // this->output_adaptor.write("/moments/species_" + ii.str() + "/Qyyy/cycle_" + cc.str(), PSK::Dimens(_grid->getNXN() - 2, _grid->getNYN() - 2, _grid->getNZN() - 2), i, _field->getQyyys());
                // this->output_adaptor.write("/moments/species_" + ii.str() + "/Qyzz/cycle_" + cc.str(), PSK::Dimens(_grid->getNXN() - 2, _grid->getNYN() - 2, _grid->getNZN() - 2), i, _field->getQyzzs());
                // this->output_adaptor.write("/moments/species_" + ii.str() + "/Qzzz/cycle_" + cc.str(), PSK::Dimens(_grid->getNXN() - 2, _grid->getNYN() - 2, _grid->getNZN() - 2), i, _field->getQzzzs());
                // this->output_adaptor.write("/moments/species_" + ii.str() + "/Qxyz/cycle_" + cc.str(), PSK::Dimens(_grid->getNXN() - 2, _grid->getNYN() - 2, _grid->getNZN() - 2), i, _field->getQxyzs());
                // this->output_adaptor.write("/moments/species_" + ii.str() + "/Qxxz/cycle_" + cc.str(), PSK::Dimens(_grid->getNXN() - 2, _grid->getNYN() - 2, _grid->getNZN() - 2), i, _field->getQxxzs());
                // this->output_adaptor.write("/moments/species_" + ii.str() + "/Qyyz/cycle_" + cc.str(), PSK::Dimens(_grid->getNXN() - 2, _grid->getNYN() - 2, _grid->getNZN() - 2), i, _field->getQyyzs());
            }
		}

        //! ************************* Energies ************************* !//

		//* Kinetic energy for species s (normalized to q)
		if (tag.find("K_energy", 0) != string::npos) 
		{
			double K_en;
			for (int i = 0; i < ns; ++i) 
			{
				K_en = 0;
				stringstream ii;
				ii << i;
				double qom = fabs(_col->getQOM(i));
				
				for (int n = 0; n < _part[i]->getNOP(); n++) 
				{
					K_en += fabs(_part[i]->getQ(n)) * _part[i]->getU(n) * _part[i]->getU(n);
					K_en += fabs(_part[i]->getQ(n)) * _part[i]->getV(n) * _part[i]->getV(n);
					K_en += fabs(_part[i]->getQ(n)) * _part[i]->getW(n) * _part[i]->getW(n);
				}
				K_en *= 0.5 / qom;
				
				this->output_adaptor.write("/energy/kinetic/species_" + ii.str() + "/cycle_" + cc.str(), K_en);
			}
		}

		//* Magnetic field energy (nodes) is computed without ghost cells
		if (tag.find("B_energy", 0) != string::npos) 
		{
			double B_en = 0.0;
			for (int i = 1; i < _grid->getNXN(); i++)
				for (int j = 1; j < _grid->getNYN(); j++)
					for (int k = 1; k < _grid->getNYN(); k++)
						B_en += _field->getBx(i, j, k) * _field->getBx(i, j, k) + _field->getBy(i, j, k) * _field->getBy(i, j, k) + _field->getBz(i, j, k) * _field->getBz(i, j, k);
			B_en = B_en / 32 / atan(1.0); //! here there should be a getfourPI for the right value of pi 
			
			this->output_adaptor.write("/energy/magnetic/cycle_" + cc.str(), B_en);
		}

		//* Electric field energy (nodes) is computed without ghost cells
		if (tag.find("E_energy", 0) != string::npos) 
		{
            double E_en = 0.0;
            for (int i = 1; i < _grid->getNXN(); i++)
                for (int j = 1; j < _grid->getNYN(); j++)
                    for (int k = 1; k < _grid->getNYN(); k++)
                        E_en += _field->getEx(i, j, k) * _field->getEx(i, j, k) + _field->getEy(i, j, k) * _field->getEy(i, j, k) + _field->getEz(i, j, k) * _field->getEz(i, j, k);
            E_en = E_en / 32 / atan(1.0); //! here there should be a getfourPI for the right value of pi 
            
            this->output_adaptor.write("/energy/electric/cycle_" + cc.str(), E_en);
		}
    }

  	//! ============================================================================================================ !//

    //! Particle Output
    void output_particles(const string & tag, int cycle) 
    {
        stringstream cc;
        cc << cycle;
        const int ns = _col->getNs();
        const int nstestpart = _col->getNsTestPart();

        //! ************************* Positions ************************* !//

        if (tag.find("position", 0) != string::npos) 
        {
            for (int i = 0; i < ns; ++i) 
            {
                stringstream ii;
                ii << i;
                this->output_adaptor.write("/particles/species_" + ii.str() + "/x/cycle_" + cc.str(), PSK::Dimens(_part[i]->getNOP()), _part[i]->getXall());
                this->output_adaptor.write("/particles/species_" + ii.str() + "/y/cycle_" + cc.str(), PSK::Dimens(_part[i]->getNOP()), _part[i]->getYall());
                this->output_adaptor.write("/particles/species_" + ii.str() + "/z/cycle_" + cc.str(), PSK::Dimens(_part[i]->getNOP()), _part[i]->getZall());
            }
        }

        //* Test Particles
        if (tag.find("testpartpos", 0) != string::npos) 
        {
            for (int i = 0; i < nstestpart; ++i) 
            {
                stringstream ii;
                ii << (_part[i+ns]->get_species_num());
                this->output_adaptor.write("/testparticles/species_" + ii.str() + "/x/cycle_" + cc.str(), PSK::Dimens(_part[i+ns]->getNOP()), _part[i+ns]->getXall());
                this->output_adaptor.write("/testparticles/species_" + ii.str() + "/y/cycle_" + cc.str(), PSK::Dimens(_part[i+ns]->getNOP()), _part[i+ns]->getYall());
                this->output_adaptor.write("/testparticles/species_" + ii.str() + "/z/cycle_" + cc.str(), PSK::Dimens(_part[i+ns]->getNOP()), _part[i+ns]->getZall());
            }
        }

        //! ************************* Velocities ************************* !//

        if (tag.find("velocity", 0) != string::npos) 
        {
            for (int i = 0; i < ns; ++i) 
            {
                stringstream ii;
                ii << i;
                this->output_adaptor.write("/particles/species_" + ii.str() + "/u/cycle_" + cc.str(), PSK::Dimens(_part[i]->getNOP()), _part[i]->getUall());
                this->output_adaptor.write("/particles/species_" + ii.str() + "/v/cycle_" + cc.str(), PSK::Dimens(_part[i]->getNOP()), _part[i]->getVall());
                this->output_adaptor.write("/particles/species_" + ii.str() + "/w/cycle_" + cc.str(), PSK::Dimens(_part[i]->getNOP()), _part[i]->getWall());
            }
        }
        
        //* Test Particles
        if (tag.find("testpartvel", 0) != string::npos) 
        {
            for (int i = 0; i < nstestpart; ++i) 
            {
                stringstream ii;
                ii << (_part[i+ns]->get_species_num());
                this->output_adaptor.write("/testparticles/species_" + ii.str() + "/u/cycle_" + cc.str(), PSK::Dimens(_part[i+ns]->getNOP()), _part[i+ns]->getUall());
                this->output_adaptor.write("/testparticles/species_" + ii.str() + "/v/cycle_" + cc.str(), PSK::Dimens(_part[i+ns]->getNOP()), _part[i+ns]->getVall());
                this->output_adaptor.write("/testparticles/species_" + ii.str() + "/w/cycle_" + cc.str(), PSK::Dimens(_part[i+ns]->getNOP()), _part[i+ns]->getWall());
            }
        }

        //! ************************* Charge ************************* !//

        if (tag.find("q", 0) != string::npos) 
        {
            for (int i = 0; i < ns; ++i) 
            {
                stringstream ii;
                ii << i;
                this->output_adaptor.write("/particles/species_" + ii.str() + "/q/cycle_" + cc.str(), PSK::Dimens(_part[i]->getNOP()), _part[i]->getQall());
            }
        }

        //* Test Particles
        if (tag.find("testpartcharge", 0) != string::npos) 
        {
            for (int i = 0; i < nstestpart; ++i)
            {
                stringstream ii;
                ii <<  (_part[i+ns]->get_species_num());
                this->output_adaptor.write("/testparticles/species_" + ii.str() + "/q/cycle_" + cc.str(), PSK::Dimens(_part[i+ns]->getNOP()), _part[i+ns]->getQall());
            }
        }

        //! ************************* ID ************************* !//

        if (tag.find("ID", 0) != string::npos) 
        {
            for (int i = 0; i < ns; ++i) 
            {
                stringstream ii;
                ii << i;
                this->output_adaptor.write("/particles/species_" + ii.str() + "/ID/cycle_" + cc.str(), PSK::Dimens(_part[i]->getNOP()), _part[i]->getParticleIDall());
            }
        }

        //* Test Particles
        if (tag.find("testparttag", 0) != string::npos) 
        {
            for (int i = 0; i < nstestpart; ++i) 
            {
                stringstream ii;
                ii <<  (_part[i+ns]->get_species_num());
                this->output_adaptor.write("/testparticles/species_" + ii.str() + "/ID/cycle_" + cc.str(), PSK::Dimens(_part[i+ns]->getNOP()), _part[i+ns]->getParticleIDall());
            }
        }
    }

    //! ============================================================================================================ !//

    //! Downsampled Particles
    void output_particles_DS(const string & tag, int cycle, int sample) 
    {
        stringstream cc;
        cc << cycle;
        const int ns = _col->getNs();

        //* Downsampled Particles
        if (tag.find("position_DS", 0) != string::npos) 
        {
            // int sample = _col->getParticlesDownsampleFactor();
            std::vector < double >X, Y, Z;
            for (int i = 0; i < ns; ++i) 
            {
                stringstream ii;
                ii << i;
                const int num_samples = _part[i]->getNOP()/sample;
                X.reserve(num_samples);
                Y.reserve(num_samples);
                Z.reserve(num_samples);
                
                for (int n = 0; n < _part[i]->getNOP(); n += sample) 
                {
                    X.push_back(_part[i]->getX(n));
                    Y.push_back(_part[i]->getY(n));
                    Z.push_back(_part[i]->getZ(n));
                }

                this->output_adaptor.write("/particles_DS/species_" + ii.str() + "/x/cycle_" + cc.str(), PSK::Dimens(X.size()), X);
                this->output_adaptor.write("/particles_DS/species_" + ii.str() + "/y/cycle_" + cc.str(), PSK::Dimens(Y.size()), Y);
                this->output_adaptor.write("/particles_DS/species_" + ii.str() + "/z/cycle_" + cc.str(), PSK::Dimens(Z.size()), Z);
            }
        }
                
        if (tag.find("velocity_DS", 0) != string::npos) 
        {
            // int sample = _col->getParticlesDownsampleFactor();
            std::vector < double >U, V, W;
            for (int i = 0; i < ns; ++i) 
            {
                stringstream ii;
                ii << i;
                const int num_samples = _part[i]->getNOP()/sample;
                U.reserve(num_samples);
                V.reserve(num_samples);
                W.reserve(num_samples);
                
                for (int n = 0; n < _part[i]->getNOP(); n += sample) 
                {
                    U.push_back(_part[i]->getU(n));
                    V.push_back(_part[i]->getV(n));
                    W.push_back(_part[i]->getW(n));
                }
                
                this->output_adaptor.write("/particles_DS/species_" + ii.str() + "/u/cycle_" + cc.str(), PSK::Dimens(U.size()), U);
                this->output_adaptor.write("/particles_DS/species_" + ii.str() + "/v/cycle_" + cc.str(), PSK::Dimens(V.size()), V);
                this->output_adaptor.write("/particles_DS/species_" + ii.str() + "/w/cycle_" + cc.str(), PSK::Dimens(W.size()), W);
            }
        }

        //* Downsampled Particles
        if (tag.find("q_DS", 0) != string::npos) 
        {
            // int sample = _col->getParticlesDownsampleFactor();
            std::vector < double >Q;
            for (int i = 0; i < ns; ++i) 
            {
                stringstream ii;
                ii << i;
                const int num_samples = _part[i]->getNOP()/sample;
                Q.reserve(num_samples);
                
                for (int n = 0; n < _part[i]->getNOP(); n += sample)
                    Q.push_back(_part[i]->getQ(n));
                
                this->output_adaptor.write("/particles_DS/species_" + ii.str() + "/q/cycle_" + cc.str(), PSK::Dimens(Q.size()), Q);
            }
        }

        //* Downsampled Particles
        if (tag.find("ID_DS", 0) != string::npos) 
        {
            // int sample = _col->getParticlesDownsampleFactor();
            std::vector <double>ID;
            for (int i = 0; i < ns; ++i) 
            {
                stringstream ii;
                ii << i;
                const double* pclID = _part[i]->getParticleIDall();
                const int num_samples = _part[i]->getNOP()/sample;
                ID.reserve(num_samples);

                for (int n = 0; n < _part[i]->getNOP(); n += sample)
                    ID.push_back(pclID[n]);
                
                this->output_adaptor.write("/particles_DS/species_" + ii.str() + "/ID/cycle_" + cc.str(), PSK::Dimens(ID.size()), &ID[0]);
            }
        }

    }

};

#endif // _PSK_OUTPUT_H_
