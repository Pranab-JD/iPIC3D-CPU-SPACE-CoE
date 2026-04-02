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
#include "hdf5.h"

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

    virtual void output_fields(const std::string & tag, int cycle, string precision) = 0;
    virtual void output_particles(const std::string & tag, int cycle, string precision) = 0;
    virtual void output_particles_DS(const std::string & tag, int cycle, int sample, string precision) = 0;
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
    virtual void output_fields(const std::string & tag, int cycle, string precision) = 0;
    virtual void output_particles(const std::string & tag, int cycle, string precision) = 0;
    virtual void output_particles_DS(const std::string & tag, int cycle, int sample, string precision) = 0;

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

    void output_fields(const std::string & tag, int cycle, string precision) 
    {
        typename std::list < OutputAgentBase * >::iterator p = agents_list.begin();
        while (p != agents_list.end())
            (*p++)->output_fields(tag, cycle, precision);
    }

    void output_particles(const std::string & tag, int cycle, string precision) 
    {
        typename std::list < OutputAgentBase * >::iterator p = agents_list.begin();
        while (p != agents_list.end())
            (*p++)->output_particles(tag, cycle, precision);
    }

    void output_particles_DS(const std::string & tag, int cycle, int sample, string precision) 
    {
        typename std::list < OutputAgentBase * >::iterator p = agents_list.begin();
        while (p != agents_list.end())
            (*p++)->output_particles_DS(tag, cycle, sample, precision);
    }

  };

    class coutOutputAdaptor:public OutputAdaptor 
    {

    public:
        coutOutputAdaptor(void) {;} 
        
        void open(const std::string & outf) 
        {
            std::cout << "coutPSKOutputAdaptor open() file: " << outf << "\n";
        }
        void close(void) 
        {
            std::cout << "coutPSKOutputAdaptor close()\n";
        }

        //* write int functions
        void write(const std::string & objname, int i) 
        {
            std::cout << "coutPSKOutputAdaptor write int: <" << objname << "> : " << i << "\n";
        }

        void write(const std::string & objname, const Dimens dimens, const int *i_array) 
        {
            std::cout << "coutPSKOutputAdaptor write int* array: <" << objname << "> : " << "\n";
        }
        void write(const std::string & objname, const Dimens dimens, const std::vector < int >&i_array) 
        {
            std::cout << "coutPSKOutputAdaptor write vector<int> array: <" << objname << "> : " << "\n";
        }

        //* write float functions
        void write(const std::string & objname, float f) 
        {
            std::cout << "coutPSKOutputAdaptor write float: <" << objname << "> : " << f << "\n";
        }
        void write(const std::string & objname, const Dimens dimens, const float *f_array) 
        {
            std::cout << "coutPSKOutputAdaptor write float* array: <" << objname << "> : " << "\n";
        }
        void write(const std::string & objname, const Dimens dimens, const std::vector < float >&f_array) 
        {
            std::cout << "coutPSKOutputAdaptor write vector<float> array: <" << objname << "> : " << "\n";
        }

        //* write double functions
        void write(const std::string & objname, double d) 
        {
            std::cout << "coutPSKOutputAdaptor write double: <" << objname << "> : " << d << "\n";
        }
        void write(const std::string & objname, const Dimens dimens, const double *d_array) 
        {
            std::cout << "coutPSKOutputAdaptor write double* array: <" << objname << "> : " << "\n";
        }
        void write(const std::string & objname, const Dimens dimens, const std::vector < double >&d_array) 
        {
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
    myOutputAgent(void) {;}

    bool contains_tag(const std::string& taglist, const std::string& target) 
    {
        std::istringstream iss(taglist);
        std::string token;
        while (iss >> token) 
        {
            if (token == target) 
            {
                return true;
            }
        }
        return false;
    }

    void set_simulation_pointers(Field * field, Grid * grid, VCtopology3D * vct, Collective * col) 
    {
        _field = field;
        _grid = grid;
        _vct = vct;
        _col = col;
    }

    void set_simulation_pointers_part(Particles * part) 
    {
        _part.push_back(part);
    }

    void delete_if_exists(hid_t file_id, const std::string& path)
    {
        H5Ldelete(file_id, path.c_str(), H5P_DEFAULT);
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

    void convert_to_single_precision(array3_double& input_array, std::vector<float>& output_array, int nx, int ny, int nz)
    {
        const int nx_inner = nx - 2;
        const int ny_inner = ny - 2;
        const int nz_inner = nz - 2;

        output_array.resize(static_cast<size_t>(nx_inner * ny_inner * nz_inner));

        for (int i = 1; i < nx - 1; ++i)
            for (int j = 1; j < ny - 1; ++j)
                for (int k = 1; k < nz - 1; ++k)
                {
                    // Flattened index in the *inner* array (not original nx/ny/nz)
                    size_t flat_index = (i - 1) * ny_inner * nz_inner + (j - 1) * nz_inner + (k - 1);
                    output_array[flat_index] = static_cast<float>(input_array.get(i, j, k));
                }
    }

    void compare_double_and_single_precision(const array3_double& input_array, const std::vector<float>& output_array,
                                            int nx, int ny, int nz, double tolerance = 1e-5)
    {
        int error_count = 0;

        for (int iz = 0; iz < nz; iz++)
            for (int iy = 0; iy < ny; iy++)
                for (int ix = 0; ix < nx; ix++)
                {
                    size_t flat_index = static_cast<size_t>(ix + nx * (iy + ny * iz));
                    double original_value = input_array.get(ix, iy, iz);
                    float converted_value = output_array[flat_index];
                    double diff = std::abs(original_value - static_cast<double>(converted_value));

                    if (diff > tolerance)
                    {
                        std::cout << "Mismatch at (" << ix << ", " << iy << ", " << iz << "): "
                                << "double=" << original_value << ", float=" << converted_value
                                << ", diff=" << diff << "\n";
                        error_count++;
                    }
                }

        if (error_count == 0)
            std::cout << "All values match within tolerance (" << tolerance << ")\n";
        else
            std::cout << "Total mismatches: " << error_count << "\n";
    }

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

    //! ============================================================================================================ !//

    //! Moment and Field Output
    void output_fields(const string & tag, int cycle, string precision) 
	{
		stringstream ss;
		stringstream cc;
		stringstream ii;
		ss << MPIdata::instance().get_rank();
		cc << cycle;
		const int ns = _col->getNs();

        const int nxc = _grid->getNXC();
        const int nyc = _grid->getNYC();
        const int nzc = _grid->getNZC();
        const int nxn = _grid->getNXN();
        const int nyn = _grid->getNYN();
        const int nzn = _grid->getNZN();
		
		if (tag.find("last_cycle", 0) != string::npos)
			this->output_adaptor.write("/last_cycle", cycle);

		//! ************************* Fields ************************* !//
        //TODO: Atm, writing in single precision involves writing the desired data to an array in double, converting that array to single, and writing this data
        //TODO: This may be further optimised - PJD

        //* Temporary arrays to store fields and moments in single & double precision
        array3_double temp_double(_grid->getNXN(), _grid->getNYN(), _grid->getNZN()); 
        vector<float> field_single ((_grid->getNXN() - 2) *  (_grid->getNYN() - 2) * (_grid->getNZN() - 2));

		//* B field (defined at nodes) is written without ghost cells
        if (contains_tag(tag, "B"))
		{
            const PSK::Dimens b_dims(_grid->getNXN() - 2, _grid->getNYN() - 2, _grid->getNZN() - 2);
            const std::string bx_path = "/fields/Bx/cycle_" + cc.str();
            const std::string by_path = "/fields/By/cycle_" + cc.str();
            const std::string bz_path = "/fields/Bz/cycle_" + cc.str();
            hid_t fid = this->output_adaptor.get_file_id();

            if (precision == "DOUBLE")
            {
                delete_if_exists(fid, bx_path);
                this->output_adaptor.write(bx_path, b_dims, _field->getBx());

                delete_if_exists(fid, by_path);
                this->output_adaptor.write(by_path, b_dims, _field->getBy());

                delete_if_exists(fid, bz_path);
                this->output_adaptor.write(bz_path, b_dims, _field->getBz());
            }
            else if (precision == "SINGLE")
            {
                for (int i = 0; i < _grid->getNXN(); i++)
                    for (int j = 0; j < _grid->getNYN(); j++)
                        for (int k = 0; k < _grid->getNZN(); k++)
                            temp_double.set(i, j, k, _field->getBx(i, j, k));

                convert_to_single_precision(temp_double, field_single, _grid->getNXN(), _grid->getNYN(), _grid->getNZN());
                delete_if_exists(fid, bx_path);
                this->output_adaptor.write(bx_path, b_dims, field_single);

                for (int i = 0; i < _grid->getNXN(); i++)
                    for (int j = 0; j < _grid->getNYN(); j++)
                        for (int k = 0; k < _grid->getNZN(); k++)
                            temp_double.set(i, j, k, _field->getBy(i, j, k));
                        
                convert_to_single_precision(temp_double, field_single, _grid->getNXN(), _grid->getNYN(), _grid->getNZN());
                delete_if_exists(fid, by_path);
                this->output_adaptor.write(by_path, b_dims, field_single);
                
                for (int i = 0; i < _grid->getNXN(); i++)
                    for (int j = 0; j < _grid->getNYN(); j++)
                        for (int k = 0; k < _grid->getNZN(); k++)
                            temp_double.set(i, j, k, _field->getBz(i, j, k));
                        
                convert_to_single_precision(temp_double, field_single, _grid->getNXN(), _grid->getNYN(), _grid->getNZN());
                delete_if_exists(fid, bz_path);
                this->output_adaptor.write("/fields/Bz/cycle_" + cc.str(), PSK::Dimens(_grid->getNXN() - 2, _grid->getNYN() - 2, _grid->getNZN() - 2), field_single);
            }
		}

        //* B field (defined at cell centres) is written without ghost cells
        if (contains_tag(tag, "B_c"))
        {
            const PSK::Dimens bc_dims(_grid->getNXC() - 2, _grid->getNYC() - 2, _grid->getNZC() - 2);
            const std::string bxc_path = "/fields/Bxc/cycle_" + cc.str();
            const std::string byc_path = "/fields/Byc/cycle_" + cc.str();
            const std::string bzc_path = "/fields/Bzc/cycle_" + cc.str();
            hid_t fid = this->output_adaptor.get_file_id();

            delete_if_exists(fid, bxc_path);
            this->output_adaptor.write(bxc_path, bc_dims, _field->getBxc());

            delete_if_exists(fid, byc_path);
            this->output_adaptor.write(byc_path, bc_dims, _field->getByc());

            delete_if_exists(fid, bzc_path);
            this->output_adaptor.write(bzc_path, bc_dims, _field->getBzc());
        }

        if (contains_tag(tag, "divergence"))
        {
            const PSK::Dimens div_dims(_grid->getNXC() - 2, _grid->getNYC() - 2, _grid->getNZC() - 2);
            const std::string rhoc_path = "/moments/rhoc_avg/cycle_" + cc.str();
            const std::string dive_path = "/moments/div_E_avg/cycle_" + cc.str();
            hid_t fid = this->output_adaptor.get_file_id();

            delete_if_exists(fid, rhoc_path);
            this->output_adaptor.write(rhoc_path, div_dims, _field->getRHOc_avg());

            delete_if_exists(fid, dive_path);
            this->output_adaptor.write(dive_path, div_dims, _field->getDivAverage());
        }

        if (contains_tag(tag, "B_ext"))
        {
            const PSK::Dimens bext_dims(_grid->getNXN() - 2, _grid->getNYN() - 2, _grid->getNZN() - 2);
            const std::string bxext_path = "/fields/Bx_ext/cycle_" + cc.str();
            const std::string byext_path = "/fields/By_ext/cycle_" + cc.str();
            const std::string bzext_path = "/fields/Bz_ext/cycle_" + cc.str();
            hid_t fid = this->output_adaptor.get_file_id();

            delete_if_exists(fid, bxext_path);
            this->output_adaptor.write(bxext_path, bext_dims, _field->getBx_ext());

            delete_if_exists(fid, byext_path);
            this->output_adaptor.write(byext_path, bext_dims, _field->getBy_ext());

            delete_if_exists(fid, bzext_path);
            this->output_adaptor.write(bzext_path, bext_dims, _field->getBz_ext());
        }

    	//* E field (defined at nodes) is written without ghost cells
        if (contains_tag(tag, "E"))
        {
            const PSK::Dimens e_dims(_grid->getNXN() - 2, _grid->getNYN() - 2, _grid->getNZN() - 2);
            const std::string ex_path = "/fields/Ex/cycle_" + cc.str();
            const std::string ey_path = "/fields/Ey/cycle_" + cc.str();
            const std::string ez_path = "/fields/Ez/cycle_" + cc.str();
            hid_t fid = this->output_adaptor.get_file_id();

            if (precision == "DOUBLE")
            {
                delete_if_exists(fid, ex_path);
                this->output_adaptor.write(ex_path, e_dims, _field->getEx());

                delete_if_exists(fid, ey_path);
                this->output_adaptor.write(ey_path, e_dims, _field->getEy());

                delete_if_exists(fid, ez_path);
                this->output_adaptor.write(ez_path, e_dims, _field->getEz());
            }
            else if (precision == "SINGLE")
            {
                for (int i = 1; i < _grid->getNXN() - 1; i++)
                    for (int j = 1; j < _grid->getNYN() - 1; j++)
                        for (int k = 1; k < _grid->getNZN() - 1; k++)
                            temp_double.set(i, j, k, _field->getEx(i, j, k));

                convert_to_single_precision(temp_double, field_single, _grid->getNXN(), _grid->getNYN(), _grid->getNZN());
                delete_if_exists(fid, ex_path);
                this->output_adaptor.write(ex_path, e_dims, field_single);

                for (int i = 1; i < _grid->getNXN() - 1; i++)
                    for (int j = 1; j < _grid->getNYN() - 1; j++)
                        for (int k = 1; k < _grid->getNZN() - 1; k++)
                            temp_double.set(i, j, k, _field->getEy(i, j, k));

                convert_to_single_precision(temp_double, field_single, _grid->getNXN(), _grid->getNYN(), _grid->getNZN());
                delete_if_exists(fid, ey_path);
                this->output_adaptor.write(ey_path, e_dims, field_single);

                for (int i = 1; i < _grid->getNXN() - 1; i++)
                    for (int j = 1; j < _grid->getNYN() - 1; j++)
                        for (int k = 1; k < _grid->getNZN() - 1; k++)
                            temp_double.set(i, j, k, _field->getEz(i, j, k));

                convert_to_single_precision(temp_double, field_single, _grid->getNXN(), _grid->getNYN(), _grid->getNZN());
                delete_if_exists(fid, ez_path);
                this->output_adaptor.write(ez_path, e_dims, field_single);
            }
        }

		//! ************************* Moments ************************* !//

		//* J (total current, defined at nodes) is written without ghost cells
        if (contains_tag(tag, "J"))
        {
            const PSK::Dimens j_dims(_grid->getNXN() - 2, _grid->getNYN() - 2, _grid->getNZN() - 2);
            const std::string jx_path = "/moments/Jx/cycle_" + cc.str();
            const std::string jy_path = "/moments/Jy/cycle_" + cc.str();
            const std::string jz_path = "/moments/Jz/cycle_" + cc.str();
            hid_t fid = this->output_adaptor.get_file_id();

            if (precision == "DOUBLE")
            {
                delete_if_exists(fid, jx_path);
                this->output_adaptor.write(jx_path, j_dims, _field->getJx());

                delete_if_exists(fid, jy_path);
                this->output_adaptor.write(jy_path, j_dims, _field->getJy());

                delete_if_exists(fid, jz_path);
                this->output_adaptor.write(jz_path, j_dims, _field->getJz());
            }
            else if (precision == "SINGLE")
            {
                for (int i = 1; i < _grid->getNXN() - 1; i++)
                    for (int j = 1; j < _grid->getNYN() - 1; j++)
                        for (int k = 1; k < _grid->getNZN() - 1; k++)
                            temp_double.set(i, j, k, _field->getJx(i, j, k));

                convert_to_single_precision(temp_double, field_single, _grid->getNXN(), _grid->getNYN(), _grid->getNZN());
                delete_if_exists(fid, jx_path);
                this->output_adaptor.write(jx_path, j_dims, field_single);

                for (int i = 1; i < _grid->getNXN() - 1; i++)
                    for (int j = 1; j < _grid->getNYN() - 1; j++)
                        for (int k = 1; k < _grid->getNZN() - 1; k++)
                            temp_double.set(i, j, k, _field->getJy(i, j, k));

                convert_to_single_precision(temp_double, field_single, _grid->getNXN(), _grid->getNYN(), _grid->getNZN());
                delete_if_exists(fid, jy_path);
                this->output_adaptor.write(jy_path, j_dims, field_single);

                for (int i = 1; i < _grid->getNXN() - 1; i++)
                    for (int j = 1; j < _grid->getNYN() - 1; j++)
                        for (int k = 1; k < _grid->getNZN() - 1; k++)
                            temp_double.set(i, j, k, _field->getJz(i, j, k));

                convert_to_single_precision(temp_double, field_single, _grid->getNXN(), _grid->getNYN(), _grid->getNZN());
                delete_if_exists(fid, jz_path);
                this->output_adaptor.write(jz_path, j_dims, field_single);
            }
        }

		//* Js (current for each species, defined at nodes) is written without ghost cells
        if (contains_tag(tag, "J_s"))
        {
            //* rhos (charge density for each species, defined at nodes) is written without ghost cells
            const PSK::Dimens js_dims(_grid->getNXN() - 2, _grid->getNYN() - 2, _grid->getNZN() - 2);
            hid_t fid = this->output_adaptor.get_file_id();

            for (int is = 0; is < ns; ++is)
            {
                stringstream ii;
                ii << is;

                const std::string jxs_path = "/moments/species_" + ii.str() + "/Jx/cycle_" + cc.str();
                const std::string jys_path = "/moments/species_" + ii.str() + "/Jy/cycle_" + cc.str();
                const std::string jzs_path = "/moments/species_" + ii.str() + "/Jz/cycle_" + cc.str();

                if (precision == "DOUBLE")
                {
                    delete_if_exists(fid, jxs_path);
                    this->output_adaptor.write(jxs_path, js_dims, is, _field->getJxs());

                    delete_if_exists(fid, jys_path);
                    this->output_adaptor.write(jys_path, js_dims, is, _field->getJys());

                    delete_if_exists(fid, jzs_path);
                    this->output_adaptor.write(jzs_path, js_dims, is, _field->getJzs());
                }
                else if (precision == "SINGLE")
                {
                    for (int i = 1; i < _grid->getNXN() - 1; i++)
                        for (int j = 1; j < _grid->getNYN() - 1; j++)
                            for (int k = 1; k < _grid->getNZN() - 1; k++)
                                temp_double.set(i, j, k, _field->getJxs(i, j, k, is));

                    convert_to_single_precision(temp_double, field_single, _grid->getNXN(), _grid->getNYN(), _grid->getNZN());
                    delete_if_exists(fid, jxs_path);
                    this->output_adaptor.write(jxs_path, js_dims, field_single);

                    for (int i = 1; i < _grid->getNXN() - 1; i++)
                        for (int j = 1; j < _grid->getNYN() - 1; j++)
                            for (int k = 1; k < _grid->getNZN() - 1; k++)
                                temp_double.set(i, j, k, _field->getJys(i, j, k, is));
                            
                    convert_to_single_precision(temp_double, field_single, _grid->getNXN(), _grid->getNYN(), _grid->getNZN());
                    delete_if_exists(fid, jys_path);
                    this->output_adaptor.write(jys_path, js_dims, field_single);

                    for (int i = 1; i < _grid->getNXN() - 1; i++)
                        for (int j = 1; j < _grid->getNYN() - 1; j++)
                            for (int k = 1; k < _grid->getNZN() - 1; k++)
                                temp_double.set(i, j, k, _field->getJzs(i, j, k, is));
                            
                    convert_to_single_precision(temp_double, field_single, _grid->getNXN(), _grid->getNYN(), _grid->getNZN());
                    delete_if_exists(fid, jzs_path);
                    this->output_adaptor.write(jzs_path, js_dims, field_single);
                }
            }
        }

        //* rhos (charge density for each species, defined at nodes) is written without ghost cells
        if (contains_tag(tag, "rho_s"))
        {
            const PSK::Dimens rhos_dims(_grid->getNXN() - 2, _grid->getNYN() - 2, _grid->getNZN() - 2);
            hid_t fid = this->output_adaptor.get_file_id();

            for (int is = 0; is < ns; ++is)
            {
                stringstream ii;
                ii << is;

                const std::string rho_path = "/moments/species_" + ii.str() + "/rho/cycle_" + cc.str();

                if (precision == "DOUBLE")
                {
                    delete_if_exists(fid, rho_path);
                    this->output_adaptor.write(rho_path, rhos_dims, is, _field->getRHOns());
                }
                else if (precision == "SINGLE")
                {
                    for (int i = 1; i < _grid->getNXN() - 1; i++)
                        for (int j = 1; j < _grid->getNYN() - 1; j++)
                            for (int k = 1; k < _grid->getNZN() - 1; k++)
                                temp_double.set(i, j, k, _field->getRHOns(i, j, k, is));
                            
                    convert_to_single_precision(temp_double, field_single, _grid->getNXN(), _grid->getNYN(), _grid->getNZN());
                    delete_if_exists(fid, rho_path);
                    this->output_adaptor.write(rho_path, rhos_dims, field_single);
                }
            }
        }

        //* rhos (overall charge density) is written without ghost cells
        if (contains_tag(tag, "rho"))
        {
            const PSK::Dimens rho_dims(_grid->getNXN() - 2, _grid->getNYN() - 2, _grid->getNZN() - 2);
            const std::string rho_path = "/moments/rho/cycle_" + cc.str();
            hid_t fid = this->output_adaptor.get_file_id();

            if (precision == "DOUBLE")
            {
                delete_if_exists(fid, rho_path);
                this->output_adaptor.write(rho_path, rho_dims, _field->getRHOn());
            }
            else if (precision == "SINGLE")
            {
                for (int i = 1; i < _grid->getNXN() - 1; i++)
                    for (int j = 1; j < _grid->getNYN() - 1; j++)
                        for (int k = 1; k < _grid->getNZN() - 1; k++)
                            temp_double.set(i, j, k, _field->getRHOn(i, j, k));
                        
                convert_to_single_precision(temp_double, field_single, _grid->getNXN(), _grid->getNYN(), _grid->getNZN());
                delete_if_exists(fid, rho_path);
                this->output_adaptor.write(rho_path, rho_dims, field_single);
            }
        }

		//* Pressure tensor (for each species, defined at nodes) is written without ghost cells
		if (contains_tag(tag, "pressure"))
        {
            const PSK::Dimens p_dims(_grid->getNXN() - 2, _grid->getNYN() - 2, _grid->getNZN() - 2);
            hid_t fid = this->output_adaptor.get_file_id();

            for (int is = 0; is < ns; ++is) 
            {
                stringstream ii;
                ii << is;

                const std::string pxx_path = "/moments/species_" + ii.str() + "/pXX/cycle_" + cc.str();
                const std::string pxy_path = "/moments/species_" + ii.str() + "/pXY/cycle_" + cc.str();
                const std::string pxz_path = "/moments/species_" + ii.str() + "/pXZ/cycle_" + cc.str();
                const std::string pyy_path = "/moments/species_" + ii.str() + "/pYY/cycle_" + cc.str();
                const std::string pyz_path = "/moments/species_" + ii.str() + "/pYZ/cycle_" + cc.str();
                const std::string pzz_path = "/moments/species_" + ii.str() + "/pZZ/cycle_" + cc.str();

                if (precision == "DOUBLE")
                {
                    delete_if_exists(fid, pxx_path);
                    this->output_adaptor.write(pxx_path, p_dims, is, _field->getpXXsn());

                    delete_if_exists(fid, pxy_path);
                    this->output_adaptor.write(pxy_path, p_dims, is, _field->getpXYsn());

                    delete_if_exists(fid, pxz_path);
                    this->output_adaptor.write(pxz_path, p_dims, is, _field->getpXZsn());

                    delete_if_exists(fid, pyy_path);
                    this->output_adaptor.write(pyy_path, p_dims, is, _field->getpYYsn());

                    delete_if_exists(fid, pyz_path);
                    this->output_adaptor.write(pyz_path, p_dims, is, _field->getpYZsn());

                    delete_if_exists(fid, pzz_path);
                    this->output_adaptor.write(pzz_path, p_dims, is, _field->getpZZsn());
                }
                else if (precision == "SINGLE")
                {
                    for (int i = 1; i < _grid->getNXN() - 1; i++)
                        for (int j = 1; j < _grid->getNYN() - 1; j++)
                            for (int k = 1; k < _grid->getNZN() - 1; k++)
                                temp_double.set(i, j, k, _field->getpXXsn(i, j, k, is));

                    convert_to_single_precision(temp_double, field_single, _grid->getNXN(), _grid->getNYN(), _grid->getNZN());
                    delete_if_exists(fid, pxx_path);
                    this->output_adaptor.write(pxx_path, p_dims, field_single);

                    for (int i = 1; i < _grid->getNXN() - 1; i++)
                        for (int j = 1; j < _grid->getNYN() - 1; j++)
                            for (int k = 1; k < _grid->getNZN() - 1; k++)
                                temp_double.set(i, j, k, _field->getpXYsn(i, j, k, is));
                            
                    convert_to_single_precision(temp_double, field_single, _grid->getNXN(), _grid->getNYN(), _grid->getNZN());
                    delete_if_exists(fid, pxy_path);
                    this->output_adaptor.write(pxy_path, p_dims, field_single);

                    for (int i = 1; i < _grid->getNXN() - 1; i++)
                        for (int j = 1; j < _grid->getNYN() - 1; j++)
                            for (int k = 1; k < _grid->getNZN() - 1; k++)
                                temp_double.set(i, j, k, _field->getpXZsn(i, j, k, is));
                            
                    convert_to_single_precision(temp_double, field_single, _grid->getNXN(), _grid->getNYN(), _grid->getNZN());
                    delete_if_exists(fid, pxz_path);
                    this->output_adaptor.write(pxz_path, p_dims, field_single);

                    for (int i = 1; i < _grid->getNXN() - 1; i++)
                        for (int j = 1; j < _grid->getNYN() - 1; j++)
                            for (int k = 1; k < _grid->getNZN() - 1; k++)
                                temp_double.set(i, j, k, _field->getpYYsn(i, j, k, is));

                    convert_to_single_precision(temp_double, field_single, _grid->getNXN(), _grid->getNYN(), _grid->getNZN());
                    delete_if_exists(fid, pyy_path);
                    this->output_adaptor.write(pyy_path, p_dims, field_single);

                    for (int i = 1; i < _grid->getNXN() - 1; i++)
                        for (int j = 1; j < _grid->getNYN() - 1; j++)
                            for (int k = 1; k < _grid->getNZN() - 1; k++)
                                temp_double.set(i, j, k, _field->getpYZsn(i, j, k, is));
                            
                    convert_to_single_precision(temp_double, field_single, _grid->getNXN(), _grid->getNYN(), _grid->getNZN());
                    delete_if_exists(fid, pyz_path);
                    this->output_adaptor.write(pyz_path, p_dims, field_single);

                    for (int i = 1; i < _grid->getNXN() - 1; i++)
                        for (int j = 1; j < _grid->getNYN() - 1; j++)
                            for (int k = 1; k < _grid->getNZN() - 1; k++)
                                temp_double.set(i, j, k, _field->getpZZsn(i, j, k, is));
                            
                    convert_to_single_precision(temp_double, field_single, _grid->getNXN(), _grid->getNYN(), _grid->getNZN());
                    delete_if_exists(fid, pzz_path);
                    this->output_adaptor.write(pzz_path, p_dims, field_single);
                }
            }
        }

        //* Energy flux density (for each species, defined at nodes) is written without ghost cells
        if (contains_tag(tag, "E_flux"))
        {
            const PSK::Dimens ef_dims(_grid->getNXN() - 2, _grid->getNYN() - 2, _grid->getNZN() - 2);
            hid_t fid = this->output_adaptor.get_file_id();

            for (int is = 0; is < ns; ++is) 
            {
                stringstream ii;
                ii << is;

                const std::string efx_path = "/moments/species_" + ii.str() + "/EFx/cycle_" + cc.str();
                const std::string efy_path = "/moments/species_" + ii.str() + "/EFy/cycle_" + cc.str();
                const std::string efz_path = "/moments/species_" + ii.str() + "/EFz/cycle_" + cc.str();

                if (precision == "DOUBLE")
                {
                    delete_if_exists(fid, efx_path);
                    this->output_adaptor.write(efx_path, ef_dims, is, _field->getEFxs());

                    delete_if_exists(fid, efy_path);
                    this->output_adaptor.write(efy_path, ef_dims, is, _field->getEFys());

                    delete_if_exists(fid, efz_path);
                    this->output_adaptor.write(efz_path, ef_dims, is, _field->getEFzs());
                }
                else if (precision == "SINGLE")
                {
                    for (int i = 1; i < _grid->getNXN() - 1; i++)
                        for (int j = 1; j < _grid->getNYN() - 1; j++)
                            for (int k = 1; k < _grid->getNZN() - 1; k++)
                                temp_double.set(i, j, k, _field->getEFxs(i, j, k, is));

                    convert_to_single_precision(temp_double, field_single, _grid->getNXN(), _grid->getNYN(), _grid->getNZN());
                    delete_if_exists(fid, efx_path);
                    this->output_adaptor.write(efx_path, ef_dims, field_single);

                    for (int i = 1; i < _grid->getNXN() - 1; i++)
                        for (int j = 1; j < _grid->getNYN() - 1; j++)
                            for (int k = 1; k < _grid->getNZN() - 1; k++)
                                temp_double.set(i, j, k, _field->getEFys(i, j, k, is));
                            
                    convert_to_single_precision(temp_double, field_single, _grid->getNXN(), _grid->getNYN(), _grid->getNZN());
                    delete_if_exists(fid, efy_path);
                    this->output_adaptor.write(efy_path, ef_dims, field_single);

                    for (int i = 1; i < _grid->getNXN() - 1; i++)
                        for (int j = 1; j < _grid->getNYN() - 1; j++)
                            for (int k = 1; k < _grid->getNZN() - 1; k++)
                                temp_double.set(i, j, k, _field->getEFzs(i, j, k, is));
                            
                    convert_to_single_precision(temp_double, field_single, _grid->getNXN(), _grid->getNYN(), _grid->getNZN());
                    delete_if_exists(fid, efz_path);
                    this->output_adaptor.write(efz_path, ef_dims, field_single);
                }
            }
        }

        //* Heat flux tensor (for each species, defined at nodes) is written without ghost cells
        if (contains_tag(tag, "H_flux"))
        {
            const PSK::Dimens q_dims(_grid->getNXN() - 2, _grid->getNYN() - 2, _grid->getNZN() - 2);
            hid_t fid = this->output_adaptor.get_file_id();

            for (int is = 0; is < ns; ++is) 
            {
                stringstream ii;
                ii << is;

                const std::string qxxx_path = "/moments/species_" + ii.str() + "/Qxxx/cycle_" + cc.str();
                const std::string qxxy_path = "/moments/species_" + ii.str() + "/Qxxy/cycle_" + cc.str();
                const std::string qxyy_path = "/moments/species_" + ii.str() + "/Qxyy/cycle_" + cc.str();
                const std::string qxzz_path = "/moments/species_" + ii.str() + "/Qxzz/cycle_" + cc.str();
                const std::string qyyy_path = "/moments/species_" + ii.str() + "/Qyyy/cycle_" + cc.str();
                const std::string qyzz_path = "/moments/species_" + ii.str() + "/Qyzz/cycle_" + cc.str();
                const std::string qzzz_path = "/moments/species_" + ii.str() + "/Qzzz/cycle_" + cc.str();
                const std::string qxyz_path = "/moments/species_" + ii.str() + "/Qxyz/cycle_" + cc.str();
                const std::string qxxz_path = "/moments/species_" + ii.str() + "/Qxxz/cycle_" + cc.str();
                const std::string qyyz_path = "/moments/species_" + ii.str() + "/Qyyz/cycle_" + cc.str();

                if (precision == "DOUBLE")
                {
                    delete_if_exists(fid, qxxx_path);
                    this->output_adaptor.write(qxxx_path, q_dims, is, _field->getQxxxs());

                    delete_if_exists(fid, qxxy_path);
                    this->output_adaptor.write(qxxy_path, q_dims, is, _field->getQxxys());

                    delete_if_exists(fid, qxyy_path);
                    this->output_adaptor.write(qxyy_path, q_dims, is, _field->getQxyys());

                    delete_if_exists(fid, qxzz_path);
                    this->output_adaptor.write(qxzz_path, q_dims, is, _field->getQxzzs());

                    delete_if_exists(fid, qyyy_path);
                    this->output_adaptor.write(qyyy_path, q_dims, is, _field->getQyyys());

                    delete_if_exists(fid, qyzz_path);
                    this->output_adaptor.write(qyzz_path, q_dims, is, _field->getQyzzs());

                    delete_if_exists(fid, qzzz_path);
                    this->output_adaptor.write(qzzz_path, q_dims, is, _field->getQzzzs());

                    delete_if_exists(fid, qxyz_path);
                    this->output_adaptor.write(qxyz_path, q_dims, is, _field->getQxyzs());

                    delete_if_exists(fid, qxxz_path);
                    this->output_adaptor.write(qxxz_path, q_dims, is, _field->getQxxzs());

                    delete_if_exists(fid, qyyz_path);
                    this->output_adaptor.write(qyyz_path, q_dims, is, _field->getQyyzs());
                }
                else if (precision == "SINGLE")
                {
                    for (int i = 1; i < _grid->getNXN() - 1; i++)
                        for (int j = 1; j < _grid->getNYN() - 1; j++)
                            for (int k = 1; k < _grid->getNZN() - 1; k++)
                                temp_double.set(i, j, k, _field->getQxxxs(i, j, k, is));

                    convert_to_single_precision(temp_double, field_single, _grid->getNXN(), _grid->getNYN(), _grid->getNZN());
                    delete_if_exists(fid, qxxx_path);
                    this->output_adaptor.write(qxxx_path, q_dims, field_single);

                    for (int i = 1; i < _grid->getNXN() - 1; i++)
                        for (int j = 1; j < _grid->getNYN() - 1; j++)
                            for (int k = 1; k < _grid->getNZN() - 1; k++)
                                temp_double.set(i, j, k, _field->getQxxys(i, j, k, is));
                            
                    convert_to_single_precision(temp_double, field_single, _grid->getNXN(), _grid->getNYN(), _grid->getNZN());
                    delete_if_exists(fid, qxxy_path);
                    this->output_adaptor.write(qxxy_path, q_dims, field_single);

                    for (int i = 1; i < _grid->getNXN() - 1; i++)
                        for (int j = 1; j < _grid->getNYN() - 1; j++)
                            for (int k = 1; k < _grid->getNZN() - 1; k++)
                                temp_double.set(i, j, k, _field->getQxyys(i, j, k, is));
                            
                    convert_to_single_precision(temp_double, field_single, _grid->getNXN(), _grid->getNYN(), _grid->getNZN());
                    delete_if_exists(fid, qxyy_path);
                    this->output_adaptor.write(qxyy_path, q_dims, field_single);

                    for (int i = 1; i < _grid->getNXN() - 1; i++)
                        for (int j = 1; j < _grid->getNYN() - 1; j++)
                            for (int k = 1; k < _grid->getNZN() - 1; k++)
                                temp_double.set(i, j, k, _field->getQxzzs(i, j, k, is));
                            
                    convert_to_single_precision(temp_double, field_single, _grid->getNXN(), _grid->getNYN(), _grid->getNZN());
                    delete_if_exists(fid, qxzz_path);
                    this->output_adaptor.write(qxzz_path, q_dims, field_single);
                                    
                    for (int i = 1; i < _grid->getNXN() - 1; i++)
                        for (int j = 1; j < _grid->getNYN() - 1; j++)
                            for (int k = 1; k < _grid->getNZN() - 1; k++)
                                temp_double.set(i, j, k, _field->getQyyys(i, j, k, is));

                    convert_to_single_precision(temp_double, field_single, _grid->getNXN(), _grid->getNYN(), _grid->getNZN());
                    delete_if_exists(fid, qyyy_path);
                    this->output_adaptor.write(qyyy_path, q_dims, field_single);

                    for (int i = 1; i < _grid->getNXN() - 1; i++)
                        for (int j = 1; j < _grid->getNYN() - 1; j++)
                            for (int k = 1; k < _grid->getNZN() - 1; k++)
                                temp_double.set(i, j, k, _field->getQyzzs(i, j, k, is));
                            
                    convert_to_single_precision(temp_double, field_single, _grid->getNXN(), _grid->getNYN(), _grid->getNZN());
                    delete_if_exists(fid, qyzz_path);
                    this->output_adaptor.write(qyzz_path, q_dims, field_single);

                    for (int i = 1; i < _grid->getNXN() - 1; i++)
                        for (int j = 1; j < _grid->getNYN() - 1; j++)
                            for (int k = 1; k < _grid->getNZN() - 1; k++)
                                temp_double.set(i, j, k, _field->getQzzzs(i, j, k, is));
                            
                    convert_to_single_precision(temp_double, field_single, _grid->getNXN(), _grid->getNYN(), _grid->getNZN());
                    delete_if_exists(fid, qzzz_path);
                    this->output_adaptor.write(qzzz_path, q_dims, field_single);

                    for (int i = 1; i < _grid->getNXN() - 1; i++)
                        for (int j = 1; j < _grid->getNYN() - 1; j++)
                            for (int k = 1; k < _grid->getNZN() - 1; k++)
                                temp_double.set(i, j, k, _field->getQxyzs(i, j, k, is));
                            
                    convert_to_single_precision(temp_double, field_single, _grid->getNXN(), _grid->getNYN(), _grid->getNZN());
                    delete_if_exists(fid, qxyz_path);
                    this->output_adaptor.write(qxyz_path, q_dims, field_single);

                    for (int i = 1; i < _grid->getNXN() - 1; i++)
                        for (int j = 1; j < _grid->getNYN() - 1; j++)
                            for (int k = 1; k < _grid->getNZN() - 1; k++)
                                temp_double.set(i, j, k, _field->getQxxzs(i, j, k, is));
                            
                    convert_to_single_precision(temp_double, field_single, _grid->getNXN(), _grid->getNYN(), _grid->getNZN());
                    delete_if_exists(fid, qxxz_path);
                    this->output_adaptor.write(qxxz_path, q_dims, field_single);

                    for (int i = 1; i < _grid->getNXN() - 1; i++)
                        for (int j = 1; j < _grid->getNYN() - 1; j++)
                            for (int k = 1; k < _grid->getNZN() - 1; k++)
                                temp_double.set(i, j, k, _field->getQyyzs(i, j, k, is));
                            
                    convert_to_single_precision(temp_double, field_single, _grid->getNXN(), _grid->getNYN(), _grid->getNZN());
                    delete_if_exists(fid, qyyz_path);
                    this->output_adaptor.write(qyyz_path, q_dims, field_single);
                }
            }
        }
    }


    //! ************************* Particles ************************* !//

    void output_particles(const string & tag, int cycle, string precision) 
    {
        stringstream cc;
        cc << cycle;
        const int ns = _col->getNs();
        const int nstestpart = _col->getNsTestPart();

        //* Temporary arrays to store particle data in single & double precision, respectively
        std::vector<float> particle_single; const double* particle_double = nullptr;      
        hid_t fid = this->output_adaptor.get_file_id();

        if (contains_tag(tag, "position"))
        {
            for (int is = 0; is < ns; ++is) 
            {
                long long nop = _part[is]->getNOP();
                
                stringstream ii;
                ii << is;

                const std::string x_path = "/particles/species_" + ii.str() + "/x/cycle_" + cc.str();
                const std::string y_path = "/particles/species_" + ii.str() + "/y/cycle_" + cc.str();
                const std::string z_path = "/particles/species_" + ii.str() + "/z/cycle_" + cc.str();

                if (precision == "DOUBLE")
                {
                    delete_if_exists(fid, x_path);
                    this->output_adaptor.write(x_path, PSK::Dimens(nop), _part[is]->getXall());

                    delete_if_exists(fid, y_path);
                    this->output_adaptor.write(y_path, PSK::Dimens(nop), _part[is]->getYall());

                    delete_if_exists(fid, z_path);
                    this->output_adaptor.write(z_path, PSK::Dimens(nop), _part[is]->getZall());
                }
                else if (precision == "SINGLE")
                {
                    particle_single.resize(nop);
                    
                    particle_double = _part[is]->getXall();
                    for (int i = 0; i < nop; i++)
                        particle_single[i] = static_cast<float>(particle_double[i]);

                    delete_if_exists(fid, x_path);
                    this->output_adaptor.write(x_path, PSK::Dimens(nop), particle_single);

                    particle_double = _part[is]->getYall();
                    for (int i = 0; i < nop; i++)
                        particle_single[i] = static_cast<float>(particle_double[i]);

                    delete_if_exists(fid, y_path);
                    this->output_adaptor.write(y_path, PSK::Dimens(nop), particle_single);

                    particle_double = _part[is]->getZall();
                    for (int i = 0; i < nop; i++)
                        particle_single[i] = static_cast<float>(particle_double[i]);

                    delete_if_exists(fid, z_path);
                    this->output_adaptor.write(z_path, PSK::Dimens(nop), particle_single);
                }
            }
        }

        if (contains_tag(tag, "velocity"))
        {
            for (int is = 0; is < ns; ++is) 
            {
                long long nop = _part[is]->getNOP();
                stringstream ii;
                ii << is;

                const std::string u_path = "/particles/species_" + ii.str() + "/u/cycle_" + cc.str();
                const std::string v_path = "/particles/species_" + ii.str() + "/v/cycle_" + cc.str();
                const std::string w_path = "/particles/species_" + ii.str() + "/w/cycle_" + cc.str();

                if (precision == "DOUBLE")
                {
                    delete_if_exists(fid, u_path);
                    this->output_adaptor.write(u_path, PSK::Dimens(nop), _part[is]->getUall());

                    delete_if_exists(fid, v_path);
                    this->output_adaptor.write(v_path, PSK::Dimens(nop), _part[is]->getVall());

                    delete_if_exists(fid, w_path);
                    this->output_adaptor.write(w_path, PSK::Dimens(nop), _part[is]->getWall());
                }
                else if (precision == "SINGLE")
                {
                    particle_single.resize(nop);
                    
                    particle_double = _part[is]->getUall();
                    for (int i = 0; i < nop; i++)
                        particle_single[i] = static_cast<float>(particle_double[i]);

                    delete_if_exists(fid, u_path);
                    this->output_adaptor.write(u_path, PSK::Dimens(nop), particle_single);

                    particle_double = _part[is]->getVall();
                    for (int i = 0; i < nop; i++)
                        particle_single[i] = static_cast<float>(particle_double[i]);

                    delete_if_exists(fid, v_path);
                    this->output_adaptor.write(v_path, PSK::Dimens(nop), particle_single);

                    particle_double = _part[is]->getWall();
                    for (int i = 0; i < nop; i++)
                        particle_single[i] = static_cast<float>(particle_double[i]);

                    delete_if_exists(fid, w_path);
                    this->output_adaptor.write(w_path, PSK::Dimens(nop), particle_single);
                }
            }
        }
        
        if (contains_tag(tag, "q"))
        {
            for (int is = 0; is < ns; ++is) 
            {
                long long nop = _part[is]->getNOP();
                stringstream ii;
                ii << is;

                const std::string q_path = "/particles/species_" + ii.str() + "/q/cycle_" + cc.str();

                if (precision == "DOUBLE")
                {
                    delete_if_exists(fid, q_path);
                    this->output_adaptor.write(q_path, PSK::Dimens(nop), _part[is]->getQall());
                }
                else if (precision == "SINGLE")
                {
                    particle_single.resize(nop);
                    particle_double = _part[is]->getQall();

                    for (int i = 0; i < nop; i++)
                        particle_single[i] = static_cast<float>(particle_double[i]);

                    delete_if_exists(fid, q_path);
                    this->output_adaptor.write(q_path, PSK::Dimens(nop), particle_single);
                }
            }
        }

        if (contains_tag(tag, "ID"))
        {
            for (int is = 0; is < ns; ++is) 
            {
                long long nop = _part[is]->getNOP();
                stringstream ii;
                ii << is;
                const std::string id_path = "/particles/species_" + ii.str() + "/ID/cycle_" + cc.str();

                delete_if_exists(fid, id_path);
                this->output_adaptor.write(id_path, PSK::Dimens(nop), _part[is]->getParticleIDall());
            }
        }
    }

    
    //! ************************* Downsampled Particles ************************* !//

    void output_particles_DS(const string & tag, int cycle, int sample, string precision) 
    {
        //* sample --> Particle downsampling factor

        std::vector <double> X, Y, Z, U, V, W, Q;
        std::vector <float> X_single, Y_single, Z_single, U_single, V_single, W_single, Q_single;

        stringstream cc;
        cc << cycle;
        const int ns = _col->getNs();
        hid_t fid = this->output_adaptor.get_file_id();

        if (contains_tag(tag, "position_DS"))
        {
            for (int is = 0; is < ns; ++is) 
            {
                long long nop = _part[is]->getNOP();
                stringstream ii;
                ii << is;
                const int num_samples = nop/sample;
                const std::string x_path = "/particles_DS/species_" + ii.str() + "/x/cycle_" + cc.str();
                const std::string y_path = "/particles_DS/species_" + ii.str() + "/y/cycle_" + cc.str();
                const std::string z_path = "/particles_DS/species_" + ii.str() + "/z/cycle_" + cc.str();

                if (precision == "DOUBLE")
                {
                    X.clear(); X.reserve(num_samples);
                    Y.clear(); Y.reserve(num_samples);
                    Z.clear(); Z.reserve(num_samples);
                    
                    for (int n = 0; n < nop; n += sample) 
                    {
                        X.push_back(_part[is]->getX(n));
                        Y.push_back(_part[is]->getY(n));
                        Z.push_back(_part[is]->getZ(n));
                    }

                    delete_if_exists(fid, x_path);
                    this->output_adaptor.write(x_path, PSK::Dimens(X.size()), X);

                    delete_if_exists(fid, y_path);
                    this->output_adaptor.write(y_path, PSK::Dimens(Y.size()), Y);

                    delete_if_exists(fid, z_path);
                    this->output_adaptor.write(z_path, PSK::Dimens(Z.size()), Z);
                }
                else if (precision == "SINGLE")
                {
                    X_single.clear(); X_single.reserve(num_samples);
                    Y_single.clear(); Y_single.reserve(num_samples);
                    Z_single.clear(); Z_single.reserve(num_samples);
                    
                    for (int n = 0; n < nop; n += sample) 
                    {
                        X_single.push_back(_part[is]->getX(n));
                        Y_single.push_back(_part[is]->getY(n));
                        Z_single.push_back(_part[is]->getZ(n));
                    }

                    delete_if_exists(fid, x_path);
                    this->output_adaptor.write(x_path, PSK::Dimens(X_single.size()), X_single);

                    delete_if_exists(fid, y_path);
                    this->output_adaptor.write(y_path, PSK::Dimens(Y_single.size()), Y_single);

                    delete_if_exists(fid, z_path);
                    this->output_adaptor.write(z_path, PSK::Dimens(Z_single.size()), Z_single);
                }
            }
        }
                
        if (contains_tag(tag, "velocity_DS"))
        {
            for (int is = 0; is < ns; ++is) 
            {
                long long nop = _part[is]->getNOP();
                stringstream ii;
                ii << is;
                const int num_samples = nop/sample;
                const std::string u_path = "/particles_DS/species_" + ii.str() + "/u/cycle_" + cc.str();
                const std::string v_path = "/particles_DS/species_" + ii.str() + "/v/cycle_" + cc.str();
                const std::string w_path = "/particles_DS/species_" + ii.str() + "/w/cycle_" + cc.str();

                if (precision == "DOUBLE")
                {
                    U.clear(); U.reserve(num_samples);
                    V.clear(); V.reserve(num_samples);
                    W.clear(); W.reserve(num_samples);
                    
                    for (int n = 0; n < nop; n += sample) 
                    {
                        U.push_back(_part[is]->getU(n));
                        V.push_back(_part[is]->getV(n));
                        W.push_back(_part[is]->getW(n));
                    }
                    
                    delete_if_exists(fid, u_path);
                    this->output_adaptor.write(u_path, PSK::Dimens(U.size()), U);

                    delete_if_exists(fid, v_path);
                    this->output_adaptor.write(v_path, PSK::Dimens(V.size()), V);

                    delete_if_exists(fid, w_path);
                    this->output_adaptor.write(w_path, PSK::Dimens(W.size()), W);
                }
                else if (precision == "SINGLE")
                {
                    U_single.clear(); U_single.reserve(num_samples);
                    V_single.clear(); V_single.reserve(num_samples);
                    W_single.clear(); W_single.reserve(num_samples);
                    
                    for (int n = 0; n < nop; n += sample) 
                    {
                        U_single.push_back(_part[is]->getU(n));
                        V_single.push_back(_part[is]->getV(n));
                        W_single.push_back(_part[is]->getW(n));
                    }
                    
                    delete_if_exists(fid, u_path);
                    this->output_adaptor.write(u_path, PSK::Dimens(U_single.size()), U_single);

                    delete_if_exists(fid, v_path);
                    this->output_adaptor.write(v_path, PSK::Dimens(V_single.size()), V_single);

                    delete_if_exists(fid, w_path);
                    this->output_adaptor.write(w_path, PSK::Dimens(W_single.size()), W_single);
                }
            }
        }

        if (contains_tag(tag, "q_DS"))
        {
            for (int is = 0; is < ns; ++is) 
            {
                long long nop = _part[is]->getNOP();
                stringstream ii;
                ii << is;
                const int num_samples = nop/sample;
                const std::string q_path = "/particles_DS/species_" + ii.str() + "/q/cycle_" + cc.str();

                if (precision == "DOUBLE")
                {
                    Q.clear(); Q.reserve(num_samples);
                    
                    for (int n = 0; n < nop; n += sample)
                        Q.push_back(_part[is]->getQ(n));
                    
                    delete_if_exists(fid, q_path);
                    this->output_adaptor.write(q_path, PSK::Dimens(Q.size()), Q);
                }
                else if (precision == "SINGLE")
                {
                    Q_single.clear(); Q_single.reserve(num_samples);
                    
                    for (int n = 0; n < nop; n += sample)
                        Q_single.push_back(static_cast<float>(_part[is]->getQ(n)));
                    
                    delete_if_exists(fid, q_path);
                    this->output_adaptor.write(q_path, PSK::Dimens(Q_single.size()), Q_single);
                }
            }
        }
    }
};

#endif // _PSK_OUTPUT_H_
