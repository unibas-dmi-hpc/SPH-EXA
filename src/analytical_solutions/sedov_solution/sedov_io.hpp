/*
 * MIT License
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

/*! @file
 *
 * @brief This class manage the data layer in the application.
 *        It provide a variety of read/write functions based in the number of dimensions.
 *
 * @author Jose A. Escartin <ja.escartin@gmail.com>
 *
 */

#pragma once

#include <string>
#include <vector>
#include <fstream>

#include "tests/io.hpp"
#include "tests/particle_io.hpp"

using namespace std;

class SedovFileData : public FileData
{

private:

    SedovFileData();                             // Singleton

public:

    static void writeColumns1D(
        ostream& out);                           //

    static void writeData1D(
        const size_t          n,                 //
        const vector<double>& r,                 //
        const vector<double>& rho,               //
        const vector<double>& u,                 //
        const vector<double>& p,                 //
        const vector<double>& vel,               //
        const vector<double>& cs,                //
        const double          rho_shock,         //
        const double          u_shock,           //
        const double          p_shock,           //
        const double          vel_shock,         //
        const double          cs_shock,          //
        const double          rho0,              //
        const string&         outfile);          //

    static void writeParticle1D(
        const size_t              n,             //
        const vector<ParticleIO>& vParticle,     //
        const double              rho_shock,     //
        const double              u_shock,       //
        const double              p_shock,       //
        const double              vel_shock,     //
        const double              cs_shock,      //
        const double              rho0,          //
        const string&             outfile);      //
};
