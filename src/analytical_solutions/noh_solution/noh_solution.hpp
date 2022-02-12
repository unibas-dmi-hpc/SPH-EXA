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
 * @brief This routine produces 1d solutions for the "The wall heating shock (Noh shock)"
 *        , in planar(1D), cylindrical(2D) or spherical geometry(3D)
 *
 *        This routine is a C++ conversion of one Fortran code based in the paper:
 *        - "Errors for Calculations of Strong Shocks Using an Artificial Viscosity and an Artificial Heat Flux",
 *          W.F. Noh. JCP 72 (1987), 78-120
 *
 * @author Jose A. Escartin <ja.escartin@gmail.com>
 *
 */

#pragma once

#include <string>
#include <vector>

using namespace std;

class NohSolution
{
public:
    static void create(vector<double>& r,       // Radius position
                       const size_t    dim,     // Dimensions
                       const size_t    rPoints, // Number of points between r0-r1
                       const double    time,    // Time at solution
                       const double    gamma_i, // Adiabatic coeficient
                       const double    rho0,    // Initial density
                       const double    u0,      // Initial internal energy
                       const double    p0,      // Initial pressure
                       const double    vel0,    // Initial velocity
                       const double    cs0,     // Initial sound speed
                       const string    outfile);   // Output solution filename

private:
    static void NohSol(const size_t          xgeom,   // geometry factor: 1=planar, 2=cylindircal, 3=spherical
                       const size_t          rPoints, // Number of points between r0-r1
                       const double          time,    // temporal point where solution is desired [seconds]
                       const double          gamma,   // gamma law equation of state
                       const double          rho0,    // ambient density g/cm**3 in 'rho = rho0 * r**(-omega)'
                       const double          u0,      // ambient internal energy [erg/g]
                       const double          p0,      // ambient pressure [erg/cm**3]
                       const double          vel0,    // ambient material speed [cm/s]
                       const double          cs0,     // ambient sound speed [cm/s]
                       const vector<double>& r,       // spatial points where solution is desired [cm]
                       vector<double>&       rho,     // out: density  [g/cm**3]
                       vector<double>&       u,       // out: specific internal energy [erg/g]
                       vector<double>&       p,       // out: presssure [erg/cm**3]
                       vector<double>&       vel,     // out: velocity [cm/s]
                       vector<double>&       cs);           // out: sound speed [cm/s]
};
