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
 *        - "Errors for Calculations of Strong Shocks Using an Artificial Viscosity and an Artificial Heat Flux", W.F. Noh. JCP 72 (1987), 78-120
 *
 * @author Jose A. Escartin <ja.escartin@gmail.com>
 *
 */

#pragma once

#include <cmath>

#include "NohDataFileWriter.hpp"

using namespace std;


class NohAnalyticalSolution
{
public:

    static void create(const size_t dim,              // Dimensions
                       const double r0,               // Initial radio
                       const double r1,               // End radio
                       const size_t rPoints,          // Number of points between r0-r1
                       const double time,             // Time at solution
                       const double gamma_i,          // Adiabatic coeficient
                       const double rho0,             // Initial density
                       const double vel0,             // Initial velocity
                       const string outfile)          // Output solution filename
    {
        vector<double> r  (rPoints);
        vector<double> rho(rPoints);
        vector<double> u  (rPoints);
        vector<double> p  (rPoints);
        vector<double> vel(rPoints);

        double rStep = (r1 - r0) / rPoints;
        for(size_t i = 0; i < rPoints; i++)
        {
            r[i] = r0 + (0.5 * rStep) + (i * rStep);
        }

        // Calculate theoretical solution
        NohSol(dim, rPoints, time,
               gamma_i,
               rho0, vel0,
               r, rho, u, p, vel);

        // Write solution file
        NohSolutionWriter::dump1DToAsciiFile(rPoints, r, rho, u, p, vel, outfile);
    }

private:

    static void NohSol(const size_t xgeom,            // geometry factor: 1=planar, 2=cylindircal, 3=spherical
                       const size_t rPoints,          // Number of points between r0-r1
                       const double time,             // temporal point where solution is desired [seconds]
                       const double gamma,            // gamma law equation of state
                       const double rho0,             // ambient density g/cm**3 in 'rho = rho0 * r**(-omega)'
                       const double vel0,             // ambient material speed [cm/s]
                       const vector<double>& r,       // spatial points where solution is desired [cm]
                       vector<double>& rho,           // out: density  [g/cm**3]
                       vector<double>& u,             // out: specific internal energy [erg/g]
                       vector<double>& p,             // out: presssure [erg/cm**3]
                       vector<double>& vel)           // out: velocity [cm/s]
    {
        // Frequest combination variables
        double gamm1  = gamma - 1.;
        double gamp1  = gamma + 1.;
        double gpogm  = gamp1 / gamm1;
        double xgm1   = xgeom - 1.;

        // Shock position
        double r2     = 0.5  * gamm1 * abs(vel0) * time;

        // Immediate post-shock using strong shock relations
        //double rhop = rho0 * pow(1. - (vel0 * time / r2), xgm1);

        // Loop over spatial positions
        for(size_t i = 0; i < rPoints; i++)
        {
            // Spatial point where solution is searched [cm]
            double xpos = r[i];

            if (xpos > r2)
            {
                  // if we are farther out than the shock front
                  rho[i]  = rho0 * pow(1. - (vel0 * time / xpos), xgm1);
                  u[i]    = 0.;
                  p[i]    = 0.;
                  vel[i]  = vel0;
            }
            else
            {
                  // if we are between the origin and the shock front
                  rho[i]  = rho0  * pow(gpogm, xgeom);
                  u[i]    = 0.5   * pow(vel0,  2.);
                  p[i]    = gamm1 * rho[i] * u[i];
                  vel[i]  = 0.;
            }
        }
    }

};
