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
#include <iostream>
#include <cmath>

using namespace std;

template<typename T, typename I>
class NohSolution
{
public:
    static void create(vector<T>&   r,           // Radius position
                       const I      dim,         // Dimensions
                       const I      rPoints,     // Number of points between r0-r1
                       const T      time,        // Time at solution
                       const T      gamma_i,     // Adiabatic coeficient
                       const T      rho0,        // Initial density
                       const T      u0,          // Initial internal energy
                       const T      p0,          // Initial pressure
                       const T      vel0,        // Initial velocity
                       const T      cs0,         // Initial sound speed
                       const string outfile)     // Output solution filename
    {
        vector<T> rho(rPoints);
        vector<T> u(rPoints);
        vector<T> p(rPoints);
        vector<T> vel(rPoints);
        vector<T> cs(rPoints);

        // Calculate theoretical solution
        NohSol(dim, rPoints, time, gamma_i, rho0, u0, p0, vel0, cs0, r, rho, u, p, vel, cs);

        // Write solution file
        NohFileData<T, I>::writeData1D(rPoints, r, rho, u, p, vel, cs, outfile);
    }

private:
    static void NohSol(const I          xgeom,   // geometry factor: 1=planar, 2=cylindircal, 3=spherical
                       const I          rPoints, // Number of points between r0-r1
                       const T          time,    // temporal point where solution is desired [seconds]
                       const T          gamma,   // gamma law equation of state
                       const T          rho0,    // ambient density g/cm**3 in 'rho = rho0 * r**(-omega)'
                       const T          u0,      // ambient internal energy [erg/g]
                       const T          p0,      // ambient pressure [erg/cm**3]
                       const T          vel0,    // ambient material speed [cm/s]
                       const T          cs0,     // ambient sound speed [cm/s]
                       const vector<T>& r,       // spatial points where solution is desired [cm]
                       vector<T>&       rho,     // out: density  [g/cm**3]
                       vector<T>&       u,       // out: specific internal energy [erg/g]
                       vector<T>&       p,       // out: presssure [erg/cm**3]
                       vector<T>&       vel,     // out: velocity [cm/s]
                       vector<T>&       cs)      // out: sound speed [cm/s]
    {
        // Frequest combination variables
        T gamm1 = gamma - 1.;
        T gamp1 = gamma + 1.;
        T gpogm = gamp1 / gamm1;
        T xgm1  = xgeom - 1.;

        // Shock position
        T r2 = 0.5 * gamm1 * abs(vel0) * time;

        // Immediate post-shock using strong shock relations
        // T rhop = rho0 * pow(1. - (vel0 * time / r2), xgm1);

        // Loop over spatial positions
        for (I i = 0; i < rPoints; i++)
        {
            // Spatial point where solution is searched [cm]
            T xpos = r[i];

            if (xpos > r2)
            {
                // if we are farther out than the shock front
                rho[i] = rho0 * pow(1. - (vel0 * time / xpos), xgm1);
                u[i]   = u0;
                p[i]   = p0;
                vel[i] = abs(vel0);
                cs[i]  = cs0;
            }
            else
            {
                // if we are between the origin and the shock front
                rho[i] = rho0 * pow(gpogm, xgeom);
                u[i]   = 0.5 * (vel0 * vel0);
                p[i]   = gamm1 * rho[i] * u[i];
                vel[i] = 0.;
                cs[i]  = sqrt(gamma * p[i] / rho[i]);
                ;
            }
        }
    }
};
