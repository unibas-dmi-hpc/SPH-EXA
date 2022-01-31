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

#include "noh_solution.hpp"

#include <iostream>
#include <cmath>

#include "noh_io.hpp"

void NohSolution::create(
    vector<double>& r,
    const size_t    dim,
    const size_t    rPoints,
    const double    time,
    const double    gamma_i,
    const double    rho0,
    const double    vel0,
    const string    outfile)
{
    vector<double> rho(rPoints);
    vector<double> u  (rPoints);
    vector<double> p  (rPoints);
    vector<double> vel(rPoints);
    vector<double> cs (rPoints);

    // Calculate theoretical solution
    NohSol(
        dim, rPoints, time,
        gamma_i,
        rho0, vel0,
        r, rho, u, p, vel, cs);

    // Write solution file
    NohFileData::writeData1D(
        rPoints,
        r,
        rho,u,p,
        vel,cs,
        outfile);
}

void NohSolution::NohSol(
    const size_t          xgeom,
    const size_t          rPoints,
    const double          time,
    const double          gamma,
    const double          rho0,
    const double          vel0,
    const vector<double>& r,
    vector<double>&       rho,
    vector<double>&       u,
    vector<double>&       p,
    vector<double>&       vel,
    vector<double>&       cs)
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
              cs[i]   = 0.;
        }
        else
        {
              // if we are between the origin and the shock front
              rho[i]  = rho0  * pow(gpogm, xgeom);
              u[i]    = 0.5   * pow(vel0,  2.);
              p[i]    = gamm1 * rho[i] * u[i];
              vel[i]  = 0.;
              cs[i]   = sqrt(gamma * p[i] / rho[i]);;
        }
    }
}
