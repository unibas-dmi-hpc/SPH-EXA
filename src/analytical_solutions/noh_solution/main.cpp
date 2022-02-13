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
 * @brief This program generate the analytical noh solution based in the time and the initial conditions
 *
 * @author Jose A. Escartin <ja.escartin@gmail.com>
 *
 */

#include <string>
#include <sstream>
#include <iostream>
#include <cmath>
#include <vector>
#include <filesystem>
#include <algorithm>

#include "arg_parser.hpp"

#include "noh/noh_data_generator.hpp"

#include "analytical_solutions/common/particle_io.hpp"

#include "noh_io.hpp"
#include "noh_solution.hpp"

using namespace std;
using namespace sphexa;

using T = double;
using I = uint64_t;

void printHelp(char* binName);

int main(int argc, char** argv)
{
    const ArgParser parser(argc, argv);

    if (parser.exists("-h") || parser.exists("--h") || parser.exists("-help") || parser.exists("--help"))
    {
        printHelp(argv[0]);
        exit(EXIT_SUCCESS);
    }

    // Get command line parameters
    const bool   only_sol  = parser.exists("--only_solution") ? true : false;
    const T      time      = parser.getDouble("--time", 0.);
    const I      nParts    = parser.getInt("--nParts", 0);
    const string inputFile = parser.getString("--input", "./dump_noh0.txt");
    const string outDir    = parser.getString("--outDir", "./");
    const bool   complete  = parser.exists("--complete") ? true : false;

    // Initialize vector size
    vector<T>                x(nParts);
    vector<T>                y(nParts);
    vector<T>                z(nParts);
    vector<T>                vx(nParts);
    vector<T>                vy(nParts);
    vector<T>                vz(nParts);
    vector<T>                h(nParts);
    vector<T>                rho(nParts);
    vector<T>                u(nParts);
    vector<T>                p(nParts);
    vector<T>                cs(nParts);
    vector<T>                Px(nParts);
    vector<T>                Py(nParts);
    vector<T>                Pz(nParts);
    vector<ParticleIO<T, I>> vSim(nParts);

    if (!only_sol)
    {
        if (nParts <= 0)
        {
            cout << "ERROR: --nParts: '" << nParts << "' should be > 0." << endl;
            exit(EXIT_FAILURE);
        }
        else if (!filesystem::exists(inputFile))
        {
            cout << "ERROR: --input file: '" << inputFile << "' don't exist." << endl;
            exit(EXIT_FAILURE);
        }

        // Load particles data
        NohFileData<T, I>::readData3D(inputFile, nParts, x, y, z, vx, vy, vz, h, rho, u, p, cs, Px, Py, Pz);

        // Calculate radius, velocity and sort particle data by radius
        for (I i = 0; i < nParts; i++)
        {
            T r   = sqrt((x[i] * x[i]) + (y[i] * y[i]) + (z[i] * z[i]));
            T vel = sqrt((vx[i] * vx[i]) + (vy[i] * vy[i]) + (vz[i] * vz[i]));

            vSim[i] = {
                i, r, vel, x[i], y[i], z[i], vx[i], vy[i], vz[i], h[i], rho[i], u[i], p[i], cs[i], Px[i], Py[i], Pz[i]};
        }
        sort(vSim.begin(), vSim.end(), ParticleIO<T, I>::cmp());
    }

    // Get time without rounding
    ostringstream time_long;
    time_long << time;
    string time_str = time_long.str();

    // Calculate and write theoretical solution profile in one dimension    // time = 0.6s
    const size_t dim     = NohDataGenerator<T, I>::dim;   // 3
    const double r0      = NohDataGenerator<T, I>::r0;    // .0
    const double r1      = NohDataGenerator<T, I>::r1;    // .1
    const double gamma   = NohDataGenerator<T, I>::gamma; // 5./3.
    const double rho0    = NohDataGenerator<T, I>::rho0;  // 1.
    const double u0      = NohDataGenerator<T, I>::u0;    // 0.
    const double p0      = NohDataGenerator<T, I>::p0;    // 0.
    const double vr0     = NohDataGenerator<T, I>::vr0;   // -1.
    const double cs0     = NohDataGenerator<T, I>::cs0;   // 0.
    const string solFile = outDir + "noh_solution_" + time_str + ".dat";

    // Set the positions for calculate the solution
    vector<T> rSol;
    I         nSteps = 1000;
    if (only_sol || !complete)
    {
        const T rMax  = r1; // 2. * r1;
        const T rStep = (rMax - r0) / nSteps;

        for (I i = 0; i < nSteps; i++)
        {
            rSol.push_back(r0 + (0.5 * rStep) + (i * rStep));
        }
    }
    else
    {
        nSteps = nParts;

        for (I i = 0; i < nSteps; i++)
        {
            rSol.push_back(vSim[i].r);
        }
    }

    // Calculate Sedov solution
    NohSolution<T, I>::create(rSol, dim, nSteps, time, gamma, rho0, u0, p0, vr0, cs0, solFile);

    // Write Info: Output files and colums.
    cout << "\nExecuted successfully.\n";
    cout << "Solution   file: '" << solFile << "'";

    if (!only_sol)
    {
        // Write 1D simulation solution to compare with the theoretical solution
        const string simFile = outDir + "noh_simulation_" + time_str + ".dat";
        NohFileData<T, I>::writeParticle1D(nParts, vSim, simFile);
        cout << "Simulation file: '" << simFile << "'";
    }

    cout << "\nColumns:\n";
    NohFileData<T, I>::writeColumns1D(cout);
    cout << endl;

    exit(EXIT_SUCCESS);
}

void printHelp(char* binName)
{
    printf("\nUsage:\n\n");
    printf("%s [OPTIONS]\n", binName);
    printf("\nWhere possible options are:\n\n");

    printf("\t--only_solution FLAG \t\t Calculate only the solution with n=1000 steps [False]\n\n");

    printf("\t--time          NUM  \t\t Time where the solution is calculated (secs) [0.]\n\n");

    printf("\t--nParts        PATH \t\t Number of particles in the data file [0].\n");
    printf("\t--input         PATH \t\t Path to input particle data file [./dump_noh0.txt].\n\n");

    printf("\t--outPath       PATH \t\t Path to directory where output will be saved [./].\
                 \n\t\t\t\t\t Note that directory must exist and be provided with ending slash.\
                 \n\t\t\t\t\t Example: --outDir /home/user/folderToSaveOutputFiles/\n\n");

    printf("\t--complete      FLAG \t\t Calculate the solution for each particle [False]\n");
}
