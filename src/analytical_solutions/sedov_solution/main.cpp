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
 * @brief This program generate the analytical sedov solution based in the time and the initial conditions
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
#include <numeric>

#include "cstone/primitives/gather.hpp"

#include "arg_parser.hpp"
#include "particles_data.hpp"

#include "sedov/sedov_data_file_reader.hpp"
#include "sedov/sedov_data_generator.hpp"

#include "sedov_io.hpp"
#include "sedov_solution.hpp"

using namespace std;
using namespace cstone;
using namespace sphexa;

using T       = double;
using I       = uint64_t;
using Dataset = ParticlesData<T, I>;

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
    const bool   only_sol  = parser.exists("--only_solution");
    const T      time      = parser.getDouble("--time", 0.);
    const I      nParts    = parser.getInt("--nParts", 0);
    const string inputFile = parser.getString("--input", "./dump_sedov0.txt");
    const bool   ascii     = parser.exists("--ascii");
    const bool   complete  = parser.exists("--complete");
    const string outDir    = parser.getString("--outDir", "./");

    // Get time without rounding
    ostringstream time_long;
    time_long << time;
    string time_str = time_long.str();

    // Calculate and write theoretical solution profile in one dimension
    const I dim    = SedovDataGenerator<T, I>::dim;         // 3
    const T r0     = SedovDataGenerator<T, I>::r0;          // .0
    const T r1     = SedovDataGenerator<T, I>::r1;          // 1.
    const T eblast = SedovDataGenerator<T, I>::energyTotal; // 1.
    const T gamma  = SedovDataGenerator<T, I>::gamma;       // 5./3.
    const T omega  = SedovDataGenerator<T, I>::omega;       // 0.
    const T rho0   = SedovDataGenerator<T, I>::rho0;        // 1.
    const T u0     = SedovDataGenerator<T, I>::u0;          // 0.
    const T p0     = SedovDataGenerator<T, I>::p0;          // 0.
    const T vel0   = SedovDataGenerator<T, I>::vel0;        // 0.
    const T cs0    = SedovDataGenerator<T, I>::cs0;         // 0.

    // Position variables for solution
    vector<T> rSol;
    I         nSteps;

    // Set the positions for calculate the solution, in case 'only_solution' or '!complete'
    if (only_sol || !complete)
    {
        nSteps = 1000;

        const T rMax  = 2. * r1;
        const T rStep = (rMax - r0) / nSteps;

        for (I i = 0; i < nSteps; i++)
        {
            rSol.push_back(r0 + (0.5 * rStep) + (i * rStep));
        }
    }

    if (!only_sol)
    {
        const IFileReader<Dataset>& fileReader = SedovDataFileReader<Dataset>();

        // Load ParticlesData
        Dataset d = (ascii) ? fileReader.readParticleDataFromAsciiFile(inputFile, nParts)
                            : fileReader.readParticleDataFromBinFile(inputFile, nParts);

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

        // Compute radius of each particle from x, y, z
        vector<T> radii(nParts);
        for (I i = 0; i < nParts; i++)
        {
            radii[i] = sqrt((d.x[i] * d.x[i]) + (d.y[i] * d.y[i]) + (d.z[i] * d.z[i]));
        }

        // Order radius
        vector<I> ordering(nParts);
        iota(begin(ordering), end(ordering), I(0));
        sort_by_key(begin(radii), end(radii), begin(ordering));

        // Sort ParticleData properties by radius order
        reorderInPlace(ordering, d.x.data());
        reorderInPlace(ordering, d.y.data());
        reorderInPlace(ordering, d.z.data());
        reorderInPlace(ordering, d.vx.data());
        reorderInPlace(ordering, d.vy.data());
        reorderInPlace(ordering, d.vz.data());
        reorderInPlace(ordering, d.ro.data());
        reorderInPlace(ordering, d.u.data());
        reorderInPlace(ordering, d.p.data());
        reorderInPlace(ordering, d.h.data());
        reorderInPlace(ordering, d.m.data());
        reorderInPlace(ordering, d.c.data());

        if (complete)
        {
            nSteps = nParts;
            for (I i = 0; i < nSteps; i++)
            {
                cout << "radii[" << i << "]=" << radii[i] << endl;
                rSol.push_back(radii[i]);
            }
        }

        const string simFile = outDir + "sedov_simulation_" + time_str + ".dat";
        cout << "Simulation file: '" << simFile << "'";

        // Write 1D simulation solution to compare with the theoretical solution
        SedovSolutionDataFile<T, I, Dataset>::writeParticle1D(nParts,
                                                              d,
                                                              SedovSolution<T, I>::rho_shock,
                                                              SedovSolution<T, I>::u_shock,
                                                              SedovSolution<T, I>::p_shock,
                                                              SedovSolution<T, I>::vel_shock,
                                                              SedovSolution<T, I>::cs_shock,
                                                              rho0,
                                                              simFile);
    }

    const string solFile = outDir + "sedov_solution_" + time_str + ".dat";
    cout << "Solution   file: '" << solFile << "'";

    // Calculate Sedov solution
    SedovSolution<T, I>::create(rSol, dim, nSteps, time, eblast, omega, gamma, rho0, u0, p0, vel0, cs0, solFile);

    cout << "\nColumns:\n";
    SedovSolutionDataFile<T, I, Dataset>::writeColumns1D(cout);

    cout << "\nExecuted successfully.\n";
    cout << endl;

    exit(EXIT_SUCCESS);
}

void printHelp(char* binName)
{
    printf("\nUsage:\n\n");
    printf("%s [OPTIONS]\n", binName);
    printf("\nWhere possible options are:\n\n");

    printf("\t--only_solution FLAG \t\t Calculate only the solution with n=1000 steps [false].\n\n");

    printf("\t--time          NUM  \t\t Time where the solution is calculated (secs) [0.].\n\n");

    printf("\t--nParts        PATH \t\t Number of particles in the data file [0].\n");
    printf("\t--input         PATH \t\t Path to input particle data file [./dump_sedov0.txt].\n");
    printf("\t--ascii         FLAG \t\t Read file in ASCII format [false].\n");
    printf("\t--complete      FLAG \t\t Calculate the solution for each particle [false].\n\n");

    printf("\t--outPath       PATH \t\t Path to directory where output will be saved [./].\
                 \n\t\t\t\t\t Note that directory must exist and be provided with ending slash.\
                 \n\t\t\t\t\t Example: --outDir /home/user/folderToSaveOutputFiles/\n");
}
