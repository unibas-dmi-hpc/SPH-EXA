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

#include "arg_parser.hpp"
#include "particles_data.hpp"

#include "sedov/sedov_data_file_reader.hpp"
#include "sedov/sedov_data_generator.hpp"

#include "sedov_io.hpp"
#include "sedov_solution.hpp"

using namespace std;
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
    const bool   only_sol  = parser.exists("--only_solution") ? true : false;
    const T      time      = parser.getDouble("--time", 0.);
    const I      nParts    = parser.getInt("--nParts", 0);
    const string inputFile = parser.getString("--input", "./dump_sedov0.txt");
    const string outDir    = parser.getString("--outDir", "./");
    const bool   complete  = parser.exists("--complete") ? true : false;

    // Load ParticlesData
    const IFileReader<Dataset>& fileReader = SedovDataFileReader<Dataset>();
    auto                        d          = fileReader.readParticleDataFromBinFile(inputFile, nParts);

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

        // Sorting ParticleData by radius
        // sort(vSim.begin(), vSim.end(), ParticleIO<T, I>::cmp());
    }

    // Get time without rounding
    ostringstream time_long;
    time_long << time;
    string time_str = time_long.str();

    // Calculate and write theoretical solution profile in one dimension
    const I      dim     = SedovDataGenerator<T, I>::dim;
    const T      r0      = SedovDataGenerator<T, I>::r0;
    const T      r1      = SedovDataGenerator<T, I>::r1;
    const T      eblast  = SedovDataGenerator<T, I>::energyTotal;
    const T      gamma   = SedovDataGenerator<T, I>::gamma;
    const T      omega   = SedovDataGenerator<T, I>::omega;
    const T      rho0    = SedovDataGenerator<T, I>::rho0;
    const T      u0      = SedovDataGenerator<T, I>::u0;
    const T      p0      = SedovDataGenerator<T, I>::p0;
    const T      vel0    = SedovDataGenerator<T, I>::vel0;
    const T      cs0     = SedovDataGenerator<T, I>::cs0;
    const string solFile = outDir + "sedov_solution_" + time_str + ".dat";

    // Set the positions for calculate the solution
    vector<T> rSol;
    I         nSteps = 1000;
    if (only_sol || !complete)
    {
        const T rMax  = 2. * r1;
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
            T radius = sqrt((d.x[i] * d.x[i]) + (d.y[i] * d.y[i]) + (d.z[i] * d.z[i]));
            rSol.push_back(radius);
        }
    }

    // Calculate Sedov solution
    SedovSolution<T, I>::create(rSol, dim, nSteps, time, eblast, omega, gamma, rho0, u0, p0, vel0, cs0, solFile);

    // Write Info: Output files and colums.
    cout << "\nExecuted successfully.\n";
    cout << "Solution   file: '" << solFile << "'";

    if (!only_sol)
    {
        // Write 1D simulation solution to compare with the theoretical solution
        const string simFile = outDir + "sedov_simulation_" + time_str + ".dat";
        SedovSolutionDataFile<T, I, Dataset>::writeParticle1D(nParts,
                                                              d,
                                                              SedovSolution<T, I>::rho_shock,
                                                              SedovSolution<T, I>::u_shock,
                                                              SedovSolution<T, I>::p_shock,
                                                              SedovSolution<T, I>::vel_shock,
                                                              SedovSolution<T, I>::cs_shock,
                                                              rho0,
                                                              simFile);
        cout << "Simulation file: '" << simFile << "'";
    }

    cout << "\nColumns:\n";
    SedovSolutionDataFile<T, I, Dataset>::writeColumns1D(cout);
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
    printf("\t--input         PATH \t\t Path to input particle data file [./dump_sedov0.txt].\n\n");

    printf("\t--outPath       PATH \t\t Path to directory where output will be saved [./].\
                 \n\t\t\t\t\t Note that directory must exist and be provided with ending slash.\
                 \n\t\t\t\t\t Example: --outDir /home/user/folderToSaveOutputFiles/\n\n");

    printf("\t--complete      FLAG \t\t Calculate the solution for each particle [False]\n");
}
