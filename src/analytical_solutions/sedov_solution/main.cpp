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
#include <iostream>
#include <cmath>
#include <vector>
#include <filesystem>
#include <algorithm>

#include "ArgParser.hpp"
#include "sedov/SedovDataGenerator.hpp"

#include "file_data.hpp"
#include "sedov_solution.hpp"

using namespace std;
using namespace sphexa;

using Real    = double;
using KeyType = uint64_t;

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
    const double time      = parser.getDouble( "--time",   0.                  );
    const size_t nParts    = parser.getInt(    "--nParts", 0                   );
    const string inputFile = parser.getString( "--input",  "./dump_sedov0.txt" );
    const string outDir    = parser.getString( "--outDir", "./"                );
    const bool   complete  = parser.exists("--complete") ? true : false;

    if (nParts <= 0){
        cout << "ERROR: --nParts: '" << nParts << "' should be > 0." << endl;
        exit(EXIT_FAILURE);
    }
    else if (!filesystem::exists(inputFile)){
        cout << "ERROR: --input file: '" << inputFile << "' don't exist." << endl;
        exit(EXIT_FAILURE);
    }

    // Initialize size
    vector<double> x  (nParts);
    vector<double> y  (nParts);
    vector<double> z  (nParts);
    vector<double> vx (nParts);
    vector<double> vy (nParts);
    vector<double> vz (nParts);
    vector<double> h  (nParts);
    vector<double> rho(nParts);
    vector<double> u  (nParts);
    vector<double> p  (nParts);
    vector<double> cs (nParts);
    vector<double> Px (nParts);
    vector<double> Py (nParts);
    vector<double> Pz (nParts);

    // Load particles data
    FileData::readData3D(
        inputFile,
        nParts,
        x,y,z,
        vx,vy,vz,
        h,
        rho,u,p,
        cs,
        Px,Py,Pz);

    // Calculate radius, velocity and local speed of sound
    vector<double> rSim(nParts);
    vector<double> vel(nParts);
    for(size_t i = 0; i < nParts; i++)
    {
        rSim[i] = sqrt( pow(x[i], 2) + pow(y[i], 2) + pow(z[i], 2) );
        vel[i]  = sqrt( pow(vx[i],2) + pow(vy[i],2) + pow(vz[i],2) );
    }

    // Get time without rounding
    ostringstream time_long;
    time_long << time;
    string time_str = time_long.str();

    // Calculate and write theoretical solution profile in one dimension
    const size_t dim     = SedovDataGenerator<Real, KeyType>::dim;
    const double r0      = SedovDataGenerator<Real, KeyType>::r0;
    const double r1      = SedovDataGenerator<Real, KeyType>::r1;
    const double eblast  = SedovDataGenerator<Real, KeyType>::energyTotal;
    const double gamma   = SedovDataGenerator<Real, KeyType>::gamma;
    const double omega   = SedovDataGenerator<Real, KeyType>::omega;
    const double rho0    = SedovDataGenerator<Real, KeyType>::rho0;
    const double u0      = SedovDataGenerator<Real, KeyType>::u0;
    const double p0      = SedovDataGenerator<Real, KeyType>::p0;
    const double vr0     = SedovDataGenerator<Real, KeyType>::vr0;
    const double cs0     = SedovDataGenerator<Real, KeyType>::cs0;
    const string solFile = outDir + "sedov_solution_" + time_str + ".dat";

    // Set the positions for calculate the solution
    vector<double> rSol;
    size_t nSteps;
    if (complete)
    {
        nSteps = nParts;

        for(size_t i = 0; i < nSteps; i++)
        {
            rSol.push_back(rSim[i]);
        }
        sort(rSol.begin(), rSol.end());
    }
    else
    {
        nSteps = 1000;

        const double rMax  = 2. * r1;
        const double rStep = (rMax - r0) / nSteps;

        for(size_t i = 0; i < nSteps; i++)
        {
            rSol.push_back(r0 + (0.5 * rStep) + (i * rStep));
        }
    }

    // Calculate Sedov solution
    SedovSolution::create(
        rSol,
        dim,
        nSteps,
        time,
        eblast,
        omega, gamma,
        rho0, u0, p0, vr0, cs0,
        solFile);

    // Write 1D simulation solution to compare with the theoretical solution
    const string simFile = outDir + "sedov_simulation_" + time_str + ".dat";
    FileData::writeData1D(
        nParts,
        rSim,
        rho,u,p,
        vel,cs,
        SedovSolution::rho_shock,
        SedovSolution::u_shock,
        SedovSolution::p_shock,
        SedovSolution::vel_shock,
        SedovSolution::cs_shock,
        rho0,
        simFile);

    // Write Info: Output files and colums.
    cout << "\nExecuted successfully.\n\n"
         << "Solution   file: '" <<  solFile << "'\n"
         << "Simulation file: '" <<  simFile << "'\n"
         << "\nColumns:\n";
    FileData::writeColumns1D(cout);
    cout << endl;

    exit(EXIT_SUCCESS);
}


void printHelp(char* binName)
{
    printf("\nUsage:\n\n");
    printf("%s [OPTIONS]\n", binName);
    printf("\nWhere possible options are:\n\n");

    printf("\t--time     NUM  \t\t Time where the solution is calculated (secs) [0.]\n\n");

    printf("\t--nParts   PATH \t\t Number of particles in the data file [0].\n");
    printf("\t--input    PATH \t\t Path to input particle data file [./dump_sedov0.txt].\n\n");

    printf("\t--outPath  PATH \t\t Path to directory where output will be saved [./].\
                \n\t\t\t\t Note that directory must exist and be provided with ending slash.\
                \n\t\t\t\t Example: --outDir /home/user/folderToSaveOutputFiles/\n\n");

    printf("\t--complete FLAG \t\t Calculate the solution for each particle [False]\n");
}
