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
#include <numeric>

#include "arg_parser.hpp"
#include "file_utils.hpp"

#include "noh/noh_data_generator.hpp"
#include "noh_solution.hpp"

using namespace std;
using namespace cstone;
using namespace sphexa;

using T       = double;
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
    const T      time   = parser.getDouble("--time", 0.);
    const string outDir = parser.getString("--outDir", "./");
    // const bool   ascii  = parser.exists("--ascii");

    // Get time without rounding
    ostringstream time_long;
    time_long << time;
    string time_str = time_long.str();

    // Calculate and write theoretical solution profile in one dimension
    const size_t dim   = NohDataGenerator::dim;   // 3
    const double r0    = NohDataGenerator::r0;    // .0
    const double r1    = NohDataGenerator::r1;    // 1.
    const double gamma = NohDataGenerator::gamma; // 5./3.
    const double rho0  = NohDataGenerator::rho0;  // 1.
    const double u0    = NohDataGenerator::u0;    // 1.e-20
    const double p0    = NohDataGenerator::p0;    // 0.
    const double vr0   = NohDataGenerator::vr0;   // -1.
    const double cs0   = NohDataGenerator::cs0;   // 0.

    // Position variables for solution
    size_t    nSteps   = 100000;
    size_t    nSamples = nSteps + 2;
    vector<T> rSol(nSamples);

    // Set the positions for calculate the solution
    const T rMax  = r1;
    const T rStep = (rMax - r0) / nSteps;

    for (size_t i = 0; i < nSteps; i++)
    {
        rSol[i] = (r0 + (0.5 * rStep) + (i * rStep));
    }
    double shockFront  = nohShockFront(gamma, vr0, time);
    rSol[nSamples - 2] = shockFront;
    rSol[nSamples - 1] = shockFront + 1e-7;
    std::sort(begin(rSol), end(rSol));

    // analytical solution output
    std::vector<T> rho(nSamples), p(nSamples), u(nSamples), vel(nSamples), cs(nSamples);

    nohSol(dim, nSamples, time, gamma, rho0, u0, p0, vr0, cs0, rSol, rho, u, p, vel, cs);

    const string solFile = outDir + "noh_solution_" + time_str + ".txt";
    cout << "Solution   file: '" << solFile << "'";

    return EXIT_SUCCESS;
}

void printHelp(char* binName)
{
    printf("\nUsage:\n\n");
    printf("%s [OPTIONS]\n", binName);
    printf("\nWhere possible options are:\n\n");

    printf("\t--time          NUM  \t\t Time where the solution is calculated (secs) [0.].\n\n");

    printf("\t--outPath       PATH \t\t Path to directory where output will be saved [./].\
                 \n\t\t\t\t\t Note that directory must exist and be provided with ending slash.\
                 \n\t\t\t\t\t Example: --outDir /home/user/folderToSaveOutputFiles/\n");
}
