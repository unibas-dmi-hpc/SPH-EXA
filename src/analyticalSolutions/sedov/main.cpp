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

#include <string>
#include <cmath>
#include <vector>

#include "../../include/ArgParser.hpp"
#include "../../include/FileUtils.hpp"

#include "../../sedov/SedovDataGenerator.hpp"

#include "SedovAnalyticalSolution.hpp"
#include "FileData.hpp"

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

    const double time          = parser.getDouble("--time",   0.);
    const string inputFilePath = parser.getString("--input",  "./particles.dat");
    const string outDirectory  = parser.getString("--outDir", "./");

    const size_t nParts = SedovDataGenerator<Real, KeyType>::nParts;

    vector<double> x  (nParts);
    vector<double> y  (nParts);
    vector<double> z  (nParts);

    vector<double> vx (nParts);
    vector<double> vy (nParts);
    vector<double> vz (nParts);

    vector<double> r  (nParts);
    vector<double> vel(nParts);
    vector<double> cs (nParts);

    vector<double> rho(nParts);
    vector<double> u  (nParts);
    vector<double> p  (nParts);

    vector<double> h  (nParts);
    vector<double> m  (nParts);

    const size_t dim    = SedovDataGenerator<Real, KeyType>::dim;
    const double gamma  = SedovDataGenerator<Real, KeyType>::gamma;
    const double omega  = SedovDataGenerator<Real, KeyType>::omega;
    const double r0     = SedovDataGenerator<Real, KeyType>::r0;
    const double r1     = SedovDataGenerator<Real, KeyType>::r1;
    const double eblast = SedovDataGenerator<Real, KeyType>::energyTotal;
    const double rho0   = SedovDataGenerator<Real, KeyType>::rho0;
    const double u0     = SedovDataGenerator<Real, KeyType>::u0;
    const double p0     = SedovDataGenerator<Real, KeyType>::p0;
    const double vr0    = SedovDataGenerator<Real, KeyType>::vr0;
    const double cs0    = SedovDataGenerator<Real, KeyType>::cs0;

    string simulationFilename = "sim_sedov_" + to_string(time) + ".txt";
    string solutionFilename   = "sol_sedov_" + to_string(time) + ".txt";

    // Calculate and write theoretical solution profile in one dimension
    size_t nSteps = 1000;
    double rMax   = 2. * r1;
    SedovAnalyticalSolution::create(dim,
                                    r0, rMax,
                                    nSteps,
                                    time,
                                    eblast,
                                    omega, gamma,
                                    rho0, u0, p0, vr0, cs0,
                                    solutionFilename);

    for(size_t i = 0; i < nParts; i++)
    {
        // Load particles data
        sphexa::fileutils::readParticleDataFromBinFile(inputFilePath,
                                                       x[i],   y[i],  z[i],
                                                       vx[i],  vy[i], vz[i],
                                                       rho[i], u[i],  p[i],
                                                       h[i],   m[i]);

        // Calculate modules for position and velocity
        r[i]   = sqrt( pow(x[i], 2) + pow(y[i], 2) + pow(z[i], 2) );
        vel[i] = sqrt( pow(vx[i],2) + pow(vy[i],2) + pow(vz[i],2) );
    }

    // Write 1D simulation solution to compare with the theoretical solution
    FileData::dump1DToAsciiFile(nParts,
                                r, vel,cs,
                                rho, u, p,
                                rho0,
                                SedovAnalyticalSolution::rho_shock, SedovAnalyticalSolution::p_shock, SedovAnalyticalSolution::vel_shock,
                                simulationFilename);

    exit(EXIT_SUCCESS);
}


void printHelp(char* binName)
{
    printf("\nUsage:\n\n");
    printf("%s [OPTIONS]\n", binName);
    printf("\nWhere possible options are:\n\n");

    printf("\t--time   NUM  \t\t Time where the solution is calculated (secs) [0.]\n\n");

    printf("\t--input  PATH \t\t Path to input particle data file [./particles.dat].\n");

    printf("\t--outDir PATH \t\t Path to directory where output will be saved [./].\
                \n\t\t\t\t Note that directory must exist and be provided with ending slash.\
                \n\t\t\t\t Example: --outDir /home/user/folderToSaveOutputFiles/\n");
}
