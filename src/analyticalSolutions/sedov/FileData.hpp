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

#pragma once

#include <string>
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <fstream>

using namespace std;

class FileData
{

private:

    FileData();   // Singleton

public:

    static void read3D(
        const string&   inputFile,
        const double    nParts,
        vector<double>& x,
        vector<double>& y,
        vector<double>& z,
        vector<double>& vx,
        vector<double>& vy,
        vector<double>& vz,
        vector<double>& rho,
        vector<double>& u,
        vector<double>& p,
        vector<double>& h,
        vector<double>& m)
    {
        try
        {
            size_t i = 0;

            ifstream in(inputFile);
            string line;

            while(getline(in,line))
            {
                istringstream iss(line);
                if (i < nParts){
                    iss >> x[i];
                    iss >> y[i];
                    iss >> z[i];
                    iss >> vx[i];
                    iss >> vy[i];
                    iss >> vz[i];
                    iss >> rho[i];
                    iss >> u[i];
                    iss >> p[i];
                    iss >> h[i];
                    iss >> m[i];
                }

                i++;
            }

            if (nParts != i){
                cout << "ERROR: number of particles doesn't match with the expect (nParts=" << nParts << ", ParticleDataLines=" << i << ")" << endl;
                exit(EXIT_FAILURE);
            }

            in.close();
        }
        catch (exception &ex)
        {
            cout << "ERROR: %s. Terminating\n" << ex.what() << endl;
            exit(EXIT_FAILURE);
        }
    }

    static void write1D(
        const size_t          n,
        const vector<double>& r,
        const vector<double>& rho,
        const vector<double>& u,
        const vector<double>& p,
        const vector<double>& vel,
        const vector<double>& cs,
        const double          rho0,
        const double          rho_shock,
        const double          p_shock,
        const double          vel_shock,
        const string&         outfile)
    {
        try
        {
            ofstream out(outfile);

            out << " " << setw(15) << "r"              // Column 01 : position 1D
                << " " << setw(15) << "rho"            // Column 02 : density         (Real value)
                << " " << setw(15) << "u"              // Column 03 : internal energy (Real value)
                << " " << setw(15) << "p"              // Column 04 : pressure        (Real value)
                << " " << setw(15) << "vel"            // Column 05 : velocity 1D     (Real value)
                << " " << setw(15) << "cs"             // Column 06 : sound speed     (Real value)
                << " " << setw(15) << "rho/rho0"       // Column 07 : density         (Normalized)
                << " " << setw(15) << "rho/rhoShock"   // Column 08 : density         (Shock Normalized)
                << " " << setw(15) << "p/pShock"       // Column 09 : pressure        (Shock Normalized)
                << " " << setw(15) << "vel/velShock"   // Column 10 : velocity        (Shock Normalized)
                << endl;

            for(size_t i = 0; i < n; i++)
            {
                out << " " << setw(15) << setprecision(6) << scientific << r[i]
                    << " " << setw(15) << setprecision(6) << scientific << rho[i]
                    << " " << setw(15) << setprecision(6) << scientific << u[i]
                    << " " << setw(15) << setprecision(6) << scientific << p[i]
                    << " " << setw(15) << setprecision(6) << scientific << vel[i]
                    << " " << setw(15) << setprecision(6) << scientific << cs[i]
                    << " " << setw(15) << setprecision(6) << scientific << rho[i] / rho0
                    << " " << setw(15) << setprecision(6) << scientific << rho[i] / rho_shock
                    << " " << setw(15) << setprecision(6) << scientific << p[i]   / p_shock
                    << " " << setw(15) << setprecision(6) << scientific << vel[i] / vel_shock
                    << endl;
            }

            out.close();
        }
        catch (exception &ex)
        {
            fprintf(stderr, "ERROR: %s. Terminating\n", ex.what());
            exit(EXIT_FAILURE);
        }
    }
};
