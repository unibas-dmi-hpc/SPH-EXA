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

#include "FileData.hpp"

#include <iostream>
#include <iomanip>

void FileData::readData3D(
    const string&   inputFile,
    const double    nParts,
    vector<double>& x,
    vector<double>& y,
    vector<double>& z,
    vector<double>& vx,
    vector<double>& vy,
    vector<double>& vz,
    vector<double>& h,
    vector<double>& rho,
    vector<double>& u,
    vector<double>& p,
    vector<double>& cs,
    vector<double>& Px,
    vector<double>& Py,
    vector<double>& Pz)
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
                iss >> h[i];
                iss >> rho[i];
                iss >> u[i];
                iss >> p[i];
                iss >> cs[i];
                iss >> Px[i];
                iss >> Py[i];
                iss >> Pz[i];
            }

            i++;
        }

        if (nParts != i){
            cout << "ERROR: number of particles doesn't match with the expect (nParts=" << nParts << ", ParticleDataLines=" << i << ")." << endl;
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

void FileData::writeColumns1D(ostream& out)
{
    out << setw(16) << "#           01:r"  // Column : position 1D     (Real  value     )
        << setw(16) << "02:rho"            // Column : density         (Real  value     )
        << setw(16) << "03:u"              // Column : internal energy (Real  value     )
        << setw(16) << "04:p"              // Column : pressure        (Real  value     )
        << setw(16) << "05:vel"            // Column : velocity 1D     (Real  value     )
        << setw(16) << "06:cs"             // Column : sound speed     (Real  value     )
        << setw(16) << "07:rho/rhoShock"   // Column : density         (Shock Normalized)
        << setw(16) << "08:u/uShock"       // Column : internal energy (Shock Normalized)
        << setw(16) << "09:p/pShock"       // Column : pressure        (Shock Normalized)
        << setw(16) << "10:vel/velShock"   // Column : velocity        (Shock Normalized)
        << setw(16) << "11:cs/csShock"     // Column : sound speed     (Shock Normalized)
        << setw(16) << "12:rho/rho0"       // Column : density         (Init  Normalized)
        << endl;
}

void FileData::writeData1D(
    const size_t          n,
    const vector<double>& r,
    const vector<double>& rho,
    const vector<double>& u,
    const vector<double>& p,
    const vector<double>& vel,
    const vector<double>& cs,
    const double          rho_shock,
    const double          u_shock,
    const double          p_shock,
    const double          vel_shock,
    const double          cs_shock,
    const double          rho0,
    const string&         outfile)
{
    try
    {
        ofstream out(outfile);

        // Write Colums
        writeColumns1D(out);

        // Write Data
        for(size_t i = 0; i < n; i++)
        {
            out << setw(16) << setprecision(6) << scientific << r[i]
                << setw(16) << setprecision(6) << scientific << rho[i]
                << setw(16) << setprecision(6) << scientific << u[i]
                << setw(16) << setprecision(6) << scientific << p[i]
                << setw(16) << setprecision(6) << scientific << vel[i]
                << setw(16) << setprecision(6) << scientific << cs[i]
                << setw(16) << setprecision(6) << scientific << rho[i] / rho_shock
                << setw(16) << setprecision(6) << scientific << u[i]   / u_shock
                << setw(16) << setprecision(6) << scientific << p[i]   / p_shock
                << setw(16) << setprecision(6) << scientific << vel[i] / vel_shock
                << setw(16) << setprecision(6) << scientific << cs[i]  / cs_shock
                << setw(16) << setprecision(6) << scientific << rho[i] / rho0
                << endl;
        }

        out.close();
    }
    catch (exception &ex)
    {
        cout << "ERROR: %s. Terminating\n" << ex.what() << endl;
        exit(EXIT_FAILURE);
    }
}
