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
 * @brief This class manage the data layer in the application.
 *        It provide a variety of read/write functions based in the number of dimensions.
 *
 * @author Jose A. Escartin <ja.escartin@gmail.com>
 *
 */

#pragma once

#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>

using namespace std;

template<typename T, typename I>
class SedovSolutionFile
{

private:
    SedovSolutionFile(); // Singleton

public:
    static void writeColumns1D(ostream& out) //
    {
        out << setw(16) << "#           01:r" // Column : position 1D     (Real  value     )
            << setw(16) << "02:rho"           // Column : density         (Real  value     )
            << setw(16) << "03:u"             // Column : internal energy (Real  value     )
            << setw(16) << "04:p"             // Column : pressure        (Real  value     )
            << setw(16) << "05:vel"           // Column : velocity 1D     (Real  value     )
            << setw(16) << "06:cs"            // Column : sound speed     (Real  value     )
            << setw(16) << "07:rho/rhoShock"  // Column : density         (Shock Normalized)
            << setw(16) << "08:u/uShock"      // Column : internal energy (Shock Normalized)
            << setw(16) << "09:p/pShock"      // Column : pressure        (Shock Normalized)
            << setw(16) << "10:vel/velShock"  // Column : velocity        (Shock Normalized)
            << setw(16) << "11:cs/csShock"    // Column : sound speed     (Shock Normalized)
            << setw(16) << "12:rho/rho0"      // Column : density         (Init  Normalized)
            << endl;
    }

    static void writeData1D(const I          n,         //
                            const vector<T>& r,         //
                            const vector<T>& rho,       //
                            const vector<T>& u,         //
                            const vector<T>& p,         //
                            const vector<T>& vel,       //
                            const vector<T>& cs,        //
                            const T          rho_shock, //
                            const T          u_shock,   //
                            const T          p_shock,   //
                            const T          vel_shock, //
                            const T          cs_shock,  //
                            const T          rho0,      //
                            const string&    outfile)
    {
        try
        {
            ofstream out(outfile);

            // Write Colums
            writeColumns1D(out);

            // Write Data
            for (I i = 0; i < n; i++)
            {
                out << setw(16) << setprecision(6) << scientific << r[i]               //
                    << setw(16) << setprecision(6) << scientific << rho[i]             //
                    << setw(16) << setprecision(6) << scientific << u[i]               //
                    << setw(16) << setprecision(6) << scientific << p[i]               //
                    << setw(16) << setprecision(6) << scientific << vel[i]             //
                    << setw(16) << setprecision(6) << scientific << cs[i]              //
                    << setw(16) << setprecision(6) << scientific << rho[i] / rho_shock //
                    << setw(16) << setprecision(6) << scientific << u[i] / u_shock     //
                    << setw(16) << setprecision(6) << scientific << p[i] / p_shock     //
                    << setw(16) << setprecision(6) << scientific << vel[i] / vel_shock //
                    << setw(16) << setprecision(6) << scientific << cs[i] / cs_shock   //
                    << setw(16) << setprecision(6) << scientific << rho[i] / rho0      //
                    << endl;
            }

            out.close();
        }
        catch (exception& ex)
        {
            cout << "ERROR: %s. Terminating\n" << ex.what() << endl;
            exit(EXIT_FAILURE);
        }
    }
};

template<typename T, typename I, typename Dataset>
struct SedovSolutionDataFile : SedovSolutionFile<T, I>
{
private:
    SedovSolutionDataFile(); // Singleton

public:
    static void writeParticle1D(const I        n,         //
                                const Dataset& d,         //
                                const T        rho_shock, //
                                const T        u_shock,   //
                                const T        p_shock,   //
                                const T        vel_shock, //
                                const T        cs_shock,  //
                                const T        rho0,      //
                                const string&  outfile)
    {
        vector<T> r(n);
        vector<T> vel(n);

        for (I i = 0; i < n; i++)
        {
            r[i]   = sqrt((d.x[i] * d.x[i]) + (d.y[i] * d.y[i]) + (d.z[i] * d.z[i]));
            vel[i] = sqrt((d.vx[i] * d.vx[i]) + (d.vy[i] * d.vy[i]) + (d.vz[i] * d.vz[i]));
        }

        SedovSolutionFile<T, I>::writeData1D(
            n, r, d.ro, d.u, d.p, vel, d.c, rho_shock, u_shock, p_shock, vel_shock, cs_shock, rho0, outfile);
    }
};
