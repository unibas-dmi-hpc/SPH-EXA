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

#include "analytical_solutions/common/particle_io.hpp"

using namespace std;

template<typename T, typename I>
class NohFileData : public FileData<T, I>
{

private:
    NohFileData(); // Singleton

public:
    static void writeColumns1D(ostream& out)                                //
    {
        out << setw(16) << "#           01:r"  // Column : position 1D     (Real  value     )
            << setw(16) << "02:rho"            // Column : density         (Real  value     )
            << setw(16) << "03:u"              // Column : internal energy (Real  value     )
            << setw(16) << "04:p"              // Column : pressure        (Real  value     )
            << setw(16) << "05:vel"            // Column : velocity 1D     (Real  value     )
            << setw(16) << "06:cs"             // Column : sound speed     (Real  value     )
            << endl;
    }

    static void writeData1D(const I          n,                             //
                            const vector<T>& r,                             //
                            const vector<T>& rho,                           //
                            const vector<T>& u,                             //
                            const vector<T>& p,                             //
                            const vector<T>& vel,                           //
                            const vector<T>& cs,                            //
                            const string&    outfile)                       //
    {
        try
        {
            ofstream out(outfile);

            // Write Colums
            writeColumns1D(out);

            // Write Data
            for (size_t i = 0; i < n; i++)
            {
                out << setw(16) << setprecision(6) << scientific << r[i]    //
                    << setw(16) << setprecision(6) << scientific << rho[i]  //
                    << setw(16) << setprecision(6) << scientific << u[i]    //
                    << setw(16) << setprecision(6) << scientific << p[i]    //
                    << setw(16) << setprecision(6) << scientific << vel[i]  //
                    << setw(16) << setprecision(6) << scientific << cs[i]   //
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

    static void writeParticle1D(const I                         n,          //
                                const vector<ParticleIO<T, I>>& vParticle,  //
                                const string&                   outfile)    //
    {
        vector<T> r(n);
        vector<T> rho(n);
        vector<T> u(n);
        vector<T> p(n);
        vector<T> vel(n);
        vector<T> cs(n);

        for (I i = 0; i < n; i++)
        {
            r[i]   = vParticle[i].r;
            rho[i] = vParticle[i].rho;
            u[i]   = vParticle[i].u;
            p[i]   = vParticle[i].p;
            vel[i] = vParticle[i].vel;
            cs[i]  = vParticle[i].cs;
        }

        writeData1D(n, r, rho, u, p, vel, cs, outfile);
    }
};
