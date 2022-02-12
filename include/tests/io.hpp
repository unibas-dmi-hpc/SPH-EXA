#pragma once

#include <string>
#include <vector>
#include <sstream>
#include <iostream>

using namespace std;

class FileData
{

private:
    FileData(); // Singleton

public:
    static void readData3D(const string&   inputFile, //
                           const double    nParts,    //
                           vector<double>& x,         //
                           vector<double>& y,         //
                           vector<double>& z,         //
                           vector<double>& vx,        //
                           vector<double>& vy,        //
                           vector<double>& vz,        //
                           vector<double>& h,         //
                           vector<double>& rho,       //
                           vector<double>& u,         //
                           vector<double>& p,         //
                           vector<double>& cs,        //
                           vector<double>& Px,        //
                           vector<double>& Py,        //
                           vector<double>& Pz)        //
    {
        try
        {
            size_t i = 0;

            ifstream in(inputFile);
            string   line;

            while (getline(in, line))
            {
                istringstream iss(line);
                if (i < nParts)
                {
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

            if (nParts != i)
            {
                cout << "ERROR: number of particles doesn't match with the expect (nParts=" << nParts
                     << ", ParticleDataLines=" << i << ")." << endl;
                exit(EXIT_FAILURE);
            }

            in.close();
        }
        catch (exception& ex)
        {
            cout << "ERROR: %s. Terminating\n" << ex.what() << endl;
            exit(EXIT_FAILURE);
        }
    }
};
