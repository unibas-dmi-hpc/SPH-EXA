#pragma once

#include <string>
#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include <exception>

using namespace std;

template<typename T, typename I>
struct ParticleIO
{
    I idx; // initial index order
    T r;   // radius
    T vel; // radius

    T x;   // x position
    T y;   // y position
    T z;   // z position
    T vx;  // vx velocity
    T vy;  // vy velocity
    T vz;  // vz velocity
    T h;   // smoothing length
    T rho; // density
    T u;   // internal energy
    T p;   // pressure
    T cs;  // sound speed
    T Px;  // x momentum
    T Py;  // y momentum
    T Pz;  // z momentum

    // To sort the particles by radius
    struct cmp
    {
        bool operator()(const ParticleIO& p1, const ParticleIO& p2) { return p1.r < p2.r; }
    };
};

template<typename T, typename I>
class FileData
{

private:
    FileData(); // Singleton

public:
    static void readData3D(const string& inputFile, //
                           const I       nParts,    //
                           vector<T>&    x,         //
                           vector<T>&    y,         //
                           vector<T>&    z,         //
                           vector<T>&    vx,        //
                           vector<T>&    vy,        //
                           vector<T>&    vz,        //
                           vector<T>&    h,         //
                           vector<T>&    rho,       //
                           vector<T>&    u,         //
                           vector<T>&    p,         //
                           vector<T>&    cs,        //
                           vector<T>&    Px,        //
                           vector<T>&    Py,        //
                           vector<T>&    Pz)
    {
        try
        {
            I i = 0;

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
