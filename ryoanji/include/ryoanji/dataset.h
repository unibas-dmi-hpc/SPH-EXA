#pragma once
#include <vector>
#include <cmath>
#include <cstdlib>
#include <fstream>

#include "types.h"

class Dataset
{
private:
    unsigned long n;

    void printDots(int i)
    {
        ldiv_t tmp_i = ldiv(i, n / 33);
        if (tmp_i.rem == 0)
        {
            printf(".");
            fflush(stdout);
        }
    }

    bool read(const char* filename)
    {
        std::ifstream file(filename);
        if (!file.fail())
        {
            unsigned long ntmp;
            file.read((char*)&ntmp, sizeof(unsigned long));
            if (n == ntmp)
            {
                file.read((char*)&pos[0], n * sizeof(double4));
                file.close();
                return true;
            }
        }
        file.close();
        return false;
    }

    void write(const char* filename)
    {
        std::ofstream file(filename);
        file.write((char*)&n, sizeof(unsigned long));
        file.write((char*)&pos[0], n * sizeof(double4));
        file.close();
    }

public:
    std::vector<kvec4> pos;
    Dataset(unsigned long _n)
        : n(_n)
        , pos(_n)
    {
#if MASS
        if (read("plummer.dat")) return;
        unsigned long i   = 0;
        const float scale = 3.0 * M_PI / 16.0;
        while (i < n)
        {
            float R = 1.0 / sqrt((pow(drand48(), -2.0 / 3.0) - 1.0));
            if (R < 100.0)
            {
                float Z     = (1.0 - 2.0 * drand48()) * R;
                float theta = 2.0 * M_PI * drand48();
                float X     = sqrt(R * R - Z * Z) * cos(theta);
                float Y     = sqrt(R * R - Z * Z) * sin(theta);
                X *= scale;
                Y *= scale;
                Z *= scale;
                pos[i][0] = X;
                pos[i][1] = Y;
                pos[i][2] = Z;
                pos[i][3] = drand48() / n;
                printDots(i);
                i++;
            }
        }
        printf("\n");
        kvec4 com(0.0);
        for (i = 0; i < n; i++)
        {
            com[0] += abs(pos[i][3]) * pos[i][0];
            com[1] += abs(pos[i][3]) * pos[i][1];
            com[2] += abs(pos[i][3]) * pos[i][2];
            com[3] += abs(pos[i][3]);
        }
        com[0] /= com[3];
        com[1] /= com[3];
        com[2] /= com[3];
        for (i = 0; i < n; i++)
        {
            pos[i][0] -= com[0];
            pos[i][1] -= com[1];
            pos[i][2] -= com[2];
        }
        write("plummer.dat");
#else
        if (read("cube.dat")) return;
        float average = 0;
        for (int i = 0; i < n; i++)
        {
            pos[i][0] = drand48() * 2 * M_PI - M_PI;
            pos[i][1] = drand48() * 2 * M_PI - M_PI;
            pos[i][2] = drand48() * 2 * M_PI - M_PI;
            pos[i][3] = drand48() / n;
            average += pos[i][3];
            printDots(i);
        }
        printf("\n");
        average /= n;
        for (int i = 0; i < n; i++)
        {
            pos[i][3] -= average;
        }
        write("cube.dat");
#endif
    }
};

static std::vector<fvec4> makeCubeBodies(size_t n, double extent = 3)
{
    std::vector<fvec4> bodies(n);

    for (size_t i = 0; i < n; i++)
    {
        bodies[i][0] = drand48() * 2 * extent - extent;
        bodies[i][1] = drand48() * 2 * extent - extent;
        bodies[i][2] = drand48() * 2 * extent - extent;
        bodies[i][3] = drand48() / n;
    }

    // set non-random corners
    bodies[0][0] = -extent;
    bodies[0][1] = -extent;
    bodies[0][2] = -extent;

    bodies[n-1][0] = extent;
    bodies[n-1][1] = extent;
    bodies[n-1][2] = extent;

    return bodies;
}
