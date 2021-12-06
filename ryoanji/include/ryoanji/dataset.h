#pragma once
#include <vector>
#include <cmath>
#include <cstdlib>

#include "types.h"

namespace ryoanji
{

static void makeCubeBodies(fvec4* bodies, size_t n, double extent = 3)
{
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

    bodies[n - 1][0] = extent;
    bodies[n - 1][1] = extent;
    bodies[n - 1][2] = extent;
}

//! generate a grid with npOnEdge^3 bodies
static void makeGridBodies(fvec4* bodies, int npOnEdge, double spacing)
{
    for (size_t i = 0; i < npOnEdge; i++)
        for (size_t j = 0; j < npOnEdge; j++)
            for (size_t k = 0; k < npOnEdge; k++)
            {
                size_t linIdx     = i * npOnEdge * npOnEdge + j * npOnEdge + k;
                bodies[linIdx][0] = i * spacing;
                bodies[linIdx][1] = j * spacing;
                bodies[linIdx][2] = k * spacing;
                bodies[linIdx][3] = 1.0 / (npOnEdge * npOnEdge * npOnEdge);
            }
}

} // namespace ryoanji