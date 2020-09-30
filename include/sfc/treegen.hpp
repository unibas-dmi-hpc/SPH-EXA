#pragma once

#include "Octree.hpp"

#include "sfc/mortoncode.hpp"

namespace sphexa
{

template<class T>
Octree<T> generateOctree(const std::vector<unsigned>& mortonCodes,
                         T xmin, T xmax,
                         T ymin, T ymax,
                         T zmin, T zmax)
{
    return Octree<T>(xmin, xmax, ymin, ymax, zmin, zmax, 0, 1);
}

} // namespace sphexa
