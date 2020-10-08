#pragma once

#include <array>

namespace sphexa
{


/*! \brief stores the coordinated bounds
 *
 * Needs a slightly different behavior in the PBC case than the existing BBox
 * to manage morton code based octrees.
 *
 * \tparam T floating point type
 */
template<class T>
class Box
{
public:

    Box(T xyzMin, T xyzMax, bool hasPbc = false) :
        limits{xyzMin, xyzMax, xyzMin, xyzMax, xyzMin, xyzMax}, pbc{hasPbc, hasPbc, hasPbc}
    {}

    Box(T xmin, T xmax, T ymin, T ymax, T zmin, T zmax,
        bool pbcX = false, bool pbcY = false, bool pbcZ = false)
    : limits{xmin, xmax, ymin, ymax, zmin, zmax}, pbc{pbcX, pbcY, pbcZ}
    {}

    T xmin() const { return limits[0]; }
    T xmax() const { return limits[1]; }
    T ymin() const { return limits[2]; }
    T ymax() const { return limits[3]; }
    T zmin() const { return limits[4]; }
    T zmax() const { return limits[5]; }

    bool pbcX() const { return pbc[0]; }
    bool pbcY() const { return pbc[1]; }
    bool pbcZ() const { return pbc[2]; }

private:
    std::array<T, 6> limits;
    std::array<bool, 3> pbc;
};

} // namespace sphexa
