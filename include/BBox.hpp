#pragma once

#include <cmath>

#include "cudaFunctionAnnotation.hpp"

namespace sphexa
{

template <typename T>
class BBox
{
public:
    BBox(T xmin = -1, T xmax = 1, T ymin = -1, T ymax = 1, T zmin = -1, T zmax = 1, bool PBCx = false, bool PBCy = false, bool PBCz = false)
        : xmin(xmin)
        , xmax(xmax)
        , ymin(ymin)
        , ymax(ymax)
        , zmin(zmin)
        , zmax(zmax)
        , PBCx(PBCx)
        , PBCy(PBCy)
        , PBCz(PBCz)
    {
    }

    T xmin, xmax, ymin, ymax, zmin, zmax;
    bool PBCx, PBCy, PBCz;
};

template <typename T>
CUDA_DEVICE_HOST_FUN inline void applyPBC(const BBox<T> &bbox, const T r, T &xx, T &yy, T &zz)
{
    if (bbox.PBCx && xx > r)
        xx -= (bbox.xmax - bbox.xmin);
    else if (bbox.PBCx && xx < -r)
        xx += (bbox.xmax - bbox.xmin);

    if (bbox.PBCy && yy > r)
        yy -= (bbox.ymax - bbox.ymin);
    else if (bbox.PBCy && yy < -r)
        yy += (bbox.ymax - bbox.ymin);

    if (bbox.PBCz && zz > r)
        zz -= (bbox.zmax - bbox.zmin);
    else if (bbox.PBCz && zz < -r)
        zz += (bbox.zmax - bbox.zmin);

    // xx += bbox.PBCx * ((xx < -r) - (xx > r)) * (bbox.xmax-bbox.xmin);
    // yy += bbox.PBCy * ((yy < -r) - (yy > r)) * (bbox.ymax-bbox.ymin);
    // zz += bbox.PBCz * ((zz < -r) - (zz > r)) * (bbox.zmax-bbox.zmin);
}

template <typename T>
CUDA_DEVICE_HOST_FUN inline T distancePBC(const BBox<T> &bbox, const T hi, const T x1, const T y1, const T z1, const T x2, const T y2, const T z2)
{
    T xx = x1 - x2;
    T yy = y1 - y2;
    T zz = z1 - z2;

    applyPBC<T>(bbox, 2.0 * hi, xx, yy, zz);

    return std::sqrt(xx * xx + yy * yy + zz * zz);
}

} // namespace sphexa
