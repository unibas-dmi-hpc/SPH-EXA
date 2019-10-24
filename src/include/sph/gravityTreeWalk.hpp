#pragma once

#include <cmath>

#include "../Octree.hpp"

namespace sphexa
{
namespace sph
{

template <typename T>
void treeWalkRef(const Octree<T> &node, const int i, const T *x, const T *y, const T *z, const T *h, T *fx, T *fy, T *fz, T *ugrav)
{
    const auto gnode = dynamic_cast<const GravityOctree<T> &>(node);

    const T d1 = std::abs(x[i] - gnode.xce);
    const T d2 = std::abs(y[i] - gnode.yce);
    const T d3 = std::abs(z[i] - gnode.zce);
    //    const T dc = 4.0 * h[i] + gnode.dx / 2.0;
    const T dc = 3.0 * h[i] + gnode.dx / 2.0;

    if (d1 <= dc && d2 <= dc && d3 < dc) // intersecting
    {
        if (gnode.dx == 0) // node is a leaf
        {
            const auto j = gnode.particleIdx;

            if (i != j) // skip calculating gravity contribution of myself
            // if (!(x[i] == gnode.xce && y[i] == gnode.yce && z[i] == gnode.zce))
            {
                const T dd2 = d1 * d1 + d2 * d2 + d3 * d3;
                const T dd5 = std::sqrt(dd2);

                T g0;

                if (dd2 > 2 * h[i] && dd2 > 2 * h[j]) { g0 = 1.0 / dd5 / dd2; }
                else
                {
                    const T hij = h[i] + h[j];
                    const T vgr = dd5 / hij;
                    const T mefec = std::min(1.0, vgr * vgr * vgr);
                    g0 = mefec / dd5 / dd2;
                }

                fx[i] -= g0 * d1;
                fy[i] -= g0 * d2;
                fz[i] -= g0 * d3;
                ugrav[i] += g0 * dc;
            }
        }
        else
        {
            for (const auto &child : gnode.cells) // go deeper to the childs
                treeWalkRef(*child, i, x, y, z, h, fx, fy, fz, ugrav);
        }
    }

    else // not intersecting
    {
        const T r1 = x[i] - gnode.xcm;
        const T r2 = y[i] - gnode.ycm;
        const T r3 = z[i] - gnode.zcm;
        const T dd2 = r1 * r1 + r2 * r2 + r3 * r3;

        if (gnode.dx * gnode.dx <= gnode.tol * dd2)
        {
            const T dd5 = sqrt(dd2);
            const T d32 = 1 / dd5 / dd2;

            T g0;

            if (gnode.dx == 0) // node is a leaf
            {
                const int j = gnode.particleIdx;
                const T v1 = dd5 / h[i];
                const T v2 = dd5 / h[j];

                if (v1 > 2 && v2 > 2) { g0 = gnode.mTot * d32; }
                else
                {
                    const T hij = h[i] + h[j];
                    const T vgr = dd5 / hij;
                    const T mefec = std::min(1.0, vgr * vgr * vgr);
                    g0 = mefec * d32 * gnode.mTot;
                }

                fx[i] -= g0 * r1;
                fy[i] -= g0 * r2;
                fz[i] -= g0 * r3;
                ugrav[i] += g0 * dd2;
            }
            else // node is not leaf
            {
                g0 = gnode.mTot * d32; // Base Value
                fx[i] -= g0 * r1;
                fy[i] -= g0 * r2;
                fz[i] -= g0 * r3;
                ugrav[i] += g0 * d32; // eof Base value

                const T r5 = dd2 * dd2 * dd5;
                const T r7 = r5 * dd2;

                const T qr1 = r1 * gnode.qxx + r2 * gnode.qxy + r3 * gnode.qxz;
                const T qr2 = r1 * gnode.qxy + r2 * gnode.qyy + r3 * gnode.qyz;
                const T qr3 = r1 * gnode.qxz + r2 * gnode.qyz + r3 * gnode.qzz;

                const T rqr = r1 * qr1 + r2 * qr2 + r3 * qr3;

                const T c1 = (-7.5 / r7) * rqr;
                const T c2 = 3.0 / r5;
                const T c3 = 0.5 * gnode.trq;

                fx[i] += c1 * r1 + c2 * (qr1 + c3 * r1);
                fy[i] += c1 * r2 + c2 * (qr2 + c3 * r2);
                fz[i] += c1 * r3 + c2 * (qr3 + c3 * r3);
                ugrav[i] -= (1.5 / r5) * rqr + c3 * d32;
            }
            // go to next node
        }
        else // go deeper
        {
            for (const auto &child : gnode.cells) // go deeper to the childs
                treeWalkRef(*child, i, x, y, z, h, fx, fy, fz, ugrav);
        }
    }
}

template <typename T, typename Dataset>
void gravityTreeWalk(const std::vector<int> &clist, const GravityOctree<T> &tree, Dataset &d)
{
    const size_t n = clist.size();

    const T *x = d.x.data();
    const T *y = d.y.data();
    const T *z = d.z.data();
    const T *h = d.h.data();

    T *fx = d.fx.data();
    T *fy = d.fy.data();
    T *fz = d.fz.data();
    T *ugrav = d.ugrav.data();

    for (size_t pi = 0; pi < n; ++pi)
    {
        // const int i = clist[pi];
        // if (i==627) printf("%d, %d: [%f, %f, %f] ", pi, clist[i], x[i], y[i], z[i]);
    }

    for (size_t pi = 0; pi < n; ++pi)
    {
        const int i = clist[pi];

        treeWalkRef(tree, i, x, y, z, h, fx, fy, fz, ugrav);
    }

    // printf("Gravity components\n");

    // for (size_t pi = 0; pi < n; ++pi)
    // {
    //     int i = clist[pi];
    //     printf("%d: [%f, %f] ", clist[i], d.fx[i], d.ugrav[i]);
    //     if (i % 10 == 0) printf("\n");
    // }
}

} // namespace sph
} // namespace sphexa
