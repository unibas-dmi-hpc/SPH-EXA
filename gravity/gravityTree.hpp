#pragma once

#include "sphexa.hpp"

#include "cstone/bsearch.hpp"
#include "cstone/octree.hpp"

namespace gravity
{

template<class T>
struct GravityData {
    T mTot = 0.0;
    T xce, yce, zce;
    T xcm = 0.0, ycm = 0.0, zcm = 0.0;

    T qxx, qxy, qxz;
    T qyy, qyz;
    T qzz;

    T qxxa = 0.0, qxya = 0.0, qxza = 0.0;
    T qyya = 0.0, qyza = 0.0;
    T qzza = 0.0;

    T trq;
    int pcount = 0;

    //std::vector<int> particleIdxList;
    //std::vector<int> globalParticleIdxList;

    T dx;                // side of a cell;
    int particleIdx = 0; // filled only if node is a leaf
};

template<class T>
struct GravityTree {
    std::vector<GravityData<T>> nodeData;
};

template<class T>
void gatherGravValues(GravityData<T>* gv, bool global, int assignee)
{
#ifdef USE_MPI
    if (global && assignee == -1)
    {
        MPI_Allreduce(MPI_IN_PLACE, &(*gv).mTot, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        MPI_Allreduce(MPI_IN_PLACE, &(*gv).xcm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, &(*gv).ycm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, &(*gv).zcm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        MPI_Allreduce(MPI_IN_PLACE, &(*gv).qxxa, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, &(*gv).qxya, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, &(*gv).qxza, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, &(*gv).qyya, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, &(*gv).qyza, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, &(*gv).qzza, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    }
#endif
}

/**
 * @brief Computes gravity forces for a given tree node represented by the start and end morton codes
 * 
 * @tparam I T float or double.
 * @tparam T I 32- or 64-bit unsigned.
 * \param[in] firstCode   lower Morton code
 * \param[in] secondCode  upper Morton code
 * @param list 
 * @param x List of x coordinates.
 * @param y List of y coordinates.
 * @param z List of z coordinates.
 * @param m Vector of masses.
 * @param codes 
 * @param box 
 * @param withGravitySync 
 */
template<class I, class T>
CUDA_HOST_DEVICE_FUN
GravityData<T> particleGravity(const T* x,
                                const T* y,
                                const T* z,
                                const T* m,
                                int nParticles,
                                T xmin,
                                T xmax,
                                T ymin,
                                T ymax,
                                T zmin,
                                T zmax,
                                bool withGravitySync = false)
{
    GravityData<T> gv;
    gv.xce = (xmax - xmin) / 2.0;
    gv.yce = (ymax - ymin) / 2.0;
    gv.zce = (zmax - zmin) / 2.0;
    for (size_t i = 0 ; i < nParticles ; ++ i)
    {
        T xx = x[i];
        T yy = y[i];
        T zz = z[i];

        T m_i = m[i];

        gv.xcm += xx * m_i;
        gv.ycm += yy * m_i;
        gv.zcm += zz * m_i;

        gv.mTot += m_i;

        T rx = xx - gv.xce;
        T ry = yy - gv.yce;
        T rz = zz - gv.zce;

        gv.qxxa += rx * rx * m_i;
        gv.qxya += rx * ry * m_i;
        gv.qxza += rx * rz * m_i;
        gv.qyya += ry * ry * m_i;
        gv.qyza += ry * rz * m_i;
        gv.qzza += rz * rz * m_i;

        gv.particleIdx = i;
    }

    // if all nodes are global, why do we need an assignee ?? should global nodes also be assigned !?
    bool global = false;
    int assignee = -1;
    if (withGravitySync) gatherGravValues(&gv, global, assignee);

    if (nParticles > 1 || global)
    {
        gv.xcm /= gv.mTot;
        gv.ycm /= gv.mTot;
        gv.zcm /= gv.mTot;

        T rx = gv.xce - gv.xcm;
        T ry = gv.yce - gv.ycm;
        T rz = gv.zce - gv.zcm;
        gv.qxx = gv.qxxa - rx * rx * gv.mTot;
        gv.qxy = gv.qxya - rx * ry * gv.mTot;
        gv.qxz = gv.qxza - rx * rz * gv.mTot;
        gv.qyy = gv.qyya - ry * ry * gv.mTot;
        gv.qyz = gv.qyza - ry * rz * gv.mTot;
        gv.qzz = gv.qzza - rz * rz * gv.mTot;

        gv.trq = gv.qxx + gv.qyy + gv.qzz;
    }
    else if (nParticles == 1)
    {
        size_t idx = gv.particleIdx;

        gv.xcm = x[idx];
        gv.ycm = y[idx];
        gv.zcm = z[idx];

        gv.xce = x[idx];
        gv.yce = y[idx];
        gv.zce = z[idx];

        gv.qxx = 0;
        gv.qxy = 0;
        gv.qxz = 0;
        gv.qyy = 0;
        gv.qyz = 0;
        gv.qzz = 0;

        gv.trq = 0;
        gv.dx = 0; // used to indicate that node is a leaf
    }
    return gv;
}

/**
 * @brief Compute equivalent gravity component for a tree node represented by the first and last morton code
 * 
 * @tparam I 
 * @tparam T 
 * @param firstCode 
 * @param secondCode 
 * @param x 
 * @param y 
 * @param z 
 * @param m 
 * @param codes 
 * @param box 
 * @param withGravitySync 
 * @return GravityData<T> 
 */
template<class I, class T>
GravityData<T> computeNodeGravity(const I firstCode,
                                   const I secondCode,
                                   const T* x,
                                   const T* y,
                                   const T* z,
                                   const T* m,
                                   const I* codes,
                                   const size_t n,
                                   const cstone::Box<T>& box,
                                   bool withGravitySync = false)
{
    int startIndex = stl::lower_bound(codes, codes + n, firstCode) - codes;
    int endIndex = stl::upper_bound(codes, codes + n, secondCode) - codes;
    int nParticles = endIndex - startIndex;
    // NOTE: using morton codes to compute geometrical center. It might not be accurate.
    T xmin = decodeXCoordinate(firstCode, box);
    T xmax = decodeXCoordinate(secondCode, box);
    T ymin = decodeYCoordinate(firstCode, box);
    T ymax = decodeYCoordinate(secondCode, box);
    T zmin = decodeZCoordinate(firstCode, box);
    T zmax = decodeZCoordinate(secondCode, box);

    return particleGravity<I, T>(x + startIndex, y + startIndex, z + startIndex, m + startIndex, nParticles,
                               xmin, xmax, ymin, ymax, zmin, zmax, withGravitySync);
}

/**
 * @brief Generates the global gravity values for all the nodes in a given tree
 * 
 * @tparam I 
 * @tparam T 
 * @param tree 
 * @param x 
 * @param y 
 * @param z 
 * @param m 
 * @param codes 
 * @param box 
 * @param withGravitySync 
 * @return GravityTree<T> 
 */
template<class I, class T>
void buildGlobalGravityTree(const std::vector<I>& tree,
                            const std::vector<T>& x,
                            const std::vector<T>& y,
                            const std::vector<T>& z,
                            const std::vector<T>& m,
                            const std::vector<I>& codes,
                            const cstone::Box<T>& box,
                            GravityTree<T>& gravityTree,
                            bool withGravitySync = false)
{
    /*
     * TODO: When computing node gravity, re-use children nodes to compute higher node gravity data.
     * This might save precious time if data can be aggregated to higher levels (at least the sum of masses can be)
     */
    size_t n = cstone::nNodes(tree);
    gravityTree.nodeData.reserve(n);
    gravityTree.nodeData.resize(n);
    size_t i = 0;
    for(auto it = tree.cbegin(); it+1 != tree.cend(); ++it)
    {
        gravityTree.nodeData[i++] = std::move(computeNodeGravity<I, T>(*it, *(it+1), x.data(), y.data(), z.data(), m.data(), codes.data(), codes.size(), box, withGravitySync));
    }
}

} //namespace sphexa
