#pragma once

#include "sphexa.hpp"

#include "cstone/bsearch.hpp"
#include "cstone/octree.hpp"

namespace gravity
{

template <class I>
using InternalNode = std::pair<I, I>;

template <class T>
struct GravityData
{
    T mTot = 0.0;
    T xce, yce, zce;
    T xcm = 0.0, ycm = 0.0, zcm = 0.0;

    T qxx = 0.0, qxy = 0.0, qxz = 0.0;
    T qyy = 0.0, qyz = 0.0;
    T qzz = 0.0;

    T qxxa = 0.0, qxya = 0.0, qxza = 0.0;
    T qyya = 0.0, qyza = 0.0;
    T qzza = 0.0;

    T trq = 0.0;
    int pcount = 0;

    // std::vector<int> particleIdxList;
    // std::vector<int> globalParticleIdxList;

    T dx;                // side of a cell;
    int particleIdx = 0; // filled only if node is a leaf

    void print()
    {
        printf("mTot = %f, trq = %f\n", mTot, trq);
    }
};

template <class T>
using GravityTree = std::vector<GravityData<T>>;

template <class I, class T>
using TreeData = std::map<InternalNode<I>, GravityData<T>>;

template <class T>
void gatherGravValues(GravityData<T> *gv, bool global, int assignee)
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
template <class I, class T>
CUDA_HOST_DEVICE_FUN GravityData<T> particleGravity(const T *x, const T *y, const T *z, const T *m, int nParticles, T xmin, T xmax, T ymin,
                                                    T ymax, T zmin, T zmax, bool withGravitySync = false)
{
    GravityData<T> gv;
    gv.xce = (xmax - xmin) / 2.0;
    gv.yce = (ymax - ymin) / 2.0;
    gv.zce = (zmax - zmin) / 2.0;
    for (size_t i = 0; i < nParticles; ++i)
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
    gv.pcount = nParticles;
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
template <class I, class T>
GravityData<T> computeNodeGravity(const I firstCode, const I secondCode, const T *x, const T *y, const T *z, const T *m, const I *codes,
                                  const size_t n, const cstone::Box<T> &box, bool withGravitySync = false)
{
    // TODO: Search only from codes+lastParticle to the end since we know particles can not be in multiple nodes
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

    return particleGravity<I, T>(x + startIndex, y + startIndex, z + startIndex, m + startIndex, nParticles, xmin, xmax, ymin, ymax, zmin,
                                 zmax, withGravitySync);
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
template <class I, class T>
void calculateLeafGravityData(const std::vector<I> &tree,
                              // const std::vector<InernalNode<I>>& internalNodes,
                              const std::vector<T> &x, const std::vector<T> &y, const std::vector<T> &z, const std::vector<T> &m,
                              const std::vector<I> &codes, const cstone::Box<T> &box, TreeData<I, T> &gravityTreeData,
                              bool withGravitySync = false)
{
    for (auto it = tree.begin(); it + 1 != tree.end(); ++it)
    {
        I firstCode = *it;
        I secondCode = *(it+1);
        InternalNode<I> node = std::make_pair(firstCode, secondCode);
        gravityTreeData[node] = computeNodeGravity<I, T>(*it, *(it + 1), x.data(), y.data(), z.data(), m.data(), codes.data(),
                                                              codes.size(), box, withGravitySync);
    }
}

/**
 * @brief Computes gravity contribution for a given tree node represented by the start and end morton codes
 *        with respect to a given total mass and center of mass of the parent node
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
template <class I, class T>
CUDA_HOST_DEVICE_FUN GravityData<T> particleGravityContribution(const T *x, const T *y, const T *z, const T *m, int nParticles, T xce,
                                                                T yce, T zce, T xcm, T ycm, T zcm, T mTot)
{
    GravityData<T> gv;
    for (size_t i = 0; i < nParticles; ++i)
    {
        T xx = x[i];
        T yy = y[i];
        T zz = z[i];

        T m_i = m[i];

        T rx = xx - xce;
        T ry = yy - yce;
        T rz = zz - zce;

        gv.qxxa += rx * rx * m_i;
        gv.qxya += rx * ry * m_i;
        gv.qxza += rx * rz * m_i;
        gv.qyya += ry * ry * m_i;
        gv.qyza += ry * rz * m_i;
        gv.qzza += rz * rz * m_i;
    }
    gv.pcount = nParticles;
    return gv;
}

template <class I, class T>
GravityData<T> aggregateNodeGravity(const std::vector<I> &tree,
                                    // const std::vector<BinaryNode<I>>& internalNodes,
                                    TreeData<I, T> &treeData, const std::vector<InternalNode<I>> &gravityNodes,
                                    const std::vector<T> &x, const std::vector<T> &y, const std::vector<T> &z, const std::vector<T> &m,
                                    const std::vector<I> &codes, const cstone::Box<T> &box)
{
    // std::vector<cstone::BinaryNode<I>> internalNodes = cstone::createInternalTree(tree);
    GravityData<T> gv;
    for (auto it = gravityNodes.begin(); it != gravityNodes.end(); ++it)
    {
        InternalNode<I> node = *it;
        GravityData<T> current = treeData[node];
        gv.mTot += current.mTot;
        gv.xcm += current.xcm * current.mTot;
        gv.ycm += current.ycm * current.mTot;
        gv.zcm += current.zcm * current.mTot;
    }
    gv.xcm /= gv.mTot;
    gv.ycm /= gv.mTot;
    gv.zcm /= gv.mTot;

    T aggXmin = decodeXCoordinate(gravityNodes.front().first, box);
    T aggXmax = decodeXCoordinate(gravityNodes.back().second, box);
    gv.xce = (aggXmax - aggXmin) / 2;

    T aggYmin = decodeYCoordinate(gravityNodes.front().first, box);
    T aggYmax = decodeYCoordinate(gravityNodes.back().second, box);
    gv.yce = (aggYmax - aggYmin) / 2;

    T aggZmin = decodeZCoordinate(gravityNodes.front().first, box);
    T aggZmax = decodeZCoordinate(gravityNodes.back().second, box);
    gv.zce = (aggZmax - aggZmin) / 2;

    T rx = gv.xce - gv.xcm;
    T ry = gv.yce - gv.ycm;
    T rz = gv.zce - gv.zcm;

    T rxxm = rx * rx * gv.mTot;
    T rxym = rx * ry * gv.mTot;
    T rxzm = rx * rz * gv.mTot;
    T ryym = ry * ry * gv.mTot;
    T ryzm = ry * rz * gv.mTot;
    T rzzm = rz * rz * gv.mTot;

    size_t n = codes.size();

    for (auto it = gravityNodes.begin(); it != gravityNodes.end(); ++it)
    {
        I firstCode = it->first;
        I secondCode = it->second;
        int startIndex = stl::lower_bound(codes.data(), codes.data() + n, firstCode) - codes.data();
        int endIndex = stl::upper_bound(codes.data(), codes.data() + n, secondCode) - codes.data();
        int nParticles = endIndex - startIndex;
        // NOTE: using morton codes to compute geometrical center. It might not be accurate.
        T xmin = decodeXCoordinate(firstCode, box);
        T xmax = decodeXCoordinate(secondCode, box);
        T ymin = decodeYCoordinate(firstCode, box);
        T ymax = decodeYCoordinate(secondCode, box);
        T zmin = decodeZCoordinate(firstCode, box);
        T zmax = decodeZCoordinate(secondCode, box);

        GravityData<T> partialGravity = particleGravityContribution<I, T>(
            x.data() + startIndex, y.data() + startIndex, z.data() + startIndex, m.data() + startIndex, nParticles, (xmax - xmin) / 2,
            (ymax - ymin) / 2, (zmax - zmin) / 2, gv.xcm, gv.ycm, gv.ycm, gv.mTot);

        T dxxa = partialGravity.qxxa - rxxm;
        T dyya = partialGravity.qyya - ryym;
        T dzza = partialGravity.qzza - rzzm;

        gv.qxx += dxxa;
        gv.qxy += partialGravity.qxya - rxym;
        gv.qxz += partialGravity.qxza - rxzm;
        gv.qyy += dyya;
        gv.qyz += partialGravity.qyza - ryzm;
        gv.qzz += dzza;

        gv.trq += dxxa + dyya + dzza;
        gv.pcount += partialGravity.pcount;
    }
    return gv;
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
template <class I, class T>
void buildGlobalGravityTree(const std::vector<I> &tree,
                            // const std::vector<InernalNode<I>>& internalNodes,
                            const std::vector<T> &x, const std::vector<T> &y, const std::vector<T> &z, const std::vector<T> &m,
                            const std::vector<I> &codes, const cstone::Box<T> &box, GravityTree<T> &gravityTree,
                            bool withGravitySync = false)
{
    TreeData<I, T> treeData;
    calculateLeafGravityData(tree, x, y, z, m, codes, box, treeData);
    std::vector<InternalNode<I>> gravityNodes;
    gravityNodes.emplace_back(0700000000, 0710000000);
    gravityNodes.emplace_back(0710000000, 0720000000);
    gravityNodes.emplace_back(0720000000, 0730000000);
    gravityNodes.emplace_back(0730000000, 0740000000);
    gravityNodes.emplace_back(0740000000, 0750000000);
    gravityNodes.emplace_back(0750000000, 0760000000);
    gravityNodes.emplace_back(0760000000, 0770000000);
    gravityNodes.emplace_back(0770000000, 01000000000);
    GravityData<T> gv = aggregateNodeGravity<I, T>(tree, treeData, gravityNodes, x, y, z, m, codes, box);
    gv.print();
    gv = computeNodeGravity<I, T>(0700000000, 01000000000, x.data(), y.data(), z.data(), m.data(), codes.data(), codes.size(), box, false);
    gv.print();

}

/**
 * @brief
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
 */
template <class I, class T>
void showParticles(const std::vector<I> &tree, const std::vector<T> &x, const std::vector<T> &y, const std::vector<T> &z,
                   const std::vector<T> &m, const std::vector<I> &codes, const cstone::Box<T> &box)
{
    // size_t n = cstone::nNodes(tree);
    size_t n = codes.size();
    size_t totalParticles = 0;
    size_t i = 0;
    for (auto it = tree.cbegin(); it + 1 != tree.cend(); ++it)
    {
        I firstCode = *it;
        I secondCode = *(it + 1);
        int startIndex = stl::lower_bound(codes.data(), codes.data() + n, firstCode) - codes.data();
        int endIndex = stl::upper_bound(codes.data(), codes.data() + n, secondCode) - codes.data();
        int nParticles = endIndex - startIndex;
        totalParticles += nParticles;

        T xmin = decodeXCoordinate(firstCode, box);
        T xmax = decodeXCoordinate(secondCode, box);
        T ymin = decodeYCoordinate(firstCode, box);
        T ymax = decodeYCoordinate(secondCode, box);
        T zmin = decodeZCoordinate(firstCode, box);
        T zmax = decodeZCoordinate(secondCode, box);

        int level = cstone::treeLevel(secondCode - firstCode);

        printf("%o, %o, %d, %d, %f %f %f %f %f %f\n", firstCode, secondCode, level, nParticles, xmin, xmax, ymin, ymax, zmin, zmax);

        for (int i = 0; i < nParticles; ++i)
        {
            // x.data()[startIndex + i];
        }
    }

    printf("found %ld particles\n", totalParticles);
}

} // namespace gravity
