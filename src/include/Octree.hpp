#pragma once

#include <cmath>
#include <memory>
#include <vector>
#include <algorithm>

#include "Task.hpp"

namespace sphexa
{

template <typename T>
class Octree
{
public:
    Octree(const T xmin, const T xmax, const T ymin, const T ymax, const T zmin, const T zmax, int comm_rank, int comm_size)
        : xmin(xmin)
        , xmax(xmax)
        , ymin(ymin)
        , ymax(ymax)
        , zmin(zmin)
        , zmax(zmax)
        , comm_rank(comm_rank)
        , comm_size(comm_size)
    {
    }

    Octree() {}

    ~Octree()
    {
        cells.clear();
    }


    int getAssignedRank() const { return assignee; }

    int getStorageOffset() const { return zCurveOffset; }

    int getGlobalParticleCount() const { return globalParticleCount; }

    int getGlobalNodeCount() const { return globalNodeCount; }

private:
    T xmin = INFINITY, xmax = -INFINITY, ymin = INFINITY, ymax = -INFINITY, zmin = INFINITY, zmax = -INFINITY;

    int comm_rank = -1;
    int comm_size = -1;
    int assignee = -1;

    int zCurveOffset = 0;               // offset into particle property arrays
    int localParticleCount = 0;

    int globalParticleCount = 0;
    int globalNodeCount = 0;            // number of global nodes below this one

    std::vector<std::shared_ptr<Octree>> cells;

    T localMaxH = 0.0;
    T globalMaxH = 0.0;

    bool halo = false;
    bool global = false;

    static const int nX = 2, nY = 2, nZ = 2;
    static const int ncells = 8;
    static const int bucketSize = 64, maxGlobalBucketSize = 512, minGlobalBucketSize = 256;


    static inline T normalize(T d, T min, T max) { return (d - min) / (max - min); }

    static inline T distancesq(const T x1, const T y1, const T z1, const T x2, const T y2, const T z2)
    {
        T xx = x1 - x2;
        T yy = y1 - y2;
        T zz = z1 - z2;

        return xx * xx + yy * yy + zz * zz;
    }

    inline void check_add_start(const int id, const T *x, const T *y, const T *z, const T xi, const T yi, const T zi, const T r,
                                const int ngmax, int *neighbors, int &neighborsCount) const
    {
        T r2 = r * r;

        // int maxCount = std::min(localParticleCount, ngmax - neighborsCount);

        for (int i = 0; i < localParticleCount; i++)
        {
            int ordi = zCurveOffset + i;

            T dist = distancesq(xi, yi, zi, x[ordi], y[ordi], z[ordi]);
            if (dist < r2 && ordi != id && neighborsCount < ngmax)
               neighbors[neighborsCount++] = ordi;
        }
    }

    inline void distributeParticles(const std::vector<int> &list, const std::vector<T> &ax, const std::vector<T> &ay,
                                    const std::vector<T> &az, std::vector<std::vector<int>> &cellList)
    {
        for (unsigned int i = 0; i < list.size(); i++)
        {
            T xx = ax[list[i]];
            T yy = ay[list[i]];
            T zz = az[list[i]];

            T hx = std::min(std::max((int)(normalize(xx, xmin, xmax) * nX), 0), nX - 1);
            T hy = std::min(std::max((int)(normalize(yy, ymin, ymax) * nY), 0), nY - 1);
            T hz = std::min(std::max((int)(normalize(zz, zmin, zmax) * nZ), 0), nZ - 1);

            unsigned int l = hz * nX * nY + hy * nX + hx;

            cellList[l].push_back(list[i]);
        }
    }

    inline bool overlap(Octree *a)
    {
        T radius = a->globalMaxH * 2.0;

        //Check if Box1's max is greater than Box2's min and Box1's min is less than Box2's max
        return(a->xmax+radius > xmin &&
               a->xmin-radius < xmax &&
               a->ymax+radius > ymin &&
               a->ymin-radius < ymax &&
               a->zmax+radius > zmin &&
               a->zmin-radius < zmax);
    }

    void makeSubCells()
    {
        cells.resize(ncells);

        for (int hz = 0; hz < nZ; hz++)
        {
            for (int hy = 0; hy < nY; hy++)
            {
                for (int hx = 0; hx < nX; hx++)
                {
                    T ax = xmin + hx * (xmax - xmin) / nX;
                    T bx = xmin + (hx + 1) * (xmax - xmin) / nX;
                    T ay = ymin + hy * (ymax - ymin) / nY;
                    T by = ymin + (hy + 1) * (ymax - ymin) / nY;
                    T az = zmin + hz * (zmax - zmin) / nZ;
                    T bz = zmin + (hz + 1) * (zmax - zmin) / nZ;

                    unsigned int i = hz * nX * nY + hy * nX + hx;

                    if (cells[i] == nullptr) cells[i] = std::make_shared<Octree>(ax, bx, ay, by, az, bz, comm_rank, comm_size);
                }
            }
        }
    }

public:

    void findNeighborsRec(const int id, const T *x, const T *y, const T *z, const T xi, const T yi, const T zi, const T ri, const int ngmax,
                          int *neighbors, int &neighborsCount) const
    {
        if ((int)cells.size() == ncells)
        {
            int mix = std::max((int)(normalize(xi - ri, xmin, xmax) * nX), 0);
            int miy = std::max((int)(normalize(yi - ri, ymin, ymax) * nY), 0);
            int miz = std::max((int)(normalize(zi - ri, zmin, zmax) * nZ), 0);
            int max = std::min((int)(normalize(xi + ri, xmin, xmax) * nX), nX - 1);
            int may = std::min((int)(normalize(yi + ri, ymin, ymax) * nY), nY - 1);
            int maz = std::min((int)(normalize(zi + ri, zmin, zmax) * nZ), nZ - 1);

            for (int hz = miz; hz <= maz; hz++)
            {
                for (int hy = miy; hy <= may; hy++)
                {
                    for (int hx = mix; hx <= max; hx++)
                    {
                        unsigned int l = hz * nX * nY + hy * nX + hx;

                        cells[l]->findNeighborsRec(id, x, y, z, xi, yi, zi, ri, ngmax, neighbors, neighborsCount);
                    }
                }
            }
        }
        else
            check_add_start(id, x, y, z, xi, yi, zi, ri, ngmax, neighbors, neighborsCount);
    }

    void findNeighbors(const int id, const T *x, const T *y, const T *z, const T xi, const T yi, const T zi, const T ri, const int ngmax,
                       int *neighbors, int &neighborsCount, const bool PBCx = false, const bool PBCy = false, const bool PBCz = false) const
    {
        if ((PBCx && (xi - ri < xmin || xi + ri > xmax)) || (PBCy && (yi - ri < ymin || yi + ri > ymax)) ||
            (PBCz && (zi - ri < zmin || zi + ri > zmax)))
        {
            int mix = (int)floor(normalize(xi - ri, xmin, xmax) * nX);
            int miy = (int)floor(normalize(yi - ri, ymin, ymax) * nY);
            int miz = (int)floor(normalize(zi - ri, zmin, zmax) * nZ);
            int max = (int)floor(normalize(xi + ri, xmin, xmax) * nX);
            int may = (int)floor(normalize(yi + ri, ymin, ymax) * nY);
            int maz = (int)floor(normalize(zi + ri, zmin, zmax) * nZ);

            if (!PBCx) mix = std::max(mix, 0);
            if (!PBCy) miy = std::max(miy, 0);
            if (!PBCz) miz = std::max(miz, 0);
            if (!PBCx) max = std::min(max, nX - 1);
            if (!PBCy) may = std::min(may, nY - 1);
            if (!PBCz) maz = std::min(maz, nZ - 1);

            for (int hz = miz; hz <= maz; hz++)
            {
                for (int hy = miy; hy <= may; hy++)
                {
                    for (int hx = mix; hx <= max; hx++)
                    {
                        T displz = PBCz ? ((hz < 0) - (hz >= nZ)) * (zmax - zmin) : 0;
                        T disply = PBCy ? ((hy < 0) - (hy >= nY)) * (ymax - ymin) : 0;
                        T displx = PBCx ? ((hx < 0) - (hx >= nX)) * (xmax - xmin) : 0;

                        int hzz = PBCz ? (hz % nZ) + (hz < 0) * nZ : hz;
                        int hyy = PBCy ? (hy % nY) + (hy < 0) * nY : hy;
                        int hxx = PBCx ? (hx % nX) + (hx < 0) * nX : hx;

                        unsigned int l = hzz * nY * nX + hyy * nX + hxx;

                        cells[l]->findNeighborsRec(id, x, y, z, xi + displx, yi + disply, zi + displz, ri, ngmax, neighbors,
                                                   neighborsCount);
                    }
                }
            }
        }
        else
            findNeighborsRec(id, x, y, z, xi, yi, zi, ri, ngmax, neighbors, neighborsCount);
    }

    void approximateRec(const std::vector<int> &list, const std::vector<T> &x, const std::vector<T> &y, const std::vector<T> &z,
                        const std::vector<T> &h)
    {
        this->globalNodeCount = 1;

        global = true;

        T hmax = 0.0;
        for (unsigned int i = 0; i < list.size(); i++)
        {
            T hh = h[list[i]];
            if (hh > hmax) hmax = hh;
        }

        const T sizex = xmax - xmin;
        const T sizey = ymax - ymin;
        const T sizez = zmax - zmin;

        const T size = std::max(sizez, std::max(sizey, sizex));

        // Expand node if cell bigger than 2.0 * h
        if (size > 4.0 * hmax && list.size() > 0)
        {
            std::vector<std::vector<int>> cellList(ncells);
            distributeParticles(list, x, y, z, cellList);

            makeSubCells();
            for (int i = 0; i < ncells; i++)
            {
                cells[i]->approximateRec(cellList[i], x, y, z, h);
                this->globalNodeCount += cells[i]->globalNodeCount;
            }
        }
    }

    void approximate(const std::vector<T> &x, const std::vector<T> &y, const std::vector<T> &z, const std::vector<T> &h)
    {
        std::vector<int> list(x.size());
        for (unsigned int i = 0; i < x.size(); i++)
            list[i] = i;

        this->globalNodeCount = 1;

        global = true;

        std::vector<std::vector<int>> cellList(ncells);
        distributeParticles(list, x, y, z, cellList);

        // There is always one level below the root
        // This simplifies the function for finding neighbors with
        makeSubCells();
        for (int i = 0; i < ncells; i++)
        {
            cells[i]->approximateRec(cellList[i], x, y, z, h);
            this->globalNodeCount += cells[i]->globalNodeCount;
        }
    }

    int globalRebalance(T xmin, T xmax, T ymin, T ymax,  T zmin, T zmax)
    {
        this->xmin = xmin;
        this->xmax = xmax;
        this->ymin = ymin;
        this->ymax = ymax;
        this->zmin = zmin;
        this->zmax = zmax;

        this->assignee = -1;
        this->halo = false;

        this->zCurveOffset = 0;
        this->globalNodeCount = 0;
        this->localParticleCount = 0;

        this->localMaxH = 0.0;
        this->globalMaxH = 0.0;

        int nsplits = 0;

        if (global)
        {
            this->globalNodeCount = 1;

            // Closing non global branches
            if((int)cells.size() == ncells && cells[0]->global == false)
                cells.clear();

            if ((int)cells.size() == ncells)
            {
                if (globalParticleCount < minGlobalBucketSize)
                    cells.clear();
                else
                {
                    for (int hz = 0; hz < nZ; hz++)
                    {
                        for (int hy = 0; hy < nY; hy++)
                        {
                            for (int hx = 0; hx < nX; hx++)
                            {
                                T ax = xmin + hx * (xmax - xmin) / nX;
                                T bx = xmin + (hx + 1) * (xmax - xmin) / nX;
                                T ay = ymin + hy * (ymax - ymin) / nY;
                                T by = ymin + (hy + 1) * (ymax - ymin) / nY;
                                T az = zmin + hz * (zmax - zmin) / nZ;
                                T bz = zmin + (hz + 1) * (zmax - zmin) / nZ;

                                unsigned int i = hz * nX * nY + hy * nX + hx;

                                nsplits += cells[i]->globalRebalance(ax, bx, ay, by, az, bz);
                                this->globalNodeCount += cells[i]->globalNodeCount;
                            }
                        }
                    }
                }
            }
            else // if((int)cells.size() ==  0)
            {
                if (globalParticleCount > maxGlobalBucketSize)
                {
                    makeSubCells();
                    nsplits += ncells;
                    for (int i = 0; i < ncells; i++)
                    {
                        //cells[i]->approximateRec(cellList[i], x, y, z, h);
                        cells[i]->globalNodeCount = 1;
                        cells[i]->global = true;
                        this->globalNodeCount += cells[i]->globalNodeCount;
                    }
                }
                else
                    this->globalNodeCount = 1;
            }
        }

        this->globalParticleCount = 0;

        return nsplits;
    }

    void updateGlobalCountsRec(const std::vector<int> &list, const std::vector<T> &x, const std::vector<T> &y, const std::vector<T> &z, const std::vector<T> &h,
                              std::vector<int> &ordering, std::vector<int> &globalParticleCount, std::vector<T> &globalMaxH, int zoffset = 0, int zCurveNodeIdx = 0)
    {
        this->zCurveOffset = zoffset;
        this->localParticleCount = 0;
        this->globalMaxH = 0.0;

        int it = zCurveNodeIdx;

        zCurveNodeIdx++;

        std::vector<std::vector<int>> cellList(ncells);
        distributeParticles(list, x, y, z, cellList);

        if ((int)cells.size() == ncells)
        {
            for (int i = 0; i < ncells; i++)
            {
                cells[i]->updateGlobalCountsRec(cellList[i], x, y, z, h, ordering, globalParticleCount, globalMaxH, zoffset, zCurveNodeIdx);
                this->localParticleCount += cells[i]->localParticleCount;
                this->globalMaxH = std::max(this->globalMaxH, cells[i]->globalMaxH);
                zoffset += cells[i]->localParticleCount;
                zCurveNodeIdx += cells[i]->globalNodeCount;
            }
        }
        else
        {
            for (int i = 0; i < (int)list.size(); i++)
            {
                ordering[zoffset + i] = list[i];
                if(h[list[i]] > this->globalMaxH)
                    this->globalMaxH = h[list[i]];
            }
            this->localParticleCount = list.size();
        }

        globalMaxH[it] = this->globalMaxH;
        globalParticleCount[it] = this->localParticleCount;
    }

#ifdef USE_MPI

    // after this call, for each node:
    // zCurveOffset + localParticleCount locate particles belonging to this node in x,y,z,...
    // globalMaxH = global max H of any particle
    // globalParticleCount = global sum of all local particle counts
    // localParticleCount = sum of particles (on this rank) in all children
    template <class Dataset>
    void updateGlobalCounts(const std::vector<int> &list, Dataset &d)
    {
        std::vector<int> ordering(d.count);

        std::vector<int> globalParticleCount(globalNodeCount, 0);
        std::vector<T> globalMaxH(globalNodeCount, 0.0);

        updateGlobalCountsRec(list, d.x, d.y, d.z, d.h, ordering, globalParticleCount, globalMaxH);

        MPI_Allreduce(MPI_IN_PLACE, &globalParticleCount[0], globalNodeCount, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, &globalMaxH[0], globalNodeCount, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        setParticleCountPerNode(globalParticleCount);
        setMaxHPerNode(globalMaxH);

        reorder(ordering, d);
    }
#endif

    void zCurveHaloUpdateRec(const std::vector<int> &list, const std::vector<T> &x, const std::vector<T> &y, const std::vector<T> &z, std::vector<int> &ordering, int zoffset = 0)
    {
        this->zCurveOffset = zoffset;
        this->localParticleCount = 0;// this->globalParticleCount;

        if(assignee == -1 || assignee == comm_rank || halo == true)
        {
            std::vector<std::vector<int>> cellList(ncells);
            distributeParticles(list, x, y, z, cellList);

            if ((int)cells.size() == ncells)
            {
                for (int i = 0; i < ncells; i++)
                {
                    cells[i]->zCurveHaloUpdateRec(cellList[i], x, y, z, ordering, zoffset);
                    this->localParticleCount += cells[i]->localParticleCount;
                    zoffset += cells[i]->localParticleCount;
                }
            }
            else if(assignee == comm_rank || halo == true)
            {
                // If this is a halo node then we may not have all the particles yet
                // But we know how much space to reserve!
                this->localParticleCount = this->globalParticleCount;
                for (int i = 0; i < (int)list.size(); i++)
                    ordering[zoffset + i] = list[i];
            }
        }
    }

    // Some nodes previously not on this rank
    // are now halos. The tree needs to be reindexed to update its zCurve ordering
    // and accomodate the halo nodes that are going to be populated through halo exchange
    template <class Dataset>
    void zCurveHaloUpdate(const std::vector<int> &list, Dataset &d, int nPlusHalos)
    {
        std::vector<int> ordering(nPlusHalos);
        zCurveHaloUpdateRec(list, d.x, d.y, d.z, ordering);
        reorder(ordering, d);
    }

    void buildTreeRec(const std::vector<int> &list, const std::vector<T> &x, const std::vector<T> &y, const std::vector<T> &z,
                              std::vector<int> &ordering, int zoffset = 0)
    {
        this->zCurveOffset = zoffset;
        this->localParticleCount = list.size();

        std::vector<std::vector<int>> cellList(ncells);
        distributeParticles(list, x, y, z, cellList);

        if((int)cells.size() == 0 && list.size() > bucketSize)
            makeSubCells();

        if(!global && assignee == -1)
            assignee = comm_rank;

        if ((int)cells.size() == ncells)
        {
            for (int i = 0; i < ncells; i++)
            {
                #pragma omp task shared(cellList, x, y, z, ordering) firstprivate(zoffset)
                cells[i]->buildTreeRec(cellList[i], x, y, z, ordering, zoffset);
                zoffset += cellList[i].size();
            }
            #pragma omp taskwait
        }
        else
        {
            for (int i = 0; i < (int)list.size(); i++)
                ordering[zoffset + i] = list[i];
        }
    }

    template <class Dataset>
    void buildTree(const std::vector<int> &list, Dataset &d)
    {
        std::vector<int> ordering(d.count);

        #pragma omp parallel
        #pragma omp single
        buildTreeRec(list, d.x, d.y, d.z, ordering);

        reorder(ordering, d);
    }

    void assignProcessesRec(std::vector<int> &work, int &pi)
    {
        if (work[pi] <= 0 && pi + 1 < (int)work.size()) pi++;

        // If the node fits on process pi, we assign it to this branch
        if (globalParticleCount <= work[pi]) assignee = pi;

        if ((int)cells.size() == ncells)
        {
            for (int i = 0; i < ncells; i++)
                cells[i]->assignProcessesRec(work, pi);
        }
        else
        {
            // Else if we are a leaf and it does not fit on process pi, we go to the next process
            if (globalParticleCount > work[pi] && pi + 1 < (int)work.size())
            {
                pi++;
                // It may not fit
                // If that happens, it means the tree is not well balanced
                // Perhaps increase the sample size?
                if (globalParticleCount > work[pi])
                {
                    printf("Node has %d particles > assignee %d which has max %d work\n", globalParticleCount, pi, work[pi]);
                    printf("Increase sample size?\n");
                }
            }

            assignee = pi;
            work[pi] -= globalParticleCount;
        }
    }

    void assignProcesses(const std::vector<int> &work, std::vector<int> &work_remaining)
    {
        int pi = 0;
        work_remaining = work;
        assignProcessesRec(work_remaining, pi);
    }

    void syncRec(std::map<int, std::vector<int>> &toSendCellsPadding, std::map<int, std::vector<int>> &toSendCellsCount, int &needed)
    {
        if (global)
        {
            // The cell is ours
            if (assignee == comm_rank) { needed += globalParticleCount - localParticleCount; }
            // The cell is associated to a process but is not ours
            else if (assignee >= 0 && localParticleCount > 0)
            {
                toSendCellsPadding[assignee].push_back(zCurveOffset);
                toSendCellsCount[assignee].push_back(localParticleCount);
            }
            // The cell is not associated. If it is not a leaf then
            else if ((int)cells.size() == ncells)
            {
                for (int i = 0; i < ncells; i++)
                {
                    cells[i]->syncRec(toSendCellsPadding, toSendCellsCount, needed);
                }
            }
        }
    }

    int findHalosList(Octree *a, std::map<int, std::map<int, Octree<T> *>> &toSendHalos, int zCurveNodeIdx = 0)
    {
        int haloCount = 0;

        if (global)
        {
            zCurveNodeIdx++;

            // If this node is a halo (i.e. overlap) node a
            if (globalParticleCount > 0)
            {
                if (assignee == -1 && (int)cells.size() == ncells && overlap(a))
                {
                    for (int i = 0; i < ncells; i++)
                    {
                        haloCount += cells[i]->findHalosList(a, toSendHalos, zCurveNodeIdx);
                        zCurveNodeIdx += cells[i]->globalNodeCount;
                    }
                }
                else if (assignee != a->assignee && (assignee == comm_rank || a->assignee == comm_rank) && overlap(a))
                {
                    if (a->assignee == comm_rank) halo = true;

                    if ((int)cells.size() == ncells)
                    {
                        for (int i = 0; i < ncells; i++)
                        {
                            haloCount += cells[i]->findHalosList(a, toSendHalos, zCurveNodeIdx);
                            zCurveNodeIdx += cells[i]->globalNodeCount;
                        }
                    }
                    else
                    {
                        if (toSendHalos[a->assignee].count(zCurveNodeIdx) == 0)
                        {
                            if (a->assignee == comm_rank)
                            {
                                haloCount += globalParticleCount;
                            }
                            toSendHalos[a->assignee][zCurveNodeIdx] = this;
                        }
                    }
                }
            }
        }

        return haloCount;
    }

    int findHalosRec(Octree *root, std::map<int, std::map<int, Octree<T> *>> &toSendHalos, bool PBCx, bool PBCy, bool PBCz)
    {
        int haloCount = 0;

        if (global)
        {
            if (assignee == -1 && (int)cells.size() == ncells)
            {
                for (int i = 0; i < ncells; i++)
                {
                    haloCount += cells[i]->findHalosRec(root, toSendHalos, PBCx, PBCy, PBCz);
                }
            }
            else if (assignee >= 0)
            {
                // Find halos from the root
                // haloCount += root->findHalosList(this, toSendHalos);

                T oldxmin = xmin, oldxmax = xmax;
                T oldymin = ymin, oldymax = ymax;
                T oldzmin = zmin, oldzmax = zmax;

                // Find halos from the root

                int mix = (int)floor(normalize(xmin - 2 * globalMaxH, root->xmin, root->xmax) * nX);
                int miy = (int)floor(normalize(ymin - 2 * globalMaxH, root->ymin, root->ymax) * nY);
                int miz = (int)floor(normalize(zmin - 2 * globalMaxH, root->zmin, root->zmax) * nZ);
                int max = (int)floor(normalize(xmax + 2 * globalMaxH, root->xmin, root->xmax) * nX);
                int may = (int)floor(normalize(ymax + 2 * globalMaxH, root->ymin, root->ymax) * nY);
                int maz = (int)floor(normalize(zmax + 2 * globalMaxH, root->zmin, root->zmax) * nZ);

                if (!PBCx) mix = std::max(mix, 0);
                if (!PBCy) miy = std::max(miy, 0);
                if (!PBCz) miz = std::max(miz, 0);
                if (!PBCx) max = std::min(max, nX - 1);
                if (!PBCy) may = std::min(may, nY - 1);
                if (!PBCz) maz = std::min(maz, nZ - 1);

                for (int hz = miz; hz <= maz; hz++)
                {
                    for (int hy = miy; hy <= may; hy++)
                    {
                        for (int hx = mix; hx <= max; hx++)
                        {
                            T displz = PBCz ? ((hz < 0) - (hz >= nZ)) * (root->zmax - root->zmin) : 0;
                            T disply = PBCy ? ((hy < 0) - (hy >= nY)) * (root->ymax - root->ymin) : 0;
                            T displx = PBCx ? ((hx < 0) - (hx >= nX)) * (root->xmax - root->xmin) : 0;

                            // int hzz = PBCz ? (hz % nZ) + (hz < 0) * nZ : hz;
                            // int hyy = PBCy ? (hy % nY) + (hy < 0) * nY : hy;
                            // int hxx = PBCx ? (hx % nX) + (hx < 0) * nX : hx;

                            // unsigned int l = hzz * nY * nX + hyy * nX + hxx;
                            
                            xmin = xmin + displx;
                            xmax = xmax + displx;
                            ymin = ymin + disply;
                            ymax = ymax + disply;
                            zmin = zmin + displz;
                            zmax = zmax + displz;

                            haloCount += root->findHalosList(this, toSendHalos);

                            xmin = oldxmin;
                            xmax = oldxmax;
                            ymin = oldymin;
                            ymax = oldymax;
                            zmin = oldzmin;
                            zmax = oldzmax;
                        }
                    }
                }
            }
        }

        return haloCount;
    }

    // collect list of nodes to send to other ranks
    // count number of halo particles to receive on this rank
    // tag halo tree nodes on this rank
    int findHalos(std::map<int, std::map<int, Octree<T> *>> &toSendHalos, bool PBCx, bool PBCy, bool PBCz)
    {
        toSendHalos.clear();
        return findHalosRec(this, toSendHalos, PBCx, PBCy, PBCz);
    }

    void mapListRec(std::vector<int> &clist, int &it)
    {
        if((int)cells.size() == ncells)
        {
            if(assignee == -1 || assignee == comm_rank)
            {
                for (int i = 0; i < ncells; i++)
                {
                    cells[i]->mapListRec(clist, it);
                }
            }
        }
        else
        {
            if (assignee == comm_rank && localParticleCount > 0)
            {
                for (int i = 0; i < localParticleCount; i++)
                    clist[it++] = zCurveOffset + i;
            }
        }
    }

    void mapList(std::vector<int> &clist)
    {
        int it = 0;
        mapListRec(clist, it);
    }

    void writeTree(FILE *fout)
    {
        fprintf(fout, "%f %f %f %f %f %f\n", xmin, xmax, ymin, ymax, zmin, zmax);

        if ((global && (assignee == comm_rank || assignee == -1)) && (int)cells.size() == ncells)
        {
            for (int i = 0; i < ncells; i++)
            {
                cells[i]->writeTree(fout);
            }
        }
    }

    void print(int l = 0)
    {
        if(global)
        {
            for (int i = 0; i < l; i++)
                printf("   ");
            printf("[%d] %d %d %d\n", assignee, zCurveOffset, localParticleCount, globalParticleCount);

            if ((int)cells.size() == ncells)
            {
                for (int i = 0; i < ncells; i++)
                    cells[i]->print(l + 1);
            }
        }
    }

private:

    void getParticleCountPerNode(std::vector<int> &particleCount, int zCurveNodeIdx = 0)
    {
        if (global)
        {
            particleCount[zCurveNodeIdx] = this->localParticleCount;

            if ((int)cells.size() == ncells)
            {
                zCurveNodeIdx++;
                for (int i = 0; i < ncells; i++)
                {
                    cells[i]->getParticleCountPerNode(particleCount, zCurveNodeIdx);
                    zCurveNodeIdx += cells[i]->globalNodeCount;
                }
            }
        }
    }

    void setParticleCountPerNode(const std::vector<int> &particleCount, int zCurveNodeIdx = 0)
    {
        if (global)
        {
            this->globalParticleCount = particleCount[zCurveNodeIdx];

            if ((int)cells.size() == ncells)
            {
                zCurveNodeIdx++;
                for (int i = 0; i < ncells; i++)
                {
                    cells[i]->setParticleCountPerNode(particleCount, zCurveNodeIdx);
                    zCurveNodeIdx += cells[i]->globalNodeCount;
                }
            }
        }
    }

    void setMaxHPerNode(const std::vector<double> &hmax, int zCurveNodeIdx = 0)
    {
        if (global)
        {
            this->globalMaxH = hmax[zCurveNodeIdx];

            if ((int)cells.size() == ncells)
            {
                zCurveNodeIdx++;
                for (int i = 0; i < ncells; i++)
                {
                    cells[i]->setMaxHPerNode(hmax, zCurveNodeIdx);
                    zCurveNodeIdx += cells[i]->globalNodeCount;
                }
            }
        }
    }

    static void reorderSwap(const std::vector<int> &ordering, std::vector<T> &arrayList)
    {
        std::vector<T> tmp(ordering.size());
        for (unsigned int i = 0; i < ordering.size(); i++)
            tmp[i] = arrayList[ordering[i]];
        tmp.swap(arrayList);
    }

    static void reorder(const std::vector<int> &ordering, std::vector<std::vector<T> *> &arrayList)
    {
        for (unsigned int i = 0; i < arrayList.size(); i++)
            reorderSwap(ordering, *arrayList[i]);
    }

    template <class Dataset>
    static void reorder(const std::vector<int> &ordering, Dataset &d)
    {
        reorder(ordering, d.data);
    }

    // void mapTasksRec(std::vector<Task> &taskList, int &it)
    // {
    //     if(assignee == comm_rank && localParticleCount < 131072 && localParticleCount > 0)
    //     {
    //         Task task(localParticleCount);
    //         for (int i = 0; i < localParticleCount; i++)
    //             task.clist[i] = zCurveOffset + i;
    //         taskList.push_back(task);
    //     }
    //     else if((int)cells.size() == ncells)
    //     {
    //         if(assignee == -1 || assignee == comm_rank)
    //         {
    //             for (int i = 0; i < ncells; i++)
    //             {
    //                 it++;
    //                 cells[i]->mapTasksRec(taskList, it);
    //             }
    //         }
    //     }
    // }

    // void mapTasks(std::vector<Task> &taskList)
    // {
    //     int it = 0;
    //     taskList.clear();
    //     mapTasksRec(taskList, it);
    // }
};

} // namespace sphexa
