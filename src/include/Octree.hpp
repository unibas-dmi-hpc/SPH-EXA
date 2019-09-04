#pragma once

#include <cmath>
#include <memory>
#include <vector>
#include <algorithm>

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

    std::vector<std::shared_ptr<Octree>> cells;

    T xmin = INFINITY, xmax = -INFINITY, ymin = INFINITY, ymax = -INFINITY, zmin = INFINITY, zmax = -INFINITY;

    int comm_rank = -1;
    int comm_size = -1;
    int assignee = -1;

    int localPadding = 0;

    int globalNodeCount = 0;

    int localParticleCount = 0;
    int globalParticleCount = 0;

    T localHmax = 0.0;
    T globalMaxH = 0.0;

    bool halo = false;

    bool global = false;

    static const int nX = 2, nY = 2, nZ = 2;
    static const int ncells = 8;
    static const int bucketSize = 64;

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

        //int maxCount = std::min(localParticleCount, ngmax - neighborsCount);

        for (int i = 0; i < localParticleCount && neighborsCount < ngmax; i++)
        {
            int ordi = localPadding + i;

            T dist = distancesq(xi, yi, zi, x[ordi], y[ordi], z[ordi]);
            if (dist < r2 && ordi != id) neighbors[neighborsCount++] = ordi;
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

    void print(int l = 0)
    {
        for (int i = 0; i < l; i++)
            printf("   ");
        printf("%d %d %d\n", localPadding, localParticleCount, globalParticleCount);

        if ((int)cells.size() == ncells)
        {
            for (int i = 0; i < ncells; i++)
                cells[i]->print(l + 1);
        }
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
        if (size > 8.0 * hmax && list.size() > 0)
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

    void localMapParticlesRec(const std::vector<int> &list, const std::vector<T> &x, const std::vector<T> &y, const std::vector<T> &z,
                              const std::vector<T> &h, std::vector<int> &ordering, bool expand, int padding = 0)
    {
        this->localPadding = padding;
        this->localParticleCount = 0;
        this->localHmax = 0.0;

        // If processes have been assigned to the tree nodes, it will ignore nodes not assigned to him
        // Thereby also discarding particles that do not belong to the current process
        if (assignee == -1 || assignee == comm_rank)
        {
            std::vector<std::vector<int>> cellList(ncells);
            distributeParticles(list, x, y, z, cellList);

            if (expand == true && assignee == comm_rank && (int)cells.size() == 0 && list.size() > bucketSize)
            {
                makeSubCells();
                for (int i = 0; i < ncells; i++)
                    cells[i]->assignee = comm_rank;
            }

            if ((int)cells.size() == ncells)
            {
                for (int i = 0; i < ncells; i++)
                {
                    // if(comm_rank == 0 && (cells[i]->assignee == -1 || cells[i]->assignee == comm_rank)) printf("%d %d %d\n",
                    // (int)ordering.size(), padding, (int)cellList[i].size());
                    cells[i]->localMapParticlesRec(cellList[i], x, y, z, h, ordering, expand, padding);
                    this->localHmax = std::max(this->localHmax, cells[i]->localHmax);
                    this->localParticleCount += cells[i]->localParticleCount;
                    padding += cells[i]->localParticleCount;
                }
            }
            else
            {
                this->localHmax = 0.0;
                this->localParticleCount = list.size();
                for (int i = 0; i < (int)list.size(); i++)
                {
                    ordering[padding + i] = list[i];
                    if (h[list[i]] > this->localHmax) this->localHmax = h[list[i]];
                }
            }
        }
        else if (halo)
        {
            std::vector<std::vector<int>> cellList(ncells);
            distributeParticles(list, x, y, z, cellList);

            // if (expand == true && (int)cells.size() == 0 && list.size() > bucketSize)
            // {
            //     makeSubCells();
            //     for (int i = 0; i < ncells; i++)
            //     {
            //         cells[i]->halo = true;
            //         cells[i]->assignee = assignee;
            //     }
            // }

            // Needed to set the padding correctly in every subnode
            if ((int)cells.size() == ncells)
            {
                for (int i = 0; i < ncells; i++)
                {
                    cells[i]->localMapParticlesRec(cellList[i], x, y, z, h, ordering, expand, padding);
                    this->localParticleCount += cells[i]->localParticleCount;
                    padding += cells[i]->localParticleCount;
                }
            }
            else if (global)
                // We are expanding before halos insertion (global only)
                this->localParticleCount = globalParticleCount;
/*            else
                // We are expanding after halos insertion
                this->localParticleCount = list.size();*/
        }
    }

    void localMapParticles(const std::vector<T> &x, const std::vector<T> &y, const std::vector<T> &z, const std::vector<T> &h,
                           std::vector<int> &ordering, bool expand)
    {
        std::vector<int> list(x.size());
        for (unsigned int i = 0; i < x.size(); i++)
            list[i] = i;

        localMapParticlesRec(list, x, y, z, h, ordering, expand);
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
                toSendCellsPadding[assignee].push_back(localPadding);
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

    inline bool overlap(T leftA, T rightA, T leftB, T rightB) { return leftA < rightB && rightA > leftB; }

    inline bool overlap(Octree *a)
    {
        T radius = a->globalMaxH * 2.0;

        return overlap(a->xmin - radius, a->xmax + radius, xmin, xmax) && overlap(a->ymin - radius, a->ymax + radius, ymin, ymax) &&
               overlap(a->zmin - radius, a->zmax + radius, zmin, zmax);
    }

    int findHalosList(Octree *a, std::map<int, std::map<int, Octree<T> *>> &toSendHalos, int ptri = 0)
    {
        int haloCount = 0;

        if (global)
        {
            ptri++;

            // If this node is a halo (i.e. overlap) node a
            if (globalParticleCount > 0)
            {
                if (assignee == -1 && (int)cells.size() == ncells && overlap(a))
                {
                    for (int i = 0; i < ncells; i++)
                    {
                        haloCount += cells[i]->findHalosList(a, toSendHalos, ptri);
                        ptri += cells[i]->globalNodeCount;
                    }
                }
                else if (assignee != a->assignee && (assignee == comm_rank || a->assignee == comm_rank) && overlap(a))
                {
                    if (a->assignee == comm_rank) halo = true;

                    if ((int)cells.size() == ncells)
                    {
                        for (int i = 0; i < ncells; i++)
                        {
                            haloCount += cells[i]->findHalosList(a, toSendHalos, ptri);
                            ptri += cells[i]->globalNodeCount;
                        }
                    }
                    else
                    {

                        if (a->assignee == comm_rank)
                        {
                            if (toSendHalos[a->assignee].count(ptri) == 0) { haloCount += globalParticleCount; }
                        }
                        toSendHalos[a->assignee][ptri] = this;
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

                            int hzz = PBCz ? (hz % nZ) + (hz < 0) * nZ : hz;
                            int hyy = PBCy ? (hy % nY) + (hy < 0) * nY : hy;
                            int hxx = PBCx ? (hx % nX) + (hx < 0) * nX : hx;

                            unsigned int l = hzz * nY * nX + hyy * nX + hxx;

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

    int findHalos(std::map<int, std::map<int, Octree<T> *>> &toSendHalos, bool PBCx, bool PBCy, bool PBCz)
    {
        toSendHalos.clear();
        return findHalosRec(this, toSendHalos, PBCx, PBCy, PBCz);
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

    void getParticleCountPerNode(std::vector<int> &particleCount, int ptri = 0)
    {
        if (global)
        {
            particleCount[ptri] = this->localParticleCount;

            if ((int)cells.size() == ncells)
            {
                ptri++;
                for (int i = 0; i < ncells; i++)
                {
                    cells[i]->getParticleCountPerNode(particleCount, ptri);
                    ptri += cells[i]->globalNodeCount;
                }
            }
        }
    }

    void setParticleCountPerNode(const std::vector<int> &particleCount, int ptri = 0)
    {
        if (global)
        {
            this->globalParticleCount = particleCount[ptri];

            if ((int)cells.size() == ncells)
            {
                ptri++;
                for (int i = 0; i < ncells; i++)
                {
                    cells[i]->setParticleCountPerNode(particleCount, ptri);
                    ptri += cells[i]->globalNodeCount;
                }
            }
        }
    }

    void computeGlobalParticleCount()
    {
        std::vector<int> localParticleCount(globalNodeCount), globalParticleCount(globalNodeCount);

        getParticleCountPerNode(localParticleCount);

        MPI_Allreduce(&localParticleCount[0], &globalParticleCount[0], globalNodeCount, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

        setParticleCountPerNode(globalParticleCount);
    }

    void getMaxHPerNode(std::vector<double> &hmax, int ptri = 0)
    {
        if (global)
        {
            hmax[ptri] = this->localHmax;

            if ((int)cells.size() == ncells)
            {
                ptri++;
                for (int i = 0; i < ncells; i++)
                {
                    cells[i]->getMaxHPerNode(hmax, ptri);
                    ptri += cells[i]->globalNodeCount;
                }
            }
        }
    }

    void setMaxHPerNode(const std::vector<double> &hmax, int ptri = 0)
    {
        if (global)
        {
            this->globalMaxH = hmax[ptri];

            if ((int)cells.size() == ncells)
            {
                ptri++;
                for (int i = 0; i < ncells; i++)
                {
                    cells[i]->setMaxHPerNode(hmax, ptri);
                    ptri += cells[i]->globalNodeCount;
                }
            }
        }
    }

    void computeGlobalMaxH()
    {
        std::vector<double> globalMaxH(globalNodeCount);

        getMaxHPerNode(globalMaxH);

        MPI_Allreduce(MPI_IN_PLACE, &globalMaxH[0], globalNodeCount, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        setMaxHPerNode(globalMaxH);
    }

    void mapListRec(std::vector<int> &clist, int &it)
    {
        if (assignee == comm_rank)
        {
            for (int i = 0; i < localParticleCount; i++)
            {
                clist[it++] = localPadding + i;
            }
        }
        else if (assignee == -1 && (int)cells.size())
        {
            for (int i = 0; i < ncells; i++)
            {
                cells[i]->mapListRec(clist, it);
                cells[i]->localParticleCount;
            }
        }
    }

    void mapList(std::vector<int> &clist)
    {
        int it = 0;
        mapListRec(clist, it);
    }
};

} // namespace sphexa