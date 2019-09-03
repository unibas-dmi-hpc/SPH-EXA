#pragma once

#include <cmath>
#include <memory>
#include <vector>
#include <algorithm>

namespace sphexa
{

template<typename T>
class Octree
{
public:
    Octree(const T xmin, const T xmax, const T ymin, const T ymax, const T zmin, const T zmax, int comm_rank, int comm_size) : 
        xmin(xmin), xmax(xmax), ymin(ymin), ymax(ymax), zmin(zmin), zmax(zmax), comm_rank(comm_rank), comm_size(comm_size) {}

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

    inline T normalize(T d, T min, T max) { return (d - min) / (max - min); }

    inline void distributeParticles(const std::vector<int> &list, const std::vector<T> &ax, const std::vector<T> &ay, const std::vector<T> &az, std::vector<std::vector<int>> &cellList)
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

                    if(cells[i] == nullptr)
                        cells[i] = std::make_shared<Octree>(ax, bx, ay, by, az, bz, comm_rank, comm_size);
                }
            }
        }
    }

    void approximateRec(const std::vector<int> &list, const std::vector<T> &x, const std::vector<T> &y, const std::vector<T> &z, const std::vector<T> &h)
    {
        this->globalNodeCount = 1;

        global = true;

        T hmax = 0.0;
        for (unsigned int i = 0; i < list.size(); i++)
        {
            T hh = h[list[i]];
            if (hh > hmax) hmax = hh;
        }

        const T sizex = xmax-xmin;
        const T sizey = ymax-ymin;
        const T sizez = zmax-zmin;

        const T size = std::max(sizez, std::max(sizey, sizex));

        // Expand node if cell bigger than 2.0 * h
        if(size > 2.0*hmax && list.size() > 0)
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

        approximateRec(list, x, y, z, h);
    }

    void localMapParticlesRec(const std::vector<int> &list, const std::vector<T> &x, const std::vector<T> &y, const std::vector<T> &z, const std::vector<T> &h,
        std::vector<int> &ordering, bool expand, int padding = 0)
    {
        this->localPadding = padding;
        this->localParticleCount = 0;
        this->localHmax = 0.0;

        // If processes have been assigned to the tree nodes, it will ignore nodes not assigned to him
        // Thereby also discarding particles that do not belong to the current process
        if(assignee == -1 || assignee == comm_rank)
        {
            std::vector<std::vector<int>> cellList(ncells);
            distributeParticles(list, x, y, z, cellList);

            if(expand == true && assignee == comm_rank && (int)cells.size() == 0 && list.size() > 64)
            {
                makeSubCells();
                for (int i = 0; i < ncells; i++)
                    cells[i]->assignee = comm_rank;
            }

            if((int)cells.size() == ncells)
            {
                for (int i = 0; i < ncells; i++)
                {
                    //if(comm_rank == 0 && (cells[i]->assignee == -1 || cells[i]->assignee == comm_rank)) printf("%d %d %d\n", (int)ordering.size(), padding, (int)cellList[i].size());
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
                for(int i=0; i<(int)list.size(); i++)
                {
                    ordering[padding+i] = list[i];
                    if(h[list[i]] > this->localHmax)
                        this->localHmax = h[list[i]];
                }
            }
        }
        else if(halo)
        {
            // Needed to set the padding correctly in every subnode
            if((int)cells.size() == ncells)
            {
                for (int i = 0; i < ncells; i++)
                {
                    std::vector<std::vector<int>> cellList(ncells);
                    cells[i]->localMapParticlesRec(cellList[i], x, y, z, h, ordering, expand, padding);
                    this->localParticleCount += cells[i]->localParticleCount;
                    padding += cells[i]->localParticleCount;
                }
            }
            else
                this->localParticleCount = globalParticleCount;
        }
    }

    void localMapParticles(const std::vector<T> &x, const std::vector<T> &y, const std::vector<T> &z, const std::vector<T> &h, std::vector<int> &ordering, bool expand)
    {
        std::vector<int> list(x.size());
        for (unsigned int i = 0; i < x.size(); i++)
            list[i] = i;

        localMapParticlesRec(list, x, y, z, h, ordering, expand);
    }

    void assignProcessesRec(std::vector<int> &work, int &pi)
    {
        if(work[pi] <= 0 && pi+1 < (int)work.size())
            pi++;

        // If the node fits on process pi, we assign it to this branch
        if(globalParticleCount <= work[pi])
            assignee = pi;

        if((int)cells.size() == ncells)
        {
            for (int i = 0; i < ncells; i++)
                cells[i]->assignProcessesRec(work, pi);
        }
        else
        {
            // Else if we are a leaf and it does not fit on process pi, we go to the next process
            if(globalParticleCount > work[pi] && pi+1 < (int)work.size())
            {
                pi++;
                // It may not fit
                // If that happens, it means the tree is not well balanced
                // Perhaps increase the sample size?
                if(globalParticleCount > work[pi])
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
        if(global)
        {
            // The cell is ours
            if(assignee == comm_rank)
            {
                needed += globalParticleCount - localParticleCount;
            }
            // The cell is associated to a process but is not ours
            else if(assignee >= 0 && localParticleCount > 0)
            {
                toSendCellsPadding[assignee].push_back(localPadding);
                toSendCellsCount[assignee].push_back(localParticleCount);
            }
            // The cell is not associated. If it is not a leaf then
            else if((int)cells.size() == ncells)
            {
                for (int i = 0; i < ncells; i++)
                {
                    cells[i]->syncRec(toSendCellsPadding, toSendCellsCount, needed);
                }
            }
        }
    }

    void sync(const std::vector<std::vector<T>*> &arrayList)
    {
        std::map<int, std::vector<int>> toSendCellsPadding, toSendCellsCount;
        std::vector<std::vector<T>> buff;
        std::vector<int> counts;
        std::vector<MPI_Request> requests;

        int needed = 0;

        syncRec(toSendCellsPadding, toSendCellsCount, needed);

        counts.resize(comm_size);
        for (int rank = 0; rank < comm_size; rank++)
        {
            if (toSendCellsCount[rank].size() > 0)
            {
                int rcount = requests.size();
                int bcount = buff.size();
                counts[rank] = 0;

                for(unsigned int i=0; i<toSendCellsCount[rank].size(); i++)
                    counts[rank] += toSendCellsCount[rank][i];

                requests.resize(rcount + arrayList.size() + 2);
                buff.resize(bcount + arrayList.size());

                MPI_Isend(&comm_rank, 1, MPI_INT, rank, 0, MPI_COMM_WORLD, &requests[rcount]);
                MPI_Isend(&counts[rank], 1, MPI_INT, rank, 1, MPI_COMM_WORLD, &requests[rcount + 1]);

                for (unsigned int i = 0; i < arrayList.size(); i++)
                {
                    buff[bcount + i].resize(counts[rank]);

                    int bi = 0;
                    // We go over every tree nodes to send for this rank
                    for(unsigned int j=0; j<toSendCellsCount[rank].size(); j++)
                    {
                        int padding = toSendCellsPadding[rank][j];
                        int count = toSendCellsCount[rank][j];
                        for (int k = 0; k < count; k++)
                        {
                            buff[bcount + i][bi++] = (*arrayList[i])[padding + k];
                        }
                    }
                    
                    MPI_Isend(&buff[bcount + i][0], counts[rank], MPI_DOUBLE, rank, 2 + i, MPI_COMM_WORLD, &requests[rcount + 2 + i]);
                }
            }
        }

        int end = arrayList[0]->size();
        for (unsigned int i = 0; i < arrayList.size(); i++)
            (*arrayList[i]).resize(end + needed);

        while (needed > 0)
        {
            //printf("Needed: %d\n", needed);
            MPI_Status status[arrayList.size() + 2];

            int rank, count;
            MPI_Recv(&rank, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status[0]);
            MPI_Recv(&count, 1, MPI_INT, rank, 1, MPI_COMM_WORLD, &status[1]);

            for (unsigned int i = 0; i < arrayList.size(); i++)
                MPI_Recv(&(*arrayList[i])[end], count, MPI_DOUBLE, rank, 2 + i, MPI_COMM_WORLD, &status[2 + i]);

            end += count;
            needed -= count;
        }

        if (requests.size() > 0)
        {
            MPI_Status status[requests.size()];
            MPI_Waitall(requests.size(), &requests[0], status);
        }
    }

    inline bool overlap(T leftA, T rightA, T leftB, T rightB) { return leftA < rightB && rightA > leftB; }
    
    inline bool overlap(Octree *a)
    {
        T radius = a->globalMaxH * 2.0;

        return overlap(a->xmin - radius, a->xmax + radius, xmin, xmax) &&
               overlap(a->ymin - radius, a->ymax + radius, ymin, ymax) &&
               overlap(a->zmin - radius, a->zmax + radius, zmin, zmax);
    }

    int findHalosList(Octree *a, std::map<int, std::map<int, Octree<T>*>> &toSendHalos, int ptri = 0)
    {
        int haloCount = 0;

        if(global)
        {
            ptri++;

            // If this node is a halo (i.e. overlap) node a
            if(globalParticleCount > 0)
            { 
                if(assignee == -1 && (int)cells.size() == ncells && overlap(a))
                {
                    for (int i = 0; i < ncells; i++)
                    {
                        haloCount += cells[i]->findHalosList(a, toSendHalos, ptri);
                        ptri += cells[i]->globalNodeCount;
                    }
                }
                else if(assignee != a->assignee && (assignee == comm_rank || a->assignee == comm_rank) && overlap(a))
                {
                    if(a->assignee == comm_rank)
                        halo = true;

                    if((int)cells.size() == ncells)
                    {
                        for (int i = 0; i < ncells; i++)
                        {
                            haloCount += cells[i]->findHalosList(a, toSendHalos, ptri);
                            ptri += cells[i]->globalNodeCount;
                        }
                    }
                    else
                    {
                        if(a->assignee == comm_rank)
                            haloCount += globalParticleCount;
                        toSendHalos[a->assignee][ptri] = this;
                    }
                }
            }
        }

        return haloCount;
    }

    int findHalosRec(Octree *root, std::map<int, std::map<int, Octree<T>*>> &toSendHalos)
    {
        int haloCount = 0;

        if(global)
        {
            if(assignee == -1 && (int)cells.size() == ncells)
            {
                for (int i = 0; i < ncells; i++)
                {
                    haloCount += cells[i]->findHalosRec(root, toSendHalos);
                }
            }
            else if(assignee >= 0)
            {
                // Find halos from the root
                haloCount += root->findHalosList(this, toSendHalos);
            }
        }

        return haloCount;
    }

    int findHalos(std::map<int, std::map<int, Octree<T>*>> &toSendHalos)
    {
        toSendHalos.clear();
        return findHalosRec(this, toSendHalos);
    }

    void writeTree(FILE *fout)
    {
        fprintf(fout, "%f %f %f %f %f %f\n", xmin, xmax, ymin, ymax, zmin, zmax);
        
        if((global && (assignee == comm_rank || assignee == -1)) && (int)cells.size() == ncells)
        {
            for (int i = 0; i < ncells; i++)
            {
                cells[i]->writeTree(fout);
            }
        }
    }

    void getParticleCountPerNode(std::vector<int> &particleCount, int ptri = 0)
    {
        if(global)
        {
            particleCount[ptri] = this->localParticleCount;

            if((int)cells.size() == ncells)
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
        if(global)
        {
            this->globalParticleCount = particleCount[ptri];

            if((int)cells.size() == ncells)
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
        if(global)
        {
            hmax[ptri] = this->localHmax;

            if((int)cells.size() == ncells)
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
        if(global)
        {
            this->globalMaxH = hmax[ptri];

            if((int)cells.size() == ncells)
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
};

} // namespace sphexa