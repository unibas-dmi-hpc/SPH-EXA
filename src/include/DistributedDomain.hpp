#pragma once

#ifdef USE_MPI
#include "mpi.h"
#endif

#include <vector>
#include <cmath>
#include <map>
#include <algorithm>

#include <unistd.h>

#include "Octree.hpp"

namespace sphexa
{

template <typename T>
class DistributedDomain
{
public:
    DistributedDomain()
    {
        MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
        MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
        MPI_Get_processor_name(processor_name, &name_len);
    }

    template <class Dataset>
    void findNeighbors(const std::vector<int> &clist, Dataset &d)
    {
        const int ngmax = d.ngmax;

        int64_t n = clist.size();
        d.neighbors.resize(n * ngmax);
        d.neighborsCount.resize(n);

#pragma omp parallel for schedule(guided)
        for (int pi = 0; pi < n; pi++)
        {
            int i = clist[pi];

            d.neighborsCount[pi] = 0;
            octree.findNeighbors(i, d.x.data(), d.y.data(), d.z.data(), d.x[i], d.y[i], d.z[i], 2 * d.h[i], ngmax, &d.neighbors[pi * ngmax], d.neighborsCount[pi], d.bbox.PBCx,
                               d.bbox.PBCy, d.bbox.PBCz);

#ifndef NDEBUG
            if (d.neighbors[pi].size() == 0)
                printf("ERROR::FindNeighbors(%d) x %f y %f z %f h = %f ngi %zu\n", i, x[i], y[i], z[i], h[i], neighborsCount[pi]);
#endif
        }
    }

    template <class Dataset>
    int64_t neighborsSum(const std::vector<int> &clist, const Dataset &d)
    {
        int64_t sum = 0;
#pragma omp parallel for reduction(+ : sum)
        for (unsigned int i = 0; i < clist.size(); i++)
            sum += d.neighborsCount[i];

#ifdef USE_MPI
        MPI_Allreduce(MPI_IN_PLACE, &sum, 1, MPI_LONG_LONG_INT, MPI_SUM, MPI_COMM_WORLD);
#endif

        return sum;
    }

    void setBox(T xmin, T xmax, T ymin, T ymax, T zmin, T zmax, bool PBCx, bool PBCy, bool PBCz)
    {
        this->xmin = xmin;
        this->xmax = xmax;
        this->ymin = ymin;
        this->ymax = ymax;
        this->zmin = zmin;
        this->zmax = zmax;
        this->PBCx = PBCx;
        this->PBCy = PBCy;
        this->PBCz = PBCz;
    }
    
    inline T normalize(T d, T min, T max) { return (d - min) / (max - min); }

    void reorderSwap(const std::vector<int> &ordering, std::vector<T> &arrayList)
    {
        std::vector<T> tmp(ordering.size());
        for (unsigned int i = 0; i < ordering.size(); i++)
            tmp[i] = arrayList[ordering[i]];
        tmp.swap(arrayList);
    }

    void reorder(const std::vector<int> &ordering, std::vector<std::vector<T> *> &arrayList)
    {
        for (unsigned int i = 0; i < arrayList.size(); i++)
            reorderSwap(ordering, *arrayList[i]);
    }

    template <class Dataset>
    void reorder(const std::vector<int> &ordering, Dataset &d)
    {
        reorder(ordering, d.arrayList);
    }

    void makeDataArray(std::vector<std::vector<T> *> &data, std::vector<T> *d) { data.push_back(d); }

    template <typename... Args>
    void makeDataArray(std::vector<std::vector<T> *> &data, std::vector<T> *first, Args... args)
    {
        data.push_back(first);
        makeDataArray(data, args...);
    }

    void sync(const std::vector<std::vector<T>*> &arrayList, const std::vector<int> &ordering)
    {
        std::map<int, std::vector<int>> toSendCellsPadding, toSendCellsCount;
        std::vector<std::vector<T>> buff;
        std::vector<int> counts;
        std::vector<MPI_Request> requests;

        int needed = 0;

        octree.syncRec(toSendCellsPadding, toSendCellsCount, needed);

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
                            buff[bcount + i][bi++] = (*arrayList[i])[ordering[padding + k]];
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

    template <typename... Args>
    void synchronizeHalos(Args... args)
    {
        std::vector<std::vector<T> *> data;
        makeDataArray(data, args...);
        synchronizeHalos(data);
    }

    void synchronizeHalos(std::vector<std::vector<T> *> &data)
    {
        // std::vector<std::vector<T>> buff;
        std::vector<MPI_Request> requests;

        for (auto const &itProc : toSendHalos)
        {
            int to = itProc.first;

            for (auto const &itNodes : itProc.second)
            {
                int ptri = itNodes.first;
                Octree<T> *cell = itNodes.second;

                int count = cell->globalParticleCount;
                int padding = cell->localPadding;
                int from = cell->assignee;

                if (from == comm_rank)
                {
                    int rcount = requests.size();
                    requests.resize(rcount + data.size());

                    for (unsigned int i = 0; i < data.size(); i++)
                    {
                        T *buff = &(*data[i])[padding];

                        //if(comm_rank == 1) printf("[%d] sends cell %d (%d) at %d to %d\n", from, ptri, count, padding, to);
                        //if(padding + count > (*data[i]).size()) printf("ERROR: %d %d %lu\n", padding, count, (*data[i]).size());
                        MPI_Isend(buff, count, MPI_DOUBLE, to, ptri, MPI_COMM_WORLD, &requests[rcount + i]);
                    }
                }
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);

        for (auto const &itProc : toSendHalos)
        {
            int to = itProc.first;

            if (to == comm_rank)

                for (auto const &itNodes : itProc.second)
                {
                    int ptri = itNodes.first;
                    Octree<T> *cell = itNodes.second;

                    int count = cell->globalParticleCount;
                    int padding = cell->localPadding;
                    int from = cell->assignee;

                    MPI_Status status[data.size()];

                    for (unsigned int i = 0; i < data.size(); i++)
                    {
                        T *buff = &(*data[i])[padding];

                        //if(comm_rank == 0) printf("[%d] recv cell %d (%d) at %d from %d\n", to, ptri, count, padding, from);
                        //if(padding + count > (*data[i]).size()) printf("ERROR: %d %d\n", padding, count);
                        MPI_Recv(buff, count, MPI_DOUBLE, from, ptri, MPI_COMM_WORLD, &status[i]); //&requests[rcount + i]);
                    }
                }
        }

        if (requests.size() > 0)
        {
            MPI_Status status[requests.size()];
            MPI_Waitall(requests.size(), &requests[0], &status[0]);
        }
    }

    void computeGlobalBoundingBox(const int n, const std::vector<T> &x, const std::vector<T> &y, const std::vector<T> &z)
    {
        if (!PBCx) xmin = INFINITY;
        if (!PBCx) xmax = -INFINITY;
        if (!PBCy) ymin = INFINITY;
        if (!PBCy) ymax = -INFINITY;
        if (!PBCz) zmin = INFINITY;
        if (!PBCz) zmax = -INFINITY;

        for (int i = 0; i < n; i++)
        {
            T xx = x[i];
            T yy = y[i];
            T zz = z[i];

            if (!PBCx && xx < xmin) xmin = xx;
            if (!PBCx && xx > xmax) xmax = xx;
            if (!PBCy && yy < ymin) ymin = yy;
            if (!PBCy && yy > ymax) ymax = yy;
            if (!PBCz && zz < zmin) zmin = zz;
            if (!PBCz && zz > zmax) zmax = zz;
        }

        MPI_Allreduce(MPI_IN_PLACE, &xmin, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, &ymin, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, &zmin, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, &xmax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, &ymax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, &zmax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    }

    template <class Dataset>
    void approximate(Dataset &d)
    {
        const std::vector<T> &x = d.x;
        const std::vector<T> &y = d.y;
        const std::vector<T> &z = d.z;
        const std::vector<T> &h = d.h;

        // Prepare processes
        const int n = d.count;
        const int ntot = d.n;
        const int split = ntot / comm_size;
        const int remaining = ntot - comm_size * split;

        std::vector<int> work(comm_size, split);
        work[0] += remaining;

        // We compute the global bounding box of the global domain
        // All processes will have the same box dimensions for the domain
        computeGlobalBoundingBox(n, x, y, z);

        //printf("Global Bounding Box: %f %f %f %f %f %f\n", xmin, xmax, ymin, ymax, zmin, zmax);

        // Each process creates a random sample of local_sample_size particles
        const int global_sample_size = local_sample_size * comm_size;
        const int ptri = local_sample_size * comm_rank;

        std::vector<T> sx(global_sample_size), sy(global_sample_size), sz(global_sample_size), sh(global_sample_size);

        for (int i = 0; i < local_sample_size; i++)
        {
            int j = rand() % n;
            sx[ptri + i] = x[j];
            sy[ptri + i] = y[j];
            sz[ptri + i] = z[j];
            sh[ptri + i] = h[j];
        }

        MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, &sx[0], local_sample_size, MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, &sy[0], local_sample_size, MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, &sz[0], local_sample_size, MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, &sh[0], local_sample_size, MPI_DOUBLE, MPI_COMM_WORLD);

        // Each process creates a tree based on the gathered sample
        octree = Octree<T>(xmin, xmax, ymin, ymax, zmin, zmax, comm_rank, comm_size);
        octree.approximate(sx, sy, sz, sh);

        printf("[%d] Global tree nodes: %d\n", comm_rank, octree.globalNodeCount);
    }

    template <class Dataset>
    void distribute(Dataset &d)
    {
        const std::vector<T> &x = d.x;
        const std::vector<T> &y = d.y;
        const std::vector<T> &z = d.z;
        const std::vector<T> &h = d.h;

        // Prepare processes
        const int n = d.count;
        const int ntot = d.n;
        const int split = ntot / comm_size;
        const int remaining = ntot - comm_size * split;

        std::vector<int> work(comm_size, split);
        work[0] += remaining;

        // We compute the global bounding box of the global domain
        // All processes will have the same box dimensions for the domain
        computeGlobalBoundingBox(n, x, y, z);

        printf("Global Bounding Box: %f %f %f %f %f %f\n", xmin, xmax, ymin, ymax, zmin, zmax);

        //printf("[%d] Global tree nodes: %d\n", comm_rank, octree.globalNodeCount);

        // We map the nodes to a 1D array and retrieve the order of the particles in the tree
        std::vector<int> ordering(n);

        // We now reorder the data in memory so that it matches the octree layout
        // In other words, iterating over the array is the same as walking the tree
        // This is the same a following a Morton / Z-Curve path
        octree.localMapParticles(x, y, z, h, ordering, false);
        octree.computeGlobalParticleCount();
        //reorder(ordering, d);

        // We then map the tree to processes
        std::vector<int> work_remaining(comm_size);
        octree.assignProcesses(work, work_remaining);

        // Quick check
        int totalAvail = 0, totalAlloc = 0;
        for (int i = 0; i < (int)work.size(); i++)
        {
            totalAlloc += work[i] - work_remaining[i];
            totalAvail += work[i];
        }

        //printf("[%d] Total Avail: %d, Total Alloc: %d || Got: %d\n", comm_rank, totalAvail, totalAlloc, work[comm_rank] - work_remaining[comm_rank]);

        // Send the particles from the old arrays either to the new arrays or to another process
        // Basically collects a map[process][nodes] to send to other processes
        // Then it knows how many particle it is missing (particleCount - localParticleCount)
        // And just loop receive until we have received all the missing particles from other processes
        sync(d.arrayList, ordering);

        printf("[%d] Total number of particles %d (local) %d (global)\n", comm_rank, octree.localParticleCount, octree.globalParticleCount);

        octree.computeGlobalMaxH();
        int haloCount = octree.findHalos(toSendHalos, PBCx, PBCy, PBCz);

        printf("[%d] haloCount: %d (%.2f%%)\n", comm_rank, haloCount, haloCount / (double)(work[comm_rank] - work_remaining[comm_rank]) * 100.0);

        // Finally remap everything
        ordering.resize(work[comm_rank] - work_remaining[comm_rank] + haloCount);
        for (unsigned int i = 0; i < ordering.size(); i++)
            ordering[i] = 0;

        octree.localMapParticles(x, y, z, h, ordering, true);
        reorder(ordering, d);
    }

    const int local_sample_size = 100;

    int comm_size, comm_rank, name_len;
    char processor_name[256];

    bool PBCx = false, PBCy = false, PBCz = false;
    T xmin = INFINITY, xmax = -INFINITY, ymin = INFINITY, ymax = -INFINITY, zmin = INFINITY, zmax = -INFINITY;

    std::map<int, std::map<int, Octree<T> *>> toSendHalos;

    int haloCount = 0;

    Octree<T> octree;
};

} // namespace sphexa