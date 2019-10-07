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
        const int64_t ngmax = d.ngmax;

        int64_t n = clist.size();
        d.neighbors.resize(n * ngmax);
        d.neighborsCount.resize(n);

#pragma omp parallel for schedule(guided)
            for (int pi = 0; pi < n; pi++)
            {
                int i = clist[pi];

                d.neighborsCount[pi] = 0;
                octree.findNeighbors(i, &d.x[0], &d.y[0], &d.z[0], d.x[i], d.y[i], d.z[i], 2.0 * d.h[i], ngmax,
                                     &d.neighbors[pi * ngmax], d.neighborsCount[pi], PBCx, PBCy, PBCz);

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

    void sync(const std::vector<std::vector<T> *> &arrayList)
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

                for (unsigned int i = 0; i < toSendCellsCount[rank].size(); i++)
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
                    for (unsigned int j = 0; j < toSendCellsCount[rank].size(); j++)
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
            // printf("Needed: %d\n", needed);
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

                        // if(comm_rank == 1) printf("[%d] sends cell %d (%d) at %d to %d\n", from, ptri, count, padding, to);
                        // if(padding + count > (*data[i]).size()) printf("ERROR: %d %d %lu\n", padding, count, (*data[i]).size());
                        MPI_Isend(buff, count, MPI_DOUBLE, to, ptri * data.size() + i, MPI_COMM_WORLD, &requests[rcount + i]);
                    }
                }
            }
        }

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

                        // if(comm_rank == 1) printf("[%d] recv cell %d (%d) at %d from %d\n", to, ptri, count, padding, from);
                        // if(padding + count > (*data[i]).size()) printf("ERROR: %d %d\n", padding, count);
                        // printf("Padding: %d\n", padding);

                        MPI_Recv(buff, count, MPI_DOUBLE, from, ptri * data.size() + i, MPI_COMM_WORLD, &status[i]); //&requests[rcount + i]);
                    }
                }
        }

        if (requests.size() > 0)
        {
            MPI_Status status[requests.size()];
            MPI_Waitall(requests.size(), &requests[0], &status[0]);
        }
    }

    void computeGlobalBoundingBox(const std::vector<int> &clist, const std::vector<T> &x, const std::vector<T> &y, const std::vector<T> &z)
    {
        if (!PBCx) xmin = INFINITY;
        if (!PBCx) xmax = -INFINITY;
        if (!PBCy) ymin = INFINITY;
        if (!PBCy) ymax = -INFINITY;
        if (!PBCz) zmin = INFINITY;
        if (!PBCz) zmax = -INFINITY;

        for (unsigned int pi = 0; pi < clist.size(); pi++)
        {
            const int i = clist[pi];

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
    void approximate(std::vector<int> &clist, Dataset &d)
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
        computeGlobalBoundingBox(clist, x, y, z);
        d.bbox.xmin = xmin;
        d.bbox.xmax = xmax;
        d.bbox.ymin = ymin;
        d.bbox.ymax = ymax;
        d.bbox.zmin = zmin;
        d.bbox.zmax = zmax;

        // printf("Global Bounding Box: %f %f %f %f %f %f\n", xmin, xmax, ymin, ymax, zmin, zmax);

        // Each process creates a random sample of local_sample_size particles
        const int global_sample_size = local_sample_size * comm_size;
        const int ptri = local_sample_size * comm_rank;

        std::vector<T> sx(global_sample_size), sy(global_sample_size), sz(global_sample_size), sh(global_sample_size);

        for (int i = 0; i < local_sample_size; i++)
        {
            int j = rand() % n;
            sx[ptri + i] = x[clist[j]];
            sy[ptri + i] = y[clist[j]];
            sz[ptri + i] = z[clist[j]];
            sh[ptri + i] = h[clist[j]];
        }

        MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, &sx[0], local_sample_size, MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, &sy[0], local_sample_size, MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, &sz[0], local_sample_size, MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, &sh[0], local_sample_size, MPI_DOUBLE, MPI_COMM_WORLD);

        // Each process creates a tree based on the gathered sample
        octree.cells.clear();

        octree = GravityOctree<T>(xmin, xmax, ymin, ymax, zmin, zmax, comm_rank, comm_size);
        octree.approximate(sx, sy, sz, sh);

        std::vector<int> ordering(n);
        octree.buildGlobalTreeAndGlobalCountAndGlobalMaxH(clist, x, y, z, h, ordering);
        reorder(ordering, d);

        for (unsigned int i = 0; i < clist.size(); i++)
            clist[i] = i;

        //printf("[%d] Global tree nodes: %d\n", comm_rank, octree.globalNodeCount);
    }

    template <class Dataset>
    void distribute(std::vector<int> &clist, Dataset &d)
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
        computeGlobalBoundingBox(clist, x, y, z);

        d.bbox.xmin = xmin;
        d.bbox.xmax = xmax;
        d.bbox.ymin = ymin;
        d.bbox.ymax = ymax;
        d.bbox.zmin = zmin;
        d.bbox.zmax = zmax;

        // printf("Global Bounding Box: %f %f %f %f %f %f\n", xmin, xmax, ymin, ymax, zmin, zmax); fflush(stdout);

        // We map the nodes to a 1D array and retrieve the order of the particles in the tree
        std::vector<int> ordering(n);
        int nsplits = 0;

        do
        {
            // Done every iteration, this will either add or remove global nodes
            // depending if there are too much / too few particles globally

            // printf("[%d] %d -> %d %d %d %d %d %d %d %d\n", comm_rank, octree.globalNodeCount, octree.cells[0]->globalParticleCount, octree.cells[1]->globalParticleCount, octree.cells[2]->globalParticleCount, octree.cells[3]->globalParticleCount, octree.cells[4]->globalParticleCount, octree.cells[5]->globalParticleCount, octree.cells[6]->globalParticleCount, octree.cells[7]->globalParticleCount);
            
            octree.globalRebalance(xmin, xmax, ymin, ymax, zmin, zmax);

            // printf("[%d] SPLITS: %d, Nodes: %d\n", comm_rank, nsplits, octree.globalNodeCount);
            // fflush(stdout);

            // We now reorder the data in memory so that it matches the octree layout
            // In other words, iterating over the array is the same as walking the tree
            // This is the same a following a Morton / Z-Curve path
            octree.buildGlobalTreeAndGlobalCountAndGlobalMaxH(clist, x, y, z, h, ordering);
            //octree.computeGlobalParticleCount();
        } while(d.iteration == 0 && nsplits > 0);

        // Getting rid of old halos
        reorder(ordering, d);

        // printf("[%d] Global tree nodes: %d\n", comm_rank, octree.globalNodeCount); fflush(stdout);

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

        // printf("[%d] Total Avail: %d, Total Alloc: %d || Got: %d\n", comm_rank, totalAvail, totalAlloc, work[comm_rank] -
        //   work_remaining[comm_rank]); fflush(stdout);

        // Send the particles from the old arrays either to the new arrays or to another process
        // Basically collects a map[process][nodes] to send to other processes
        // Then it knows how many particle it is missing (particleCount - localParticleCount)
        // And just loop receive until we have received all the missing particles from other processes
        sync(d.arrayList);

        // printf("[%d] Total number of particles %d (local) %d (global)\n", comm_rank, octree.localParticleCount, octree.globalParticleCount); fflush(stdout);

        //octree.computeGlobalMaxH();
        haloCount = octree.findHalos(toSendHalos, PBCx, PBCy, PBCz);

        workAssigned = work[comm_rank] - work_remaining[comm_rank];

        // int check = workAssigned;
        // MPI_Allreduce(MPI_IN_PLACE, &check, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        // if(comm_rank == 0)
        //     printf("CHECK: %d\n", check);

        // printf("[%d] haloCount: %d (%.2f%%)\n", comm_rank, haloCount,
        //         haloCount / (double)(workAssigned) * 100.0); fflush(stdout);

        // Finally remap everything
        ordering.resize(workAssigned + haloCount);
        #pragma omp parallel for
        for(int i=0; i<(int)ordering.size(); i++)
            ordering[i] = 0;

        // We map ALL particles
        // Particles that do not belong to us will be ignored in the localMapParticleFunction
        std::vector<int> list(d.x.size());
        #pragma omp parallel for
        for(int i=0; i<(int)d.x.size(); i++)
            list[i] = i;

        octree.buildTreeWithHalos(list, x, y, z, ordering);
        reorder(ordering, d);

        clist.resize(workAssigned);
        #pragma omp parallel for
        for(int i=0; i<workAssigned; i++)
            clist[i] = i;

        octree.mapList(clist);
        d.count = workAssigned;
    }

    template <class Dataset>
    void buildTree(Dataset &d)
    {
        // Finally remap everything
        std::vector<int> ordering(d.x.size());

        std::vector<int> list(d.x.size());
        #pragma omp parallel for
        for(int i=0; i<(int)d.x.size(); i++)
            list[i] = i;

        // We need this to expand halo
        octree.buildTree(list, d.x, d.y, d.z, d.m, ordering);
        reorder(ordering, d);
    }

    template <class Dataset>
    void updateSmoothingLength(const std::vector<int> &clist, Dataset &d)
    {
        const T c0 = 7.0;
        const T exp = 1.0/3.0;

        const size_t n = clist.size();
        const int *neighborsCount = d.neighborsCount.data();
        const int ng0 = d.ng0;
        T *h = d.h.data();

        #pragma omp parallel for
        for(int pi=0; pi<n; pi++)
        {
            const int i = clist[pi];
            const int nn = neighborsCount[pi];

            h[i] = h[i] * 0.5 * pow((1.0 + c0 * ng0 / nn), exp);

            #ifndef NDEBUG
                if(std::isinf(h[i]) || std::isnan(h[i]))
                    printf("ERROR::h(%d) ngi %d h %f\n", i, nn, h[i]);
            #endif
        }
    }

    const int local_sample_size = 100;

    int comm_size, comm_rank, name_len;
    char processor_name[256];

    bool PBCx = false, PBCy = false, PBCz = false;
    T xmin = INFINITY, xmax = -INFINITY, ymin = INFINITY, ymax = -INFINITY, zmin = INFINITY, zmax = -INFINITY;

    std::map<int, std::map<int, Octree<T> *>> toSendHalos;

    int haloCount = 0;
    int workAssigned = 0;

    GravityOctree<T> octree;
};

} // namespace sphexa
