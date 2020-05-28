#pragma once

#include "Octree.hpp"
#include "Task.hpp"

namespace sphexa
{

template <typename T>
struct GravityOctree : Octree<T>
{
    GravityOctree(const T xmin, const T xmax, const T ymin, const T ymax, const T zmin, const T zmax, const int comm_rank,
                  const int comm_size)
        : Octree<T>(xmin, xmax, ymin, ymax, zmin, zmax, comm_rank, comm_size)
    {
    }

    GravityOctree()
        : Octree<T>()
    {
    }

    ~GravityOctree() = default;

    void gravityBuildTree(const std::vector<int> &list, const std::vector<T> &x, const std::vector<T> &y, const std::vector<T> &z,
                          const std::vector<T> &m, bool withGravitySync = false)
    {
        particleIdxList = list;
        calcGeometricalCenter();
        dx = this->xmax - this->xmin;

        zeroGravValues();

        for (const auto i : list)
        {
            const T xx = x[i];
            const T yy = y[i];
            const T zz = z[i];

            const T m_i = m[i];

            xcm += xx * m_i;
            ycm += yy * m_i;
            zcm += zz * m_i;

            mTot += m_i;

            const T rx = xx - xce;
            const T ry = yy - yce;
            const T rz = zz - zce;

            qxxa += rx * rx * m_i;
            qxya += rx * ry * m_i;
            qxza += rx * rz * m_i;
            qyya += ry * ry * m_i;
            qyza += ry * rz * m_i;
            qzza += rz * rz * m_i;
        }

        if (withGravitySync) gatherGravValues();

        const size_t noParticles = list.size();
        if (noParticles > 1 || this->global)
        {
            xcm /= mTot;
            ycm /= mTot;
            zcm /= mTot;

            const T rx = xce - xcm;
            const T ry = yce - ycm;
            const T rz = zce - zcm;
            qxx = qxxa - rx * rx * mTot;
            qxy = qxya - rx * ry * mTot;
            qxz = qxza - rx * rz * mTot;
            qyy = qyya - ry * ry * mTot;
            qyz = qyza - ry * rz * mTot;
            qzz = qzza - rz * rz * mTot;

            trq = qxx + qyy + qzz;
        }
        else if (noParticles == 1)
        {
            // save the future particleIdx after reordering
            particleIdx = this->localPadding;
            const auto idx = list.front();

            xcm = x[idx];
            ycm = y[idx];
            zcm = z[idx];

            xce = x[idx];
            yce = y[idx];
            zce = z[idx];

            qxx = 0;
            qxy = 0;
            qxz = 0;
            qyy = 0;
            qyz = 0;
            qzz = 0;

            trq = 0;
            dx = 0; // used to indicate that node is a leaf
        }
    }

    inline void calcGeometricalCenter()
    {
        xce = (this->xmin + this->xmax) / 2.0;
        yce = (this->ymin + this->ymax) / 2.0;
        zce = (this->zmin + this->zmax) / 2.0;
    }

    void buildTreeRec(const std::vector<int> &list, const std::vector<T> &x, const std::vector<T> &y, const std::vector<T> &z,
                      const std::vector<T> &m, std::vector<int> &ordering, int padding = 0) override
    {
        this->localPadding = padding;
        this->localParticleCount = list.size();

        gravityBuildTree(list, x, y, z, m);

        std::vector<std::vector<int>> cellList(Octree<T>::ncells);
        this->distributeParticles(list, x, y, z, cellList);

        if ((int)this->cells.size() == 0 && list.size() > this->bucketSize) this->makeSubCells();

        if (!this->global && this->assignee == -1) this->assignee = this->comm_rank;

        if ((int)this->cells.size() == this->ncells)
        {
            for (int i = 0; i < this->ncells; i++)
            {
                if (list.size() < this->noParticlesThatPreventParallelTaskCreation)
                {
                    this->cells[i]->buildTreeRec(cellList[i], x, y, z, m, ordering, padding);
                    padding += cellList[i].size();
                }
                else
                {
#pragma omp task shared(cellList, x, y, z, m, ordering) firstprivate(padding)
                    this->cells[i]->buildTreeRec(cellList[i], x, y, z, m, ordering, padding);
                    padding += cellList[i].size();
                }
            }
#pragma omp taskwait
        }
        else
        {
            for (int i = 0; i < (int)list.size(); i++)
                ordering[padding + i] = list[i];
        }
    }

    void makeSubCells() override
    {
        this->cells.resize(this->ncells);

        for (int hz = 0; hz < this->nZ; hz++)
        {
            for (int hy = 0; hy < this->nY; hy++)
            {
                for (int hx = 0; hx < this->nX; hx++)
                {
                    T ax = this->xmin + hx * (this->xmax - this->xmin) / this->nX;
                    T bx = this->xmin + (hx + 1) * (this->xmax - this->xmin) / this->nX;
                    T ay = this->ymin + hy * (this->ymax - this->ymin) / this->nY;
                    T by = this->ymin + (hy + 1) * (this->ymax - this->ymin) / this->nY;
                    T az = this->zmin + hz * (this->zmax - this->zmin) / this->nZ;
                    T bz = this->zmin + (hz + 1) * (this->zmax - this->zmin) / this->nZ;

                    unsigned int i = hz * this->nX * this->nY + hy * this->nX + hx;

                    if (this->cells[i] == nullptr)
                        this->cells[i] = std::make_shared<GravityOctree>(ax, bx, ay, by, az, bz, this->comm_rank, this->comm_size);
                }
            }
        }
    }
    using Octree<T>::print;

    void buildGlobalGravityTree(const std::vector<T> &x, const std::vector<T> &y, const std::vector<T> &z, const std::vector<T> &m)
    {
        zeroGravValues();
        for (const auto &cell : this->cells)
        {
            asGravityOctree(cell)->zeroGravityValuesInHaloNodesRec();
        }

        buildGlobalParticleListRec();
        for (const auto &cell : this->cells)
        {
            asGravityOctree(cell)->buildGlobalParticleListRec();
        }

        for (const auto &cell : this->cells)
        {
            asGravityOctree(cell)->buildGlobalGravityTreeRec(x, y, z, m);
        }
    }

    void buildGlobalGravityTreeRec(const std::vector<T> &x, const std::vector<T> &y, const std::vector<T> &z, const std::vector<T> &m)
    {
        if (this->global)
        {
            const bool withGravitySync = true;
            gravityBuildTree(globalParticleIdxList, x, y, z, m, withGravitySync);

            for (int i = 0; i < this->ncells; i++)
            {
                if ((int)this->cells.size() == this->ncells) { asGravityOctree(this->cells[i])->buildGlobalGravityTreeRec(x, y, z, m); }
            }
        }
    }

    bool doAllChildCellsBelongToGlobalTree()
    {
        return (int)this->cells.size() == this->ncells &&
               std::all_of(this->cells.begin(), this->cells.end(), [](const auto &t) { return t->global; });
    }

    void zeroGravityValuesInHaloNodesRec()
    {

        if (doAllChildCellsBelongToGlobalTree())
        {
            for (const auto &cell : this->cells)
            {
                asGravityOctree(cell)->zeroGravityValuesInHaloNodesRec();
            }
        } // traverse bottom up

        pcount = particleIdxList.size();
        globalParticleIdxList = particleIdxList;

        if (this->halo)
        {
            zeroGravValues();
            pcount = 0;
            globalParticleIdxList.clear();
        }
    }

    void buildGlobalParticleListRec()
    {

        if (doAllChildCellsBelongToGlobalTree())
        {
            pcount = 0;
            globalParticleIdxList.clear();

            const size_t accSize =
                std::accumulate(this->cells.begin(), this->cells.end(), size_t(0), [this](const size_t acc, const auto &el) {
                    return asGravityOctree(el)->globalParticleIdxList.size() + acc;
                });
            globalParticleIdxList.reserve(accSize);
            pcount = accSize;

            for (const auto &cell : this->cells)
            {
                GravityOctree<T> *gptr = asGravityOctree(cell);
                std::copy(gptr->globalParticleIdxList.begin(), gptr->globalParticleIdxList.end(),
                          std::back_inserter(globalParticleIdxList));
            }
            for (const auto &cell : this->cells)
            {
                asGravityOctree(cell)->buildGlobalParticleListRec();
            }
        }
    }

    void zeroGravValues()
    {
        qxxa = 0.0, qxya = 0.0, qxza = 0.0;
        qyya = 0.0, qyza = 0.0;
        qzza = 0.0;
        xcm = 0.0;
        ycm = 0.0;
        zcm = 0.0;
        mTot = 0.0;

        qxx = 0.0, qxy = 0.0, qxz = 0.0;
        qyy = 0.0, qyz = 0.0;
        qzz = 0.0;
        trq = 0.0;
    }

    void print(int l = 0)
    {
        if (this->global)
        {
            for (int i = 0; i < l; i++)
                printf("   ");
            printf("[%d] lp:%d gp:%d gn:%d h:%d mTot:%.15f qxx:%.15f trq:%.15f ycm:%.15f pcount=%d, globalParticleIdxList.size=%lu\n",
                   this->assignee, this->localParticleCount, this->globalParticleCount, this->globalNodeCount, this->halo, mTot, qxx, trq,
                   ycm, pcount, globalParticleIdxList.size());

            if ((int)this->cells.size() == this->ncells)
            {
                for (int i = 0; i < this->ncells; i++)
                {
                    asGravityOctree(this->cells[i])->print(l + 1);
                }
            }
        }
    }

    inline GravityOctree<T> *asGravityOctree(const std::shared_ptr<Octree<T>> &octree)
    {
        return dynamic_cast<GravityOctree<T> *>(octree.get());
    }

    void gatherGravValues()
    {
#ifdef USE_MPI
        // if (this->global)
        if (this->global && this->assignee == -1)
        {
            //	  printf("[%d] MPI_Allreduce in tree node with %lu particles and mTot=%.15f\n", this->comm_rank, pcount, mTot);
            MPI_Allreduce(MPI_IN_PLACE, &mTot, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

            MPI_Allreduce(MPI_IN_PLACE, &xcm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(MPI_IN_PLACE, &ycm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(MPI_IN_PLACE, &zcm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

            MPI_Allreduce(MPI_IN_PLACE, &qxxa, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(MPI_IN_PLACE, &qxya, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(MPI_IN_PLACE, &qxza, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(MPI_IN_PLACE, &qyya, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(MPI_IN_PLACE, &qyza, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(MPI_IN_PLACE, &qzza, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        }
#endif
    }

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

    std::vector<int> particleIdxList;
    std::vector<int> globalParticleIdxList;

    T dx;                // side of a cell;
    int particleIdx = 0; // filled only if node is a leaf
};

}
