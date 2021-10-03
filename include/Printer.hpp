#pragma once

#include <iomanip>

namespace sphexa
{

template<typename Dataset>
class Printer
{
public:
    Printer(const Dataset& d)
        : d(d)
    {
    }

    void printTree(GravityOctree<double>& octree, [[maybe_unused]] std::ostream& out) { octree.print(); }

    void printConstants(size_t iteration, size_t nntot, std::ostream& out) const
    {
        if (d.rank == 0)
        {
            out << iteration << ' ' << d.ttot
                << ' '
                //<< d.minTmpDt << ' ' << d.minDmy << ' '
                << d.minDt << ' ' << d.etot << ' ' << d.ecin << ' ' << d.eint << ' ';
#ifdef GRAVITY
            out << d.egrav << ' ';
#endif
            out << nntot << ' ' << std::endl;
            out.flush();
        }
    }

    void printCheck(const size_t particleCount, const size_t nodeCount, const size_t haloCount,
                    const size_t totalNeighbors, std::ostream& out) const
    {
        out << "### Check ### Global Tree Nodes: " << nodeCount << ", Particles: " << particleCount
            << ", Halos: " << haloCount << std::endl;
        out << "### Check ### Computational domain: " << d.bbox.xmin << " " << d.bbox.xmax << " " << d.bbox.ymin << " "
            << d.bbox.ymax << " " << d.bbox.zmin << " " << d.bbox.zmax << std::endl;
        out << "### Check ### Total Neighbors: " << totalNeighbors
            << ", Avg neighbor count per particle: " << totalNeighbors / d.n << std::endl;
        out << "### Check ### Total time: " << d.ttot << ", current time-step: " << d.minDt << std::endl;
        out << "### Check ### Total energy: " << d.etot << ", (internal: " << d.eint << ", cinetic: " << d.ecin;
#ifdef GRAVITY
        const std::streamsize defaultPrecision = std::cout.precision();
        const std::streamsize gravityPrecision = 15;
        out << ", gravitational: " << std::setprecision(gravityPrecision) << d.egrav
            << std::setprecision(defaultPrecision);
#endif
        out << ")" << std::endl;
    }

    void printTotalIterationTime(const float duration, std::ostream& out) const
    {
        out << "=== Total time for iteration(" << d.iteration << ") " << duration << "s" << std::endl << std::endl;
    }

    void printMemory(size_t halos, std::ostream& out) const
    {
        size_t mem = d.x.capacity();
#ifdef USE_MPI
        MPI_Allreduce(MPI_IN_PLACE, &mem, 1, MPI_UNSIGNED_LONG, MPI_MAX, MPI_COMM_WORLD);

        MPI_Allreduce(MPI_IN_PLACE, &halos, 1, MPI_UNSIGNED_LONG, MPI_MAX, MPI_COMM_WORLD);
#endif
        if (d.rank == 0)
        {
            out << "### Check ### Array sizes: " << mem << " Max halos: " << halos << std::endl;
        }
    }

private:
    const Dataset& d;
};

} // namespace sphexa
