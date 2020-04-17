#pragma once

#include <iomanip>

namespace sphexa
{

template <typename Dataset>
class Printer
{
public:
    Printer(const Dataset &d)
        : d(d)
    {
    }
    /*
        void printRadiusAndGravityForce(const std::vector<int> &clist, std::ostream &out)
        {
            out << std::sqrt(d.x[i] * d.x[i] + d.y[i] * d.y[i] + d.z[i] * d.z[i]) << ' ' << d.x[i] << ' ' << d.y[i] << ' ' << d.z[i] << ' '
                << d.fx[i] << ' ' << d.fy[i] << ' ' << d.fz[i] << std::endl;
        }
    */
    void printTree(GravityOctree<double> &octree, std::ostream &out) { octree.print(); }

    void printConstants(const int iteration, const int nntot, const size_t nnmax, const size_t ngmax, std::ostream &out) const
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
            out << nntot << ' ' << nnmax << ' ' << ngmax << ' ';
            out << d.ttot / d.tkh << ' ' << d.masscloud << ' ' << d.masscloud / d.masscloudinic << ' ';
            out << std::endl;
            out.flush();
        }
    }

    void printCheck(const size_t particleCount, const size_t nodeCount, const size_t haloCount, const size_t totalNeighbors,
                    const size_t maxNeighbors, const size_t ngmax, std::ostream &out) const
    {
        out << "### Check ### Global Tree Nodes: " << nodeCount << ", Particles: " << particleCount << ", Halos: " << haloCount
            << std::endl;
        out << "### Check ### Computational domain: " << d.bbox.xmin << " " << d.bbox.xmax << " " << d.bbox.ymin << " " << d.bbox.ymax
            << " " << d.bbox.zmin << " " << d.bbox.zmax << std::endl;
        out << "### Check ### Total Neighbors: " << totalNeighbors << ", Avg neighbor count per particle: " << totalNeighbors / d.n
            << ", Max neighbor count: " << maxNeighbors << " of " << ngmax << " supported neighbors" << std::endl;
        if (maxNeighbors >= ngmax)
            out << "### WARNING ### ONE OR MORE PARTICLES HAVE REACHED THE MAXIMUM SUPPORTED NUMBER OF NEIGHBORS! Check accuracy of results carefully!" << std::endl;
        out << "### Check ### Total time: " << d.ttot << ", current time-step: " << d.minDt << std::endl;
        out << "### Check ### Total energy: " << d.etot << ", (internal: " << d.eint << ", cinetic: " << d.ecin;
#ifdef GRAVITY
        const std::streamsize defaultPrecision = std::cout.precision();
        const std::streamsize gravityPrecision = 15;
        out << ", gravitational: " << std::setprecision(gravityPrecision) << d.egrav << std::setprecision(defaultPrecision);
#endif
        out << ")" << std::endl;
    }


    void printTotalIterationTime(const float duration, std::ostream &out) const
    {
        out << "=== Total time for iteration(" << d.iteration << ") " << duration << "s" << std::endl << std::endl;
    }

private:
    const Dataset &d;
};
} // namespace sphexa
