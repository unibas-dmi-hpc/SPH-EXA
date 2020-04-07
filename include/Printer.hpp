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
        if (maxNeighbors == ngmax)
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

    void printAllDataToFile(const std::vector<int> &clist, const std::string &dumpfilename)
    {
        for (int turn = 0; turn < d.nrank; turn++)
        {
            if (turn == d.rank)
            {
                std::ofstream dump;

                if (d.rank == 0)
                    dump.open(dumpfilename);
                else
                    dump.open(dumpfilename, std::ios_base::app);

                for (unsigned int pi = 0; pi < clist.size(); pi++)
                {
                    const int i = clist[pi];
                    // const int nn = d.neighborsCount[pi];
                    // note: dt_m1[i] is already overwritten with the current dt[i] (in positions.hpp)!
                    // same for du_m1[i]
                    const double radius = std::sqrt(d.x[i] * d.x[i] + d.y[i] * d.y[i] + d.z[i] * d.z[i]);
                    dump << d.x[i] << ' ' << d.y[i] << ' ' << d.z[i] << ' ';
                    dump << d.vx[i] << ' ' << d.vy[i] << ' ' << d.vz[i] << ' ';
                    dump << d.h[i] << ' ' << d.ro[i] << ' ' << d.u[i] << ' ' << d.p[i] << ' ' << d.c[i] << ' ';
                    dump << ' ' << d.grad_P_x[i] << ' ' << d.grad_P_y[i] << ' ' << d.grad_P_z[i] << ' ';
                    dump << radius << ' ' << d.nn[i] << ' ' << d.sumkx[i] << ' ' << d.sumwh[i] << ' ';
                    dump << d.xmass[i] << ' ' << d.gradh[i] << ' ' << d.ballmass[i] << ' ';
                    dump << d.volnorm[i] << ' ';
#ifndef NDEBUG
                    dump << d.du[i] << ' ' << d.du_m1[i] << ' ' << d.du_av[i] << ' ' << d.du_av_m1[i] << ' ';
                    dump << d.dt[i] << ' ' << d.dt_m1[i] << ' ';
                    dump << d.maxvsignal[i] << ' ';
                    dump << d.c11[i] << ' ' << d.c12[i] << ' ' << d.c13[i] << ' ';
                    dump << d.c22[i] << ' ' << d.c23[i] << ' ';
                    dump << d.c33[i] << ' ';
                    dump << int(d.id[i]) << ' ';
#endif
#ifdef GRAVITY
                    dump << d.fx[i] << ' ' << d.fy[i] << ' ' << d.fz[i] << ' ' << d.ugrav[i] << ' ';
#endif
                    // T rad = sqrt(x[i] * x[i] + y[i] * y[i] + z[i] * z[i]);
                    // T vrad = (vx[i] * x[i] + vy[i] * y[i] + vz[i] * z[i]) / rad;
                    // dump << rad << ' ' << vrad << std::endl;
                    dump << d.rank << std::endl;
                }

                dump.close();

#ifdef USE_MPI
                MPI_Barrier(MPI_COMM_WORLD);
#endif
            }
            else
            {
#ifdef USE_MPI
                MPI_Barrier(MPI_COMM_WORLD);
#endif
            }
        }
    }


    void printTotalIterationTime(const float duration, std::ostream &out) const
    {
        out << "=== Total time for iteration(" << d.iteration << ") " << duration << "s" << std::endl << std::endl;
    }

private:
    const Dataset &d;
};
} // namespace sphexa
