#pragma once

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

        void printTree(GravityOctree<double> &octree, std::ostream &out) { octree.print(out); }
    */
    void printConstants(const int iteration, const int nntot, std::ostream &out)
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

    void printCheck(const size_t particleCount, const size_t nodeCount, const size_t haloCount, const size_t totalNeighbors,
                    std::ostream &out)
    {
        out << "### Check ### Global Tree Nodes: " << nodeCount << ", Particles: " << particleCount << ", Halos: " << haloCount
            << std::endl;
        out << "### Check ### Computational domain: " << d.bbox.xmin << " " << d.bbox.xmax << " " << d.bbox.ymin << " " << d.bbox.ymax
            << " " << d.bbox.zmin << " " << d.bbox.zmax << std::endl;
        out << "### Check ### Total Neighbors: " << totalNeighbors << ", Avg neighbor count per particle: " << totalNeighbors / d.n
            << std::endl;
        out << "### Check ### Total time: " << d.ttot << ", current time-step: " << d.minDt << std::endl;
        out << "### Check ### Total energy: " << d.etot << ", (internal: " << d.eint << ", cinetic: " << d.ecin;
#ifdef GRAVITY
        out << ", gravitational: " << d.egrav;
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
                    const double radius = std::sqrt(d.x[i] * d.x[i] + d.y[i] * d.y[i] + d.z[i] * d.z[i]);
                    dump << d.x[i] << ' ' << d.y[i] << ' ' << d.z[i] << ' ';
                    dump << d.vx[i] << ' ' << d.vy[i] << ' ' << d.vz[i] << ' ';
                    dump << d.h[i] << ' ' << d.ro[i] << ' ' << d.u[i] << ' ' << d.p[i] << ' ' << d.c[i] << ' ';
                    dump << ' ' << d.grad_P_x[i] << ' ' << d.grad_P_y[i] << ' ' << d.grad_P_z[i] << ' ';
                    dump << radius << ' ';
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

    void printCheckpointToFile(const std::vector<int> &clist, const std::string &dumpfilename)
    {
        std::ofstream dump;

        dump.open(dumpfilename, std::ofstream::out | std::ofstream::binary);
        dump.write((char *)&d.n, sizeof(d.n));
        dump.write((char *)&d.ttot, sizeof(d.ttot));
        dump.write((char *)&d.minDt, sizeof(d.minDt));

        for (int i = 0; i < 5; ++i)
        {
            printf("[%d] write x=%f, y=%f\n", i, d.x[i], d.y[i]);
        }

        dump.write((char *)&d.x[0], d.x.size() * sizeof(double));
        dump.write((char *)&d.y[0], d.y.size() * sizeof(double));
        dump.write((char *)&d.z[0], d.z.size() * sizeof(double));
        dump.write((char *)&d.vx[0], d.vx.size() * sizeof(double));
        dump.write((char *)&d.vy[0], d.vy.size() * sizeof(double));
        dump.write((char *)&d.vz[0], d.vz.size() * sizeof(double));
        dump.write((char *)&d.ro[0], d.ro.size() * sizeof(double));
        dump.write((char *)&d.u[0], d.ro.size() * sizeof(double));
        dump.write((char *)&d.p[0], d.p.size() * sizeof(double));
        dump.write((char *)&d.h[0], d.h.size() * sizeof(double));
        dump.write((char *)&d.m[0], d.h.size() * sizeof(double));

        dump.write((char *)&d.temp[0], d.temp.size() * sizeof(double));
        dump.write((char *)&d.mue[0], d.mue.size() * sizeof(double));
        dump.write((char *)&d.mui[0], d.mui.size() * sizeof(double));

        dump.write((char *)&d.du[0], d.du.size() * sizeof(double));
        dump.write((char *)&d.du_m1[0], d.du_m1.size() * sizeof(double));
        dump.write((char *)&d.dt[0], d.dt.size() * sizeof(double));
        dump.write((char *)&d.dt_m1[0], d.dt_m1.size() * sizeof(double));

        dump.write((char *)&d.x_m1[0], d.x_m1.size() * sizeof(double));
        dump.write((char *)&d.y_m1[0], d.y_m1.size() * sizeof(double));
        dump.write((char *)&d.z_m1[0], d.z_m1.size() * sizeof(double));


        dump.close();

    }

    void printTotalIterationTime(const float duration, std::ostream &out)
    {
        out << "=== Total time for iteration(" << d.iteration << ") " << duration << "s" << std::endl << std::endl;
    }

private:
    const Dataset &d;
};
} // namespace sphexa
