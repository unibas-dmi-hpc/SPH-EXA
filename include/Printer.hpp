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
            out << nntot << ' ' << d.ttot / d.tkh << ' ' << d.masscloud << ' ' << d.masscloud / d.masscloudinic << ' ';
            out << std::endl;
            out.flush();
        }
    }

    void printCheck(const size_t particleCount, const size_t nodeCount, const size_t haloCount, const size_t totalNeighbors,
                    const size_t maxNeighbors, std::ostream &out)
    {
        out << "### Check ### Global Tree Nodes: " << nodeCount << ", Particles: " << particleCount << ", Halos: " << haloCount
            << std::endl;
        out << "### Check ### Computational domain: " << d.bbox.xmin << " " << d.bbox.xmax << " " << d.bbox.ymin << " " << d.bbox.ymax
            << " " << d.bbox.zmin << " " << d.bbox.zmax << std::endl;
        out << "### Check ### Total Neighbors: " << totalNeighbors << ", Avg neighbor count per particle: " << totalNeighbors / d.n
            << ", Max neighbor count: " << maxNeighbors << std::endl;
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
                    // note: dt_m1[i] is already overwritten with the current dt[i] (in positions.hpp)!
                    // same for du_m1[i]
                    const double radius = std::sqrt(d.x[i] * d.x[i] + d.y[i] * d.y[i] + d.z[i] * d.z[i]);
                    dump << d.x[i] << ' ' << d.y[i] << ' ' << d.z[i] << ' ';
                    dump << d.vx[i] << ' ' << d.vy[i] << ' ' << d.vz[i] << ' ';
                    dump << d.h[i] << ' ' << d.ro[i] << ' ' << d.u[i] << ' ' << d.p[i] << ' ' << d.c[i] << ' ';
                    dump << ' ' << d.grad_P_x[i] << ' ' << d.grad_P_y[i] << ' ' << d.grad_P_z[i] << ' ';
                    dump << radius << ' ' << d.nn[i] << ' ' << d.sumkx[i] << ' ' << d.sumwh[i] << ' ';
                    dump << d.xmass[i] << ' ' << d.gradh[i] << ' ' << d.ballmass[i] << ' ';
#ifndef NDEBUG
                    dump << d.du[i] << ' ' << d.du_m1[i] << ' ' << d.du_av[i] << ' ' << d.du_av_m1[i] << ' ';
                    dump << d.dt[i] << ' ' << d.dt_m1[i] << ' ';
                    dump << d.maxvsignal[i] << ' ';
                    dump << d.c11[i] << ' ' << d.c12[i] << ' ' << d.c13[i] << ' ';
                    dump << d.c22[i] << ' ' << d.c23[i] << ' ';
                    dump << d.c33[i] << ' ';
                    dump << int(d.id[i]) << ' ';
                    dump << d.volnorm[i] << ' ';
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

    void printCheckpointToFile(const std::string &path)
    {
#ifdef USE_MPI
        printCheckpointToFileMPI(path);
        return;
#endif
        std::ofstream checkpoint;
        checkpoint.open(path, std::ofstream::out | std::ofstream::binary);

        if (checkpoint.is_open())
        {
            printf("Writing checkpoint at path: %s\n", path.c_str());

            checkpoint.write((char *)&d.n, sizeof(d.n));
            checkpoint.write((char *)&d.ttot, sizeof(d.ttot));
            checkpoint.write((char *)&d.minDt, sizeof(d.minDt));

            checkpoint.write((char *)&d.x[0], d.x.size() * sizeof(double));
            checkpoint.write((char *)&d.y[0], d.y.size() * sizeof(double));
            checkpoint.write((char *)&d.z[0], d.z.size() * sizeof(double));
            checkpoint.write((char *)&d.vx[0], d.vx.size() * sizeof(double));
            checkpoint.write((char *)&d.vy[0], d.vy.size() * sizeof(double));
            checkpoint.write((char *)&d.vz[0], d.vz.size() * sizeof(double));
            checkpoint.write((char *)&d.ro[0], d.ro.size() * sizeof(double));
            checkpoint.write((char *)&d.u[0], d.ro.size() * sizeof(double));
            checkpoint.write((char *)&d.p[0], d.p.size() * sizeof(double));
            checkpoint.write((char *)&d.h[0], d.h.size() * sizeof(double));
            checkpoint.write((char *)&d.m[0], d.h.size() * sizeof(double));

            checkpoint.write((char *)&d.temp[0], d.temp.size() * sizeof(double));
            checkpoint.write((char *)&d.mue[0], d.mue.size() * sizeof(double));
            checkpoint.write((char *)&d.mui[0], d.mui.size() * sizeof(double));

            checkpoint.write((char *)&d.du[0], d.du.size() * sizeof(double));
            checkpoint.write((char *)&d.du_m1[0], d.du_m1.size() * sizeof(double));
            checkpoint.write((char *)&d.dt[0], d.dt.size() * sizeof(double));
            checkpoint.write((char *)&d.dt_m1[0], d.dt_m1.size() * sizeof(double));

            checkpoint.write((char *)&d.x_m1[0], d.x_m1.size() * sizeof(double));
            checkpoint.write((char *)&d.y_m1[0], d.y_m1.size() * sizeof(double));
            checkpoint.write((char *)&d.z_m1[0], d.z_m1.size() * sizeof(double));

            checkpoint.close();
        }
        else
        {
            printf("Error: Can't open file to save checkpoint. Path: %s\n", path.c_str());
        }
    }

#ifdef USE_MPI
    void printCheckpointToFileMPI(const std::string &path)
    {
        MPI_File file;
        MPI_Status status;

        int err = MPI_File_open(d.comm, path.c_str(), MPI_MODE_CREATE | MPI_MODE_RDWR, MPI_INFO_NULL, &file);
        if (err != MPI_SUCCESS)
        {
            if (d.rank == 0) printf("Error %d! Can't open the file to write checkpoint. Path: %s\n", err, path.c_str());
            MPI_Abort(d.comm, err);
            exit(EXIT_FAILURE);
        }

        const size_t split = d.n / d.nrank;
        const size_t remaining = d.n - d.nrank * split;

        const MPI_Offset col = d.n * sizeof(double);
        const MPI_Offset headerOffset = 2 * sizeof(double) + sizeof(size_t);
        MPI_Offset offset = headerOffset + d.rank * split * sizeof(double);
        if (d.rank > 0) offset += remaining * sizeof(double);

        if (d.rank == 0)
        {
            MPI_File_write(file, &d.n, 1, MPI_UNSIGNED_LONG, &status);
            MPI_File_write(file, &d.ttot, 1, MPI_DOUBLE, &status);
            MPI_File_write(file, &d.minDt, 1, MPI_DOUBLE, &status);
        }
        MPI_Barrier(d.comm);

        writeParticleDataToBinFileWithMPI(file, d.count, offset, col, 0, d.x, d.y, d.z, d.vx, d.vy, d.vz, d.ro, d.u, d.p, d.h, d.m, d.temp,
                                          d.mue, d.mui, d.du, d.du_m1, d.dt, d.dt_m1, d.x_m1, d.y_m1, d.z_m1);

        MPI_File_close(&file);

        if (d.rank == 0) printf("MPI Checkpoint file saved!\n");
    }

#endif

    void printTotalIterationTime(const float duration, std::ostream &out)
    {
        out << "=== Total time for iteration(" << d.iteration << ") " << duration << "s" << std::endl << std::endl;
    }

private:
    const Dataset &d;
};
} // namespace sphexa
