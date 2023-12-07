#include <vector>
#include <complex>
#include <iostream>
#include <cmath>
#include <chrono>
#include <mpi.h>
#include <map>
#include <tuple>

#ifdef HEFFTE_ENABLED
#include "heffte.h"
#endif

namespace sphexa
{

double cubickernel(double r, double h)
{
    double u;
    double W;
    u = r / h;
    if (u >= 0.0 && u <= 1.0) { W = 1.0 / (M_PI * h * h * h) * (1 - 1.5e0 * u * u * (1.0e0 - 0.5e0 * u)); }
    else if (u > 1.0e0 && u < 2.0e0)
    {
        W = 1.0 / (M_PI * h * h * h) * 0.25e0 * (2.0e0 - u) * (2.0e0 - u) * (2.0e0 - u);
    }
    else { W = 0.0; }

    return W;
}

template<typename T>
class GriddedDomain
{
    // The key of the map is a tuple of <rank, index>
    typedef std::tuple<int, uint64_t> mapKey;

public:
    GriddedDomain(size_t domainSize, T Lb)
        : Lbox(Lb)
        , domainSize_(domainSize)
        , gridSize_(domainSize * 8)
        , gridDim_(std::cbrt(domainSize) * 2)
        , all_indexes({0, 0, 0}, {gridDim_ - 1, gridDim_ - 1, gridDim_ - 1})
        , inbox(getInbox())
    {
        boxSize_ = inbox.size[0] * inbox.size[1] * inbox.size[2];

        Gv.resize(boxSize_);
        norm.resize(boxSize_);

        std::cout << "size = " << Gv.size() << std::endl;

        calculatePixelCenters();
    }

    heffte::box3d<> getInbox()
    {
        int num_ranks; // total number of ranks in the comm

        MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
        MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

        std::array<int, 3> proc_grid = heffte::proc_setup_min_surface(all_indexes, num_ranks);

        // split all indexes across the processor grid, defines a set of boxes
        std::vector<heffte::box3d<>> all_boxes = heffte::split_world(all_indexes, proc_grid);

        return all_boxes[rank_];
    }

    void calculatePowerSpectrum()
    {
        int num_ranks; // total number of ranks in the comm

        MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

        heffte::box3d<> const outbox = inbox; // all_boxes[me]; // same inbox and outbox

        // at this stage we can manually adjust some HeFFTe options
        heffte::plan_options options = heffte::default_options<heffte::backend::fftw>();

        // define the heffte class and the input and output geometry
        heffte::fft3d<heffte::backend::fftw> fft(inbox, outbox, MPI_COMM_WORLD, options);

        // vectors with the correct sizes to store the input and output data
        // taking the size of the input and output boxes
        std::vector<T>               input(fft.size_inbox());
        std::vector<std::complex<T>> output(fft.size_outbox());

        // check the size of the vector
        for (uint64_t i = 0; i < Gv.size(); i++)
        {
            input.at(i) = Gv[i];
        }

        // // perform a forward DFT
        fft.forward(input.data(), output.data());

        // check the size of the vector
        for (uint64_t i = 0; i < Gv.size(); i++)
        {
            Gv[i] = abs(output.at(i));
        }
    }

    void rasterizeDomain(const T* xpos, const T* ypos, const T* zpos, const T* v, const T* rho, const T* h,
                         size_t numLocalParticles)
    {
        size_t counts[numLocalParticles];
        size_t counts_reduced = 0;
        std::fill_n(counts, numLocalParticles, 0);
        // std::cout << "domainSize = " << domainSize_ << ", numLocalParticles = " << numLocalParticles
        //           << ", npixels = " << gridDim_ << std::endl;
        std::cout << "in.size[0] = " << inbox.size[0] << ", in.size[1] = " << inbox.size[1]
                  << ", in.size[2] = " << inbox.size[2] << std::endl;
        int num_ranks; // total number of ranks in the comm

        MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

#pragma omp parallel for schedule(static)
        for (size_t n = 0; n < numLocalParticles; n++)
        {
            // double h1     = h[n];
            double h1   = 1.003 * std::cbrt(3. / 4. / M_PI * partperpixel / domainSize_) * Lbox / 2;
            double h2   = 2 * h1;
            double h2_2 = h2 * h2;
            double h3   = h1 * h1 * h1;
            double mass = 1. / domainSize_;
            // double weight = mass / rho[n] / h3;
            double weight = mass / rho[n] / h3; // 1.0;

            int max_intz = std::floor((zpos[n] + h2) * gridDim_ - 0.5e0);
            int min_intz = std::ceil((zpos[n] - h2) * gridDim_ - 0.5e0);

            // In here normally we check whether the indices are inside our target grid range.
            // For now, it uses the whole pixel grid, but later it needs to depend on each
            // rank's target grid range.
            for (int i = min_intz; i <= max_intz; i++)
            {
                int zindex;
                if (i < 0) { zindex = i + gridDim_; }
                else if (i >= gridDim_) { zindex = i - gridDim_; }
                else { zindex = i; }
                double z = std::min(std::abs(zpos[n] - D[zindex]), std::abs(zpos[n] + Lbox - D[zindex]));
                z        = z * z;
                if (z > h2_2) continue;

                double s1_2     = h2_2 - z;
                double s1       = std::sqrt(s1_2);
                int    min_intx = std::ceil((xpos[n] - s1) * gridDim_ - 0.5e0);
                int    max_intx = std::floor((xpos[n] + s1) * gridDim_ - 0.5e0);

                for (int j = min_intx; j <= max_intx; j++)
                {
                    int xindex;
                    if (j < 0) { xindex = j + gridDim_; }
                    else if (j >= gridDim_) { xindex = j - gridDim_; }
                    else { xindex = j; }
                    double x = std::min(std::abs(xpos[n] - D[xindex]), std::abs(xpos[n] + Lbox - D[xindex]));
                    x        = x * x;
                    if (x > s1_2) continue;

                    double s2       = std::sqrt(s1_2 - x);
                    int    min_inty = std::ceil((ypos[n] - s2) * gridDim_ - 0.5e0);
                    int    max_inty = std::floor((ypos[n] + s2) * gridDim_ - 0.5e0);

                    for (int k = min_inty; k <= max_inty; k++)
                    {
                        int yindex;
                        if (k < 0) { yindex = k + gridDim_; }
                        else if (k >= gridDim_) { yindex = k - gridDim_; }
                        else { yindex = k; }

                        double y = std::min(std::abs(ypos[n] - D[yindex]), std::abs(ypos[n] + Lbox - D[yindex]));
                        double r = std::sqrt(x + y * y + z);

                        if (r < h2)
                        {
                            // Calculate the indices according to the heffte boxes
                            int iii = (xindex % inbox.size[0]) +
                                      gridDim_ * (yindex % inbox.size[1] + gridDim_ * (zindex % inbox.size[2]));
                            // std::cout << "x, y, z = " << xindex << ", " << yindex << ", " << zindex << std::endl;

                            double W = weight * sphexa::cubickernel(r, h1);
                            if (isIndexThisRanks(xindex, yindex, zindex))
                            {
                                Gv[iii]   = Gv[iii] + W * v[n];
                                norm[iii] = norm[iii] + W;
                                counts[n]++;
                            }
                            else
                            {
                                int indexRank = getRankOfIndex(xindex, yindex, zindex);
                                if (indexRank >= num_ranks) std::terminate();
                                auto key = std::make_tuple(indexRank, iii);
                                // std::cout << "Rank: " << rank_ << "indexRank: " << indexRank << std::endl;

                                if (auto search = map_Gv.find(key); search != map_Gv.end())
                                    map_Gv[key] = map_Gv[key] + W * v[n];
                                else
                                    map_Gv.emplace(key, W * v[n]);

                                if (auto search = map_norm.find(key); search != map_norm.end())
                                    map_norm[key] = map_norm[key] + W;
                                else
                                    map_norm.emplace(key, W);
                                // handle counts in communication
                            }
                        }
                    }
                }
            }
        }

#pragma omp parallel for reduction(+ : counts_reduced)
        for (size_t i = 0; i < numLocalParticles; i++)
        {
            counts_reduced += counts[i];
        }
        std::cout << "counts: " << counts_reduced << std::endl;
        std::cout << "number of particles contributing to a pixel: " << counts_reduced * 1.0 / gridSize_ << std::endl;

        std::cout << "map_Gv.size(): " << map_Gv.size() << std::endl;
        std::cout << "map_norm.size(): " << map_norm.size() << std::endl;

        exchangeDataMap();
        // normalizeGrid();
        printGridVector();
        // calculatePowerSpectrum();
    }

    void exchangeDataMap()
    {
        if (map_Gv.empty()) return;
        std::cout << "exchange map" << std::endl;

        std::vector<int> indices;
        std::vector<T>   values;
        std::vector<int> rank_counts;
        std::vector<int> displacements;

        int num_ranks; // total number of ranks in the comm

        MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

        rank_counts.resize(num_ranks);

        for (const auto& [key, value] : map_Gv)
        {
            indices.push_back(std::get<1>(key));
            values.push_back(value);
            rank_counts[std::get<0>(key)]++;
        }
        std::cout << "calc displacements" << std::endl;
        displacements.push_back(0);
        for (int i = 1; i < rank_counts.size(); i++)
        {
            displacements.push_back(displacements[i - 1] + rank_counts[i - 1]);
        }

        // mpi_alltoallv to exchange data
        std::vector<int> indices_recv;
        std::vector<T>   values_recv;
        std::vector<int> rank_counts_recv;
        std::vector<int> displacements_recv;
        indices_recv.resize(indices.size());
        values_recv.resize(values.size());
        rank_counts_recv.resize(rank_counts.size());
        displacements_recv.resize(rank_counts.size());

        for (int i = 0; i < rank_counts.size(); i++)
        {
            std::cout << "RANK = " << rank_ << ", " << rank_counts[i] << " ";
        }

        std::cout << "mpi_alltoall start" << std::endl;
        MPI_Alltoall(rank_counts.data(), 1, MPI_INT, rank_counts_recv.data(), 1, MPI_INT, MPI_COMM_WORLD);

        std::cout << "calc displacements" << std::endl;
        displacements_recv[0] = 0;
        for (int i = 1; i < rank_counts_recv.size(); i++)
        {
            displacements_recv[i] = displacements_recv[i - 1] + rank_counts_recv[i - 1];
        }

        std::cout << "mpi_alltoallv start" << std::endl;
        MPI_Alltoallv(indices.data(), rank_counts.data(), displacements.data(), MPI_INT, indices_recv.data(),
                      rank_counts_recv.data(), displacements_recv.data(), MPI_INT, MPI_COMM_WORLD);
        // MPI_Alltoallv(values.data(), rank_counts.data(), rank_counts.data(), MPI_DOUBLE, values_recv.data(),
        //               rank_counts_recv.data(), rank_counts_recv.data(), MPI_DOUBLE, MPI_COMM_WORLD);

        for (int i = 0; i < rank_counts_recv.size(); i++)
        {
            std::cout << "RANK = " << rank_ << ", " << rank_counts_recv[i] << " ";
        }

        std::cout << "mpi_alltoall finish" << std::endl;

        // put the values_recv in the Gv using indices_recv as indices
        for (size_t i = 0; i < indices_recv.size(); i++)
        {
            Gv[indices_recv[i]] = Gv[indices_recv[i]] + values_recv[i];
        }

        // Do the same thing for norm data using the same buffers
        // for (const auto& [key, value] : map_norm)
        // {
        //     indices.push_back(std::get<1>(key));
        //     values.push_back(value);
        // }

        // MPI_Alltoallv(indices.data(), rank_counts.data(), rank_counts.data(), MPI_UNSIGNED_LONG, indices_recv.data(),
        //               rank_counts_recv.data(), rank_counts_recv.data(), MPI_UNSIGNED_LONG, MPI_COMM_WORLD);
        // MPI_Alltoallv(values.data(), rank_counts.data(), rank_counts.data(), MPI_DOUBLE, values_recv.data(),
        //               rank_counts_recv.data(), rank_counts_recv.data(), MPI_DOUBLE, MPI_COMM_WORLD);

        // for (size_t i = 0; i < indices_recv.size(); i++)
        // {
        //     norm[indices_recv[i]] = norm[indices_recv[i]] + values_recv[i];
        // }
    }

    void printGridVector()
    {
        int num_ranks; // total number of ranks in the comm
        MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

        std::string   v_filename = "Gv_sphexa_50.txt";
        std::ofstream gridfile;

        for (int turn = 0; turn < num_ranks; turn++)
        {
            if (rank_ == turn)
            {
                gridfile.open(v_filename, std::ios::trunc);
                for (size_t i = 0; i < Gv.size(); i++)
                {
                    gridfile << std::setprecision(8) << std::scientific << Gv[i] << std::endl;
                }
                gridfile.close();
            }
        }

        v_filename = "norm_sphexa_50.txt";
        std::ofstream normfile;

        for (int turn = 0; turn < num_ranks; turn++)
        {
            if (rank_ == turn)
            {
                normfile.open(v_filename, std::ios::trunc);
                for (size_t i = 0; i < norm.size(); i++)
                {
                    normfile << std::setprecision(8) << std::scientific << norm[i] << std::endl;
                }
                normfile.close();
            }
        }
    }

private:
    void calculatePixelCenters()
    {
        D.resize(gridDim_);

        T dx  = Lbox / gridDim_;
        T dx2 = dx / 2;

#pragma omp parallel for
        for (size_t i = 0; i < gridDim_; i++)
        {
            D[i] = (2 * i + 1) * dx2;
        }
    }

    // return true if the index belongs to this rank's box
    bool isIndexThisRanks(int x, int y, int z)
    {
        bool inX = (x >= inbox.low[0] && x <= inbox.high[0]);
        bool inY = (y >= inbox.low[1] && y <= inbox.high[1]);
        bool inZ = (z >= inbox.low[2] && z <= inbox.high[2]);

        return inX && inY && inZ;
    }

    // Assume a power of two number of ranks
    int getRankOfIndex(int x, int y, int z)
    {
        int inX = x / inbox.size[0];
        int inY = y / inbox.size[1];
        int inZ = z / inbox.size[2];

        // Mapping of dimensions to ranks is as follows:
        // R0 -> 0,0,0; R1 -> 0,1,0; R2 -> 0,0,1; R3 -> 0,1,1;
        // R4 -> 1,0,0; R5 -> 1,1,0 and so on.
        // return inX * 4 + inZ * 2 + inY;
        return inX * 4 + inY * 2 + inZ;
    }

    void normalizeGrid()
    {
        double average = 0.0;
        // #pragma omp parallel for reduction(+ : average)
        for (size_t i = 0; i < boxSize_; i++)
        {
            if (norm[i] == 0)
            {
                // Normally we should not get here
                std::cout << "Failed Rasterization in iteration = " << i //", j = " << j << ", k = " << k
                          << std::endl;
                // std::terminate();
                Gv[i] = 0.0;
            }
            else
            {
                Gv[i] = Gv[i] / norm[i];
                average += Gv[i] * Gv[i];
            }
        }
        // printed per rank
        std::cout << "root mean square: " << sqrt((average / boxSize_)) << std::endl;
    }

    void averaging()
    {
        int    Kmax = std::ceil(std::sqrt(3.0) * (0.5 * gridDim_));
        double k_center[Kmax];
        E.resize(Kmax);
        counts.resize(Kmax);

        for (int i = 0; i < Kmax; i++)
        {
            E[i]      = 0;
            counts[i] = 0;
        }
    }

    double          partperpixel = 8.; // Doesn't work approximately lower than 6
    size_t          gridSize_;
    int             gridDim_;
    size_t          domainSize_;
    T               Lbox;
    int             rank_;
    int             boxSize_;
    heffte::box3d<> all_indexes;
    heffte::box3d<> inbox;

    // Gridded velocity map. Key is a tuple of <rank, index>
    std::map<mapKey, T> map_Gv;
    // pixel norms map. Key is a tuple of <rank, index>
    std::map<mapKey, T> map_norm;

    // Gridded velocity
    std::vector<T> Gv;
    // Pixel norms
    std::vector<T> norm;

    // Averages
    std::vector<T> E;
    std::vector<T> counts;

    // Pixel centers
    std::vector<T> D;
};

// E is the output array, w is the gridded values, npixels is the number of pixels in each dimension
void shells(double w[], int npixels, double E[], double k_center[])
{
    int halfnpixels = std::floor(npixels * 0.5);
    int Kmax        = std::ceil(std::sqrt(3.0) * (halfnpixels));
    // double E[Kmax],k_center[Kmax];
    int      Kx, Ky, Kz, K;
    uint64_t iii;
    double   y_N3 = 1. / (npixels * npixels * npixels);
    int      counts[Kmax];

    for (int i = 0; i < Kmax; i++)
    {
        E[i]        = 0;
        counts[i]   = 0;
        k_center[i] = std::cbrt(0.5 * ((i + 1) * (i + 1) * (i + 1) + i * i * i));
    }

    for (int i = 0; i < npixels; i++)
    {
        Kx = i;
        if (i > halfnpixels) { Kx = npixels - i; }
        // std::cout << i << " of " << npixels << std::endl;
        for (int j = 0; j < npixels; j++)
        {
            Ky = j;
            if (j > halfnpixels) { Ky = npixels - j; }
            for (int k = 0; k < npixels; k++)
            {
                Kz = k;
                if (k > halfnpixels) { Kz = npixels - k; }
                iii = i + npixels * (j + npixels * k);
                K   = std::floor(std::sqrt(Kx * Kx + Ky * Ky + Kz * Kz));

                // std::cout << K << std::endl; getchar();
                E[K] = E[K] + w[iii] * w[iii];

                counts[K]++;
            }
        }
    }

    for (int i = 0; i < Kmax; i++)
    {
        E[i] = 4 * M_PI * y_N3 * k_center[i] * k_center[i] * E[i] / counts[i];
    }
}

} // namespace sphexa