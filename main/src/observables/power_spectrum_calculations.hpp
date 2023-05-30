#include <vector>
#include <complex>
#include <iostream>
#include <cmath>
#include <chrono>
// #include "include/heffte.h"

namespace sphexa
{

double cubickernel(double r, double h)
{
    double       u;
    const double pi = 3.141592653589793;
    double       W;
    u = r / h;
    if (u >= 0.0 && u <= 1.0) { W = 1.0 / (pi * h * h * h) * (1 - 1.5e0 * u * u * (1.0e0 - 0.5e0 * u)); }
    else if (u > 1.0e0 && u < 2.0e0) { W = 1.0 / (pi * h * h * h) * 0.25e0 * (2.0e0 - u) * (2.0e0 - u) * (2.0e0 - u); }
    else { W = 0.0; }

    return W;
}

template<typename T>
class GriddedDomain
{
public:
    GriddedDomain(size_t domainSize, T Lb)
        : Lbox(Lb)
        , domainSize_(domainSize)
        , gridSize_(domainSize * 8)
        , gridDim_(std::cbrt(domainSize) * 2)
    {
        mass = 1. / domainSize_;

        // Gix.resize(gridSize_);
        // Giy.resize(gridSize_);
        // Giz.resize(gridSize_);
        GlobalGv.resize(gridSize_);
        Gv.resize(gridSize_);
        norm.resize(gridSize_);
        GlobalNorm.resize(gridSize_);

        calculatePixelCenters();
    }

    void rasterizeDomain(const T* xpos, const T* ypos, const T* zpos, const T* v, const T* rho, const T* h,
                         size_t numLocalParticles)
    {
        size_t counts[numLocalParticles];
        size_t counts_reduced = 0;
        std::fill_n(counts, numLocalParticles, 0);
        std::cout << "domainSize = " << domainSize_ << ", numLocalParticles = " << numLocalParticles
                  << ", npixels = " << gridDim_ << std::endl;

#pragma omp parallel for
        for (size_t n = 0; n < numLocalParticles; n++)
        {
            // double h1     = h[n];
            // double h2     = 2 * h1;
            // double h2_2   = h2 * h2;
            // double h3     = h1 * h1 * h1;
            // double weight = mass / rho[n] / h3;
            double h1     = 1.003 * std::cbrt(3. / 4. / M_PI * partperpixel / domainSize_) * Lbox / 2;
            double h2     = 2 * h1;
            double h2_2   = h2 * h2;
            double h3     = h1 * h1 * h1;
            double weight = mass / 1.0; // ro[n] / h3;

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
                            uint64_t iii = xindex + gridDim_ * (yindex + gridDim_ * zindex);
                            double   W   = weight * sphexa::cubickernel(r, h1);
                            Gv[iii]      = Gv[iii] + W * v[n];
                            norm[iii]    = norm[iii] + W;
                            counts[n]++;
                            // counts_reduced++;
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

        // Need to Gather from all ranks.
        MPI_Reduce(Gv.data(), GlobalGv.data(), gridSize_, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(norm.data(), GlobalNorm.data(), gridSize_, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

        // normalizeGrid();
        // printGrid();
    }

    void printGrid()
    {
        std::string   v_filename = "Gv_sphexa_50.txt";
        std::ofstream gridfile;
        gridfile.open(v_filename, std::ios::trunc);
        for (size_t i = 0; i < gridSize_; i++)
        {
            gridfile << std::setprecision(8) << std::scientific << GlobalGv[i] << std::endl;
        }
        gridfile.close();

        v_filename = "norm_sphexa_50.txt";
        std::ofstream normfile;
        normfile.open(v_filename, std::ios::trunc);
        for (size_t i = 0; i < gridSize_; i++)
        {
            normfile << std::setprecision(8) << std::scientific << GlobalNorm[i] << std::endl;
        }
        normfile.close();
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

    void normalizeGrid()
    {
        double average = 0.0;
#pragma omp parallel for reduction(+ : average)
        for (size_t i = 0; i < gridSize_; i++)
        {
            if (GlobalNorm[i] == 0)
            {
                // std::cout << "Failed Rasterization in iteration = " //<< i << ", j = " << j << ", k = " << k
                //           << std::endl;
                // std::terminate();
                GlobalGv[i] = 0.0;
            }
            else
            {
                GlobalGv[i] = GlobalGv[i] / GlobalNorm[i];
                average += GlobalGv[i] * GlobalGv[i];
            }
        }
        std::cout << "root mean square: " << sqrt((average / gridSize_)) << std::endl;
    }

    double partperpixel = 8.; // Doesn't work approximately lower than 6
    float  mass;
    size_t gridSize_;
    size_t gridDim_;
    size_t domainSize_;
    T      Lbox;

    // Grid indices
    // std::vector<T> Gix, Giy, Giz;

    // Gridded velocity
    std::vector<T> Gv;
    // Gridded velocity
    std::vector<T> GlobalGv;
    // pixel norms
    std::vector<T> norm;
    // pixel norms
    std::vector<T> GlobalNorm;

    // Pixel centers
    std::vector<T> D;
};

void shells(double w[], int npixels, double E[], double k_center[])
{
    int halfnpixels = std::floor(npixels / 2);
    int Kmax        = std::ceil(std::sqrt(3.0) * (0.5 * npixels));
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

void fft3D(double G_1D[], int npixels)
{
    // uint64_t        npixels3 = npixels * npixels * npixels;
    // heffte::box3d<> inbox    = {{0, 0, 0}, {npixels - 1, npixels - 1, npixels - 1}};
    // heffte::box3d<> outbox   = {{0, 0, 0}, {npixels - 1, npixels - 1, npixels - 1}};

    // // define the heffte class and the input and output geometry
    // heffte::fft3d<heffte::backend::fftw> fft(inbox, outbox, MPI_COMM_WORLD);

    // // vectors with the correct sizes to store the input and output data
    // // taking the size of the input and output boxes
    // std::vector<double>               input(fft.size_inbox());
    // std::vector<std::complex<double>> output(fft.size_outbox());

    // // fill the input vector with data that looks like 0, 1, 2, ...
    // // std::iota(input.begin(), input.end(), 0); // put some data in the input
    // for (uint64_t i = 0; i < npixels * npixels * npixels; i++)
    // {
    //     input.at(i) = G_1D[i];
    // }

    // // perform a forward DFT
    // fft.forward(input.data(), output.data());

    // for (uint64_t i = 0; i < npixels * npixels * npixels; i++)
    // {
    //     G_1D[i] = abs(output.at(i));
    // }
}

} // namespace sphexa