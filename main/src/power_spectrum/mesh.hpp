#include <vector>
#include <limits>
#include "heffte.h"

template<typename T>
class Mesh
{
public:
    int rank_;
    int numRanks_;
    int gridDim_; // specifically integer because heffte library uses int
    int numShells_;

    heffte::box3d<> inbox_;
    std::vector<T>  velX_;
    std::vector<T>  velY_;
    std::vector<T>  velZ_;
    std::vector<T>  power_spectrum_;

    Mesh(int rank, int numRanks, int gridDim, int numShells)
        : rank_(rank)
        , numRanks_(numRanks)
        , gridDim_(gridDim)
        , numShells_(numShells)
        , inbox_(initInbox())
    {
        size_t inboxSize = static_cast<size_t>(inbox_.size[0]) * static_cast<size_t>(inbox_.size[1]) *
                           static_cast<size_t>(inbox_.size[2]);
        velX_.resize(inboxSize);
        velY_.resize(inboxSize);
        velZ_.resize(inboxSize);
        power_spectrum_.resize(numShells);
    }

    // Placeholder, needs to be implemented mapping cornerstone tree to the mesh
    void assign_velocities_to_mesh(T* xpos, T* ypos, T* zpos, T* vx, T* vy, T* vz, T* gridX, T* gridY, T* gridZ,
                                   size_t simDim, size_t gridDim)
    {
        size_t         simDim3 = simDim * simDim * simDim;
        std::vector<T> mesh(gridDim);

        T Lmin      = -0.5;
        T deltaMesh = 1.0 / (gridDim - 1);

        for (size_t i = 0; i < gridDim; i++)
        {
            mesh[i] = Lmin + i * deltaMesh;
        }

        for (size_t i = 0; i < gridDim; i++)
        {
            for (size_t j = 0; j < gridDim; j++)
            {
                for (size_t k = 0; k < gridDim; k++)
                {
                    T      min_distance = std::numeric_limits<T>::infinity();
                    int    min_index    = -1;
                    size_t gridIndex    = (i * gridDim + j) * gridDim + k;

                    for (size_t p = 0; p < simDim3; p++)
                    {
                        T xDistance = std::pow(xpos[p] - mesh[i], 2);
                        T yDistance = std::pow(ypos[p] - mesh[j], 2);
                        T zDistance = std::pow(zpos[p] - mesh[k], 2);
                        T distance  = xDistance + yDistance + zDistance;

                        if (distance < min_distance)
                        {
                            min_distance = distance;
                            min_index    = p;
                        }
                    }

                    gridX[gridIndex] = vx[min_index];
                    gridY[gridIndex] = vy[min_index];
                    gridZ[gridIndex] = vz[min_index];
                }
            }
        }
    }

    void calculate_power_spectrum()
    {
        calculate_fft();

        std::vector<T> freqVelo(velX_.size());

        // calculate the modulus of the velocity frequencies
        for (size_t i = 0; i < velX_.size(); i++)
        {
            freqVelo[i] = velX_[i] + velY_[i] + velZ_[i];
        }

        // perform spherical averaging
        perform_spherical_averaging(freqVelo.data());
    }

    void calculate_fft()
    {
        heffte::box3d<> outbox = inbox_;

        // change fftw depending on the configuration into cufft or rocmfft
        heffte::fft3d<heffte::backend::fftw> fft(inbox_, outbox, MPI_COMM_WORLD);

        std::vector<std::complex<T>> output(fft.size_outbox());

        fft.forward(velX_.data(), output.data());

        for (size_t i = 0; i < velX_.size(); i++)
        {
            velX_[i] = abs(output.at(i)) * abs(output.at(i));
        }

        fft.forward(velY_.data(), output.data());

        for (size_t i = 0; i < velY_.size(); i++)
        {
            velY_[i] = abs(output.at(i)) * abs(output.at(i));
        }

        fft.forward(velZ_.data(), output.data());

        for (size_t i = 0; i < velZ_.size(); i++)
        {
            velZ_[i] = abs(output.at(i)) * abs(output.at(i));
        }
    }

    // Implemented following numpy.fft.fftfreq
    void fftfreq(std::vector<T>& freq, int n, double dt)
    {
        if (n % 2 == 0)
        {
            for (int i = 0; i < n / 2; i++)
            {
                freq[i] = i / (n * dt);
            }
            for (int i = n / 2; i < n; i++)
            {
                freq[i] = (i - n) / (n * dt);
            }
        }
        else
        {
            for (int i = 0; i < (n - 1) / 2; i++)
            {
                freq[i] = i / (n * dt);
            }
            for (int i = (n - 1) / 2; i < n; i++)
            {
                freq[i] = (i - n + 1) / (n * dt);
            }
        }
    }

    // The normalized power spectrum results will be stored in rank 0
    void perform_spherical_averaging(T* ps)
    {
        std::vector<T> k_values(gridDim_);
        std::vector<T> k_1d(gridDim_);
        std::vector<T> ps_rad(numShells_);

        fftfreq(k_values, gridDim_, 1.0 / gridDim_);

        for (size_t i = 0; i < gridDim_; i++)
        {
            k_1d[i] = std::abs(k_values[i]);
        }

        // iterate over the ps array and assign the values to the correct radial bin
        for (int i = 0; i < inbox_.size[2]; i++) // slow heffte order
        {
            for (int j = 0; j < inbox_.size[1]; j++) // mid heffte order
            {
                for (int k = 0; k < inbox_.size[0]; k++) // fast heffte order
                {
                    size_t freq_index = (i * inbox_.size[1] + j) * inbox_.size[0] + k;

                    // Calculate the k indices with respect to the global mesh
                    size_t         k_index_i = i + inbox_.low[2];
                    size_t         k_index_j = j + inbox_.low[1];
                    size_t         k_index_k = k + inbox_.low[0];
                    T              kdist     = std::sqrt(k_values[k_index_i] * k_values[k_index_i] +
                                                         k_values[k_index_j] * k_values[k_index_j] +
                                                         k_values[k_index_k] * k_values[k_index_k]);
                    std::vector<T> k_dif(gridDim_);
                    for (size_t kind = 0; kind < gridDim_; kind++)
                    {
                        k_dif[kind] = std::abs(k_1d[kind] - kdist);
                    }
                    auto   it      = std::min_element(std::begin(k_dif), std::end(k_dif));
                    size_t k_index = std::distance(std::begin(k_dif), it);

                    ps_rad[k_index] += ps[freq_index];
                }
            }
        }

        MPI_Reduce(ps_rad.data(), power_spectrum_.data(), numShells_, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

        // normalize the power spectrum
        if (rank_ == 0)
        {
            T sum_ps_radial = std::accumulate(power_spectrum_.begin(), power_spectrum_.end(), 0.0);

            for (size_t i = 0; i < numShells_; i++)
            {
                power_spectrum_[i] = power_spectrum_[i] / sum_ps_radial;
            }
        }
    }

private:
    heffte::box3d<> initInbox()
    {
        heffte::box3d<> all_indexes({0, 0, 0}, {gridDim_ - 1, gridDim_ - 1, gridDim_ - 1});

        std::array<int, 3> proc_grid = heffte::proc_setup_min_surface(all_indexes, numRanks_);

        // split all indexes across the processor grid, defines a set of boxes
        std::vector<heffte::box3d<>> all_boxes = heffte::split_world(all_indexes, proc_grid);

        return all_boxes[rank_];
    }
};