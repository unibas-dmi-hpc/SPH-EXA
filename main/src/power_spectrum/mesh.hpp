#include <vector>
#include <limits>
#include "heffte.h"

template<typename T>
class Mesh
{
public:
    int             rank_;
    int             numRanks_;
    int             gridDim_; // specifically integet because heffte library uses int
    heffte::box3d<> inbox_;
    std::vector<T>  velX_;
    std::vector<T>  velY_;
    std::vector<T>  velZ_;

    Mesh(int rank, int numRanks, int gridDim)
        : rank_(rank)
        , numRanks_(numRanks)
        , gridDim_(gridDim)
        , inbox_(initInbox())
    {
        size_t inboxSize = static_cast<size_t>(inbox_.size[0]) * static_cast<size_t>(inbox_.size[1]) *
                           static_cast<size_t>(inbox_.size[2]);
        velX_.resize(inboxSize);
        velY_.resize(inboxSize);
        velZ_.resize(inboxSize);
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

    void calculate_power_spectrum(T* ps_rad, size_t numShells)
    {
        calculate_fft();

        std::vector<T> freqVelo(velX_.size());

        // calculate the modulus
        for (size_t i = 0; i < velX_.size(); i++)
        {
            freqVelo[i] = velX_[i] + velY_[i] + velZ_[i];
        }

        // perform spherical averaging
        perform_spherical_averaging(freqVelo.data(), ps_rad, numShells);
    }

    void calculate_fft()
    {
        heffte::box3d<> outbox = inbox_;

        // change fftw depending on the configuration into cufft or rocmfft
        heffte::fft3d<heffte::backend::fftw> fft(inbox_, outbox, MPI_COMM_WORLD);

        std::vector<std::complex<T>> output(fft.size_outbox());

        fft.forward(velX_.data(), output.data());
        std::cout << "fft for X dim done." << std::endl;

        for (size_t i = 0; i < velX_.size(); i++)
        {
            velX_[i] = abs(output.at(i)) * abs(output.at(i));
        }

        fft.forward(velY_.data(), output.data());
        std::cout << "fft for Y dim done." << std::endl;

        for (size_t i = 0; i < velX_.size(); i++)
        {
            velY_[i] = abs(output.at(i)) * abs(output.at(i));
        }

        fft.forward(velZ_.data(), output.data());
        std::cout << "fft for Z dim done." << std::endl;

        for (size_t i = 0; i < velX_.size(); i++)
        {
            velZ_[i] = abs(output.at(i)) * abs(output.at(i));
        }
    }

    // Need to implement this using MPI
    void perform_spherical_averaging(T* ps, T* ps_rad, size_t gridDim)
    {
        std::vector<T> k_values(gridDim);
        std::vector<T> k_1d(gridDim);
        std::vector<T> ps_radial(gridDim);

        for (size_t i = 0; i < k_values.size() / 2; i++)
        {
            k_values[i] = i;
        }
        size_t val = 0;
        for (size_t i = k_values.size(); i >= k_values.size() / 2; i--)
        {
            k_values[i] = -val;
            val++;
        }
        std::cout << "k_1d before" << std::endl;
        for (size_t i = 0; i < gridDim; i++)
        {
            k_1d[i] = std::abs(k_values[i]);
            std::cout << k_1d[i] << ", ";
        }

        for (size_t i = 0; i < gridDim; i++)
        {
            for (size_t j = 0; j < gridDim; j++)
            {
                for (size_t k = 0; k < gridDim; k++)
                {
                    T kdist =
                        std::sqrt(k_values[i] * k_values[i] + k_values[j] * k_values[j] + k_values[k] * k_values[k]);
                    std::vector<T> k_dif(gridDim);
                    for (size_t kind = 0; kind < gridDim; kind++)
                    {
                        k_dif[kind] = std::abs(k_1d[kind] - kdist);
                    }
                    auto   it      = std::min_element(std::begin(k_dif), std::end(k_dif));
                    size_t k_index = std::distance(std::begin(k_dif), it);

                    size_t ps_index = (i * gridDim + j) * gridDim + k;
                    ps_radial[k_index] += ps[ps_index];
                }
            }
        }

        T sum_ps_radial = std::accumulate(ps_radial.begin(), ps_radial.end(), 0.0);

        for (size_t i = 0; i < gridDim; i++)
        {
            ps_rad[i] = ps_radial[i] / sum_ps_radial;
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