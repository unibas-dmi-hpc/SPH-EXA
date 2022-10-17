/*
 * MIT License
 *
 * Copyright (c) 2021 CSCS, ETH Zurich
 *               2021 University of Basel
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

/*! @file
 * @brief MPI utility functions, and wrapper for "distributed data".
 *
 * @author Joseph Touzet <joseph.touzet@ens-paris-saclay.fr>
 */

#pragma once

#include "nnet-util/algorithm.hpp"

#ifdef USE_MPI
#include <mpi.h>
#else
struct mpi_comm
{
};
typedef mpi_comm MPI_Comm;
#endif

#include <vector>
#include <numeric>
#include <algorithm>

namespace sphexa::mpi
{
/*! @brief class correlating particles to detached data */
template<typename Int>
class mpi_partition
{
public:
    // references to pointers
    std::vector<int> const* node_id;
    std::vector<Int> const* particle_id;

    // send partition limits
    std::vector<int> send_disp;
    std::vector<int> send_count;

    // send partition
    std::vector<Int> send_partition;

    // send partition limits
    std::vector<int> recv_disp;
    std::vector<int> recv_count;

    // send partition
    std::vector<Int> recv_partition;

    mpi_partition() {}
    mpi_partition(const std::vector<int>& node_id_, const std::vector<Int>& particle_id_)
        : node_id(&node_id_)
        , particle_id(&particle_id_)
    {
    }

    void resize_comm_size(const int size)
    {
        send_disp.resize(size + 1, 0);
        send_count.resize(size, 0);

        recv_disp.resize(size + 1, 0);
        recv_count.resize(size, 0);
    }

    void resize_num_send(const int N)
    {
        send_partition.resize(N);
        recv_partition.resize(N);
    }
};

/*! @brief function that create a mpi partition from particle detached data pointer
 *
 * @param firstIndex   index of the first considered particle (included) in the node_id and particle_id vectors
 * @param lastIndex    index of the last considered particle (excluded) in the node_id and particle_id vectors
 * @param node_id      id of the node owning the corresponding detached data
 * @param particle_id  id of the local "detached particle" owned by the corresponding "node_id"
 * @param comm         MPI communicator
 *
 * Returns mpi_partition correlating the detached data and attached data
 */
template<typename Int>
mpi_partition<Int> partitionFromPointers(size_t firstIndex, size_t lastIndex, const std::vector<int>& node_id,
                                         const std::vector<Int>& particle_id, MPI_Comm comm)
{
    mpi_partition<Int> partition(node_id, particle_id);

#ifdef USE_MPI
    int rank, size;
    MPI_Comm_size(comm, &size);
    MPI_Comm_rank(comm, &rank);

    // prepare vector sizes
    const int n_particles = lastIndex - firstIndex;
    partition.resize_comm_size(size);
    partition.send_partition.resize(n_particles);

    // localy partition
    util::parallel_generalized_partition_from_iota(partition.send_partition.begin(), partition.send_partition.end(),
                                                   firstIndex, partition.send_disp.begin(), partition.send_disp.end(),
                                                   [&](const int idx) { return node_id[idx]; });

    // send counts
    partition.recv_disp[0] = 0;
    partition.send_disp[0] = 0;
    std::adjacent_difference(partition.send_disp.begin() + 1, partition.send_disp.end(), partition.send_count.begin());
    MPI_Alltoall(&partition.send_count[0], 1, MPI_INT, &partition.recv_count[0], 1, MPI_INT, comm);
    std::partial_sum(partition.recv_count.begin(), partition.recv_count.end(), partition.recv_disp.begin() + 1);

    // send particle id
    // prepare send buffer
    size_t              n_particles_send = partition.send_disp[size];
    std::vector<size_t> send_buffer(n_particles_send);
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n_particles_send; ++i)
        send_buffer[i] = particle_id[partition.send_partition[i]];

    // prepare recv buffer
    size_t n_particles_recv = partition.recv_disp[size];
    partition.recv_partition.resize(n_particles_recv);

    // send particle id
    MPI_Alltoallv(&send_buffer[0], &partition.send_count[0], &partition.send_disp[0], MPI_UNSIGNED_LONG_LONG,
                  &partition.recv_partition[0], &partition.recv_count[0], &partition.recv_disp[0],
                  MPI_UNSIGNED_LONG_LONG, comm);
#endif

    return partition;
}

#ifdef USE_MPI
/*! @brief function that initialize node_id and particle_id for "maximal mixing".
 *
 * @param firstIndex   index of the first particle to be considered (included) when populating the node_id and
 * particle_id vectors
 * @param lastIndex    index of the last particle to be considered (excluded)  when populating the node_id and
 * particle_id vectors
 * @param node_id      node id vector to be populated
 * @param particle_id  particle id vector to be populated
 * @param comm         MPI communicator
 */
template<typename Int>
void initializePointers(size_t firstIndex, size_t lastIndex, std::vector<int>& node_id, std::vector<Int>& particle_id,
                        MPI_Comm comm)
{
    int rank, size;
    MPI_Comm_size(comm, &size);
    MPI_Comm_rank(comm, &rank);

    // share the number of particle per node
    std::vector<size_t> n_particles(size);
    size_t              local_n_particles = lastIndex - firstIndex;
    MPI_Allgather(&local_n_particles, 1, MPI_UNSIGNED_LONG_LONG, &n_particles[0], 1, MPI_UNSIGNED_LONG_LONG, comm);

    // initialize "node_id" and "particle_id"
    size_t global_idx_begin = std::accumulate(n_particles.begin(), n_particles.begin() + rank, (size_t)0);

#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < local_n_particles; ++i)
    {
        size_t global_idx           = global_idx_begin + i;
        node_id[firstIndex + i]     = global_idx % size;
        particle_id[firstIndex + i] = global_idx / size;
    }
}
#endif

#ifdef USE_MPI
/*! @brief function that sync attached to detached data
 *
 * @param partition    MPI partition
 * @param send_vector  buffer to be sent (from attached to detached data)
 * @param recv_vector  receive buffer
 * @param datatype     MPI datatype of data to be sent/received
 * @param comm         MPI communicator
 */
template<typename T, typename Int>
void syncDataToStaticPartition(const mpi_partition<Int>& partition, const T* send_vector, T* recv_vector,
                               const MPI_Datatype datatype, MPI_Comm comm)
{
    int size;
    MPI_Comm_size(comm, &size);

    const size_t n_particles_send = partition.send_disp[size];
    const size_t n_particles_recv = partition.recv_disp[size];

    // prepare (partition) buffer
    std::vector<T> send_buffer(n_particles_send), recv_buffer(n_particles_recv);

#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n_particles_send; ++i)
        send_buffer[i] = send_vector[partition.send_partition[i]];

    // send buffer
    MPI_Alltoallv(&send_buffer[0], &partition.send_count[0], &partition.send_disp[0], datatype, &recv_buffer[0],
                  &partition.recv_count[0], &partition.recv_disp[0], datatype, comm);

// reconstruct (un-partition) vector from buffer
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n_particles_recv; ++i)
        recv_vector[partition.recv_partition[i]] = recv_buffer[i];
}
template<typename Int>
void syncDataToStaticPartition(const mpi_partition<Int>& partition, const double* send_vector, double* recv_vector,
                               MPI_Comm comm)
{
    syncDataToStaticPartition(partition, send_vector, recv_vector, MPI_DOUBLE, comm);
}
template<typename Int>
void syncDataToStaticPartition(const mpi_partition<Int>& partition, const float* send_vector, float* recv_vector,
                               MPI_Comm comm)
{
    syncDataToStaticPartition(partition, send_vector, recv_vector, MPI_FLOAT, comm);
}
template<typename Int>
void syncDataToStaticPartition(const mpi_partition<Int>& partition, const int* send_vector, int* recv_vector,
                               MPI_Comm comm)
{
    syncDataToStaticPartition(partition, send_vector, recv_vector, MPI_INT, comm);
}
template<typename Int>
void syncDataToStaticPartition(const mpi_partition<Int>& partition, const uint* send_vector, uint* recv_vector,
                               MPI_Comm comm)
{
    syncDataToStaticPartition(partition, send_vector, recv_vector, MPI_UNSIGNED, comm);
}
template<typename Int>
void syncDataToStaticPartition(const mpi_partition<Int>& partition, const size_t* send_vector, size_t* recv_vector,
                               MPI_Comm comm)
{
    syncDataToStaticPartition(partition, send_vector, recv_vector, MPI_UNSIGNED_LONG_LONG, comm);
}
template<typename T, typename Int>
void syncDataToStaticPartition(const mpi_partition<Int>& partition, const T* send_vector, T* recv_vector, MPI_Comm comm)
{
    throw std::runtime_error("Type not implictly supported by syncDataToStaticPartition\n");
}
template<typename T1, typename T2, typename Int>
void syncDataToStaticPartition(const mpi_partition<Int>& partition, const T1* send_vector, T2* recv_vector,
                               MPI_Comm comm)
{
    throw std::runtime_error("Type mismatch in syncDataToStaticPartition\n");
}

/*! @brief function that sync detached data to attached data
 *
 * @param partition    MPI partition
 * @param send_vector  buffer to be sent (from detached data to attached data)
 * @param recv_vector  receive buffer
 * @param datatype     MPI datatype of data to be sent/received
 * @param comm         MPI communicator
 */
template<typename T, typename Int>
void syncDataFromStaticPartition(const mpi_partition<Int>& partition, const T* send_vector, T* recv_vector,
                                 const MPI_Datatype datatype, MPI_Comm comm)
{
    // exact same thing as "direct_sync_data_from_partition" but with "send_ <-> recv_"

    int size;
    MPI_Comm_size(comm, &size);

    const size_t n_particles_send = partition.send_disp[size];
    const size_t n_particles_recv = partition.recv_disp[size];

    // prepare (partition) buffer
    std::vector<T> send_buffer(n_particles_recv), recv_buffer(n_particles_send);

#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n_particles_recv; ++i)
        send_buffer[i] = send_vector[partition.recv_partition[i]];

    // send buffer
    MPI_Alltoallv(&send_buffer[0], &partition.recv_count[0], &partition.recv_disp[0], datatype, &recv_buffer[0],
                  &partition.send_count[0], &partition.send_disp[0], datatype, comm);

// reconstruct (un-partition) vector from buffer
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n_particles_send; ++i)
        recv_vector[partition.send_partition[i]] = recv_buffer[i];
}
template<typename Int>
void syncDataFromStaticPartition(const mpi_partition<Int>& partition, const double* send_vector, double* recv_vector,
                                 MPI_Comm comm)
{
    syncDataFromStaticPartition(partition, send_vector, recv_vector, MPI_DOUBLE, comm);
}
template<typename Int>
void syncDataFromStaticPartition(const mpi_partition<Int>& partition, const float* send_vector, float* recv_vector,
                                 MPI_Comm comm)
{
    syncDataFromStaticPartition(partition, send_vector, recv_vector, MPI_FLOAT, comm);
}
template<typename Int>
void syncDataFromStaticPartition(const mpi_partition<Int>& partition, const int* send_vector, int* recv_vector,
                                 MPI_Comm comm)
{
    syncDataFromStaticPartition(partition, send_vector, recv_vector, MPI_INT, comm);
}
template<typename Int>
void syncDataFromStaticPartition(const mpi_partition<Int>& partition, const uint* send_vector, uint* recv_vector,
                                 MPI_Comm comm)
{
    syncDataFromStaticPartition(partition, send_vector, recv_vector, MPI_UNSIGNED, comm);
}
template<typename Int>
void syncDataFromStaticPartition(const mpi_partition<Int>& partition, const size_t* send_vector, size_t* recv_vector,
                                 MPI_Comm comm)
{
    syncDataFromStaticPartition(partition, send_vector, recv_vector, MPI_UNSIGNED_LONG_LONG, comm);
}
template<typename T, typename Int>
void syncDataFromStaticPartition(const mpi_partition<Int>& partition, const T* send_vector, T* recv_vector,
                                 MPI_Comm comm)
{
    throw std::runtime_error("Type not implictly supported by syncDataFromStaticPartition\n");
}
template<typename T1, typename T2, typename Int>
void syncDataFromStaticPartition(const mpi_partition<Int>& partition, const T1* send_vector, T2* recv_vector,
                                 MPI_Comm comm)
{
    throw std::runtime_error("Type mismatch in syncDataFromStaticPartition\n");
}
#endif
} // namespace sphexa::mpi