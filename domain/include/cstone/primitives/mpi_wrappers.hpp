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
 * @brief  A few C++ wrappers for MPI C functions
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <mpi.h>

template<class T>
struct MpiType
{
};

template<>
struct MpiType<double>
{
    constexpr operator MPI_Datatype() const noexcept { return MPI_DOUBLE; }
};

template<>
struct MpiType<float>
{
    constexpr operator MPI_Datatype() const noexcept { return MPI_FLOAT; }
};

template<>
struct MpiType<int>
{
    constexpr operator MPI_Datatype() const noexcept { return MPI_INT; }
};

template<>
struct MpiType<unsigned>
{
    constexpr operator MPI_Datatype() const noexcept { return MPI_UNSIGNED; }
};

template<>
struct MpiType<unsigned long>
{
    constexpr operator MPI_Datatype() const noexcept { return MPI_UNSIGNED_LONG; }
};

template<>
struct MpiType<unsigned long long>
{
    constexpr operator MPI_Datatype() const noexcept { return MPI_UNSIGNED_LONG_LONG; }
};

template<class T>
std::enable_if_t<std::is_same<double, std::decay_t<T>>{}>
mpiSendAsync(T* data, int count, int rank, int tag, std::vector<MPI_Request>& requests)
{
    requests.push_back(MPI_Request{});
    MPI_Isend(data, count, MPI_DOUBLE, rank, tag, MPI_COMM_WORLD, &requests.back());
}

template<class T>
std::enable_if_t<std::is_same<float, std::decay_t<T>>{}>
mpiSendAsync(T* data, int count, int rank, int tag, std::vector<MPI_Request>& requests)
{
    requests.push_back(MPI_Request{});
    MPI_Isend(data, count, MPI_FLOAT, rank, tag, MPI_COMM_WORLD, &requests.back());
}

template<class T>
std::enable_if_t<std::is_same<int, std::decay_t<T>>{}>
mpiSendAsync(T* data, int count, int rank, int tag, std::vector<MPI_Request>& requests)
{
    requests.push_back(MPI_Request{});
    MPI_Isend(data, count, MPI_INT, rank, tag, MPI_COMM_WORLD, &requests.back());
}

template<class T>
std::enable_if_t<std::is_same<unsigned, std::decay_t<T>>{}>
mpiSendAsync(T* data, int count, int rank, int tag, std::vector<MPI_Request>& requests)
{
    requests.push_back(MPI_Request{});
    MPI_Isend(data, count, MPI_UNSIGNED, rank, tag, MPI_COMM_WORLD, &requests.back());
}

template<class T>
std::enable_if_t<std::is_same<unsigned long, std::decay_t<T>>{}>
mpiSendAsync(T* data, int count, int rank, int tag, std::vector<MPI_Request>& requests)
{
    requests.push_back(MPI_Request{});
    MPI_Isend(data, count, MPI_UNSIGNED_LONG, rank, tag, MPI_COMM_WORLD, &requests.back());
}

template<class T>
std::enable_if_t<std::is_same<unsigned long long, std::decay_t<T>>{}>
mpiSendAsync(T* data, int count, int rank, int tag, std::vector<MPI_Request>& requests)
{
    requests.push_back(MPI_Request{});
    MPI_Isend(data, count, MPI_UNSIGNED_LONG_LONG, rank, tag, MPI_COMM_WORLD, &requests.back());
}

template<class T>
std::enable_if_t<std::is_same<double, std::decay_t<T>>{}>
mpiRecvSync(T* data, int count, int rank, int tag, MPI_Status* status)
{
    MPI_Recv(data, count, MPI_DOUBLE, rank, tag, MPI_COMM_WORLD, status);
}

template<class T>
std::enable_if_t<std::is_same<float, std::decay_t<T>>{}>
mpiRecvSync(T* data, int count, int rank, int tag, MPI_Status* status)
{
    MPI_Recv(data, count, MPI_FLOAT, rank, tag, MPI_COMM_WORLD, status);
}

template<class T>
std::enable_if_t<std::is_same<int, std::decay_t<T>>{}>
mpiRecvSync(T* data, int count, int rank, int tag, MPI_Status* status)
{
    MPI_Recv(data, count, MPI_INT, rank, tag, MPI_COMM_WORLD, status);
}

template<class T>
std::enable_if_t<std::is_same<unsigned, std::decay_t<T>>{}>
mpiRecvSync(T* data, int count, int rank, int tag, MPI_Status* status)
{
    MPI_Recv(data, count, MPI_UNSIGNED, rank, tag, MPI_COMM_WORLD, status);
}

template<class T>
std::enable_if_t<std::is_same<unsigned long, std::decay_t<T>>{}>
mpiRecvSync(T* data, int count, int rank, int tag, MPI_Status* status)
{
    MPI_Recv(data, count, MPI_UNSIGNED_LONG, rank, tag, MPI_COMM_WORLD, status);
}

template<class T>
std::enable_if_t<std::is_same<unsigned long long, std::decay_t<T>>{}>
mpiRecvSync(T* data, int count, int rank, int tag, MPI_Status* status)
{
    MPI_Recv(data, count, MPI_UNSIGNED_LONG_LONG, rank, tag, MPI_COMM_WORLD, status);
}
