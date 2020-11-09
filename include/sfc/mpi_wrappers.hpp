
#pragma once

#include <mpi/mpi.h>

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



