#pragma once

#include <type_traits>
#include <thrust/device_vector.h>

#include "errorcheck.cuh"

template<class ThrustVec>
typename ThrustVec::value_type* rawPtr(ThrustVec& p)
{
    assert(p.size() && "cannot get pointer to unallocated device vector memory");
    return thrust::raw_pointer_cast(p.data());
}

template<class ThrustVec>
const typename ThrustVec::value_type* rawPtr(const ThrustVec& p)
{
    assert(p.size() && "cannot get pointer to unallocated device vector memory");
    return thrust::raw_pointer_cast(p.data());
}
