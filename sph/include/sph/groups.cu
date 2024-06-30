/*! @file
 * @brief Compute groups of target particles that a) fit in a warp and b) have compact bounding boxes
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include "thrust/device_vector.h"

#include "cstone/traversal/find_neighbors.cuh"
#include "cstone/traversal/groups.cuh"

#include "sph/sph_gpu.hpp"
#include "sph/particles_data.hpp"

namespace sph
{

using cstone::GpuConfig;
using cstone::TravConfig;

/*! @brief Compute groups of SFC-consecutive particles with compact bounding volumes
 *
 * @tparam     Dataset
 * @param[in]  startIndex   first particle index to include in groups
 * @param[in]  endIndex     last particle index in include in groups
 * @param[in]  d            Dataset with particle x,y,z,h arrays
 * @param[in]  box          global coordinate bounding box
 * @param[out] groups       output particle groups
 */
template<class Dataset>
void computeSpatialGroups(size_t startIndex, size_t endIndex, Dataset& d,
                          const cstone::Box<typename Dataset::RealType>& box, GroupData<cstone::GpuTag>& groups)
{
    thrust::device_vector<util::array<GpuConfig::ThreadMask, TravConfig::nwt>> S;

    float tolFactor = 2.0f;
    cstone::computeGroupSplits<TravConfig::targetSize>(startIndex, endIndex, rawPtr(d.devData.x), rawPtr(d.devData.y),
                                                       rawPtr(d.devData.z), rawPtr(d.devData.h), d.treeView.leaves,
                                                       d.treeView.numLeafNodes, d.treeView.layout, box, tolFactor, S,
                                                       d.devData.traversalStack, groups.data);

    groups.firstBody  = startIndex;
    groups.lastBody   = endIndex;
    groups.numGroups  = groups.data.size() - 1;
    groups.groupStart = rawPtr(groups.data);
    groups.groupEnd   = rawPtr(groups.data) + 1;
}

template void computeSpatialGroups(size_t, size_t, sphexa::ParticlesData<cstone::GpuTag>&,
                                   const cstone::Box<SphTypes::CoordinateType>&, GroupData<cstone::GpuTag>&);

} // namespace sph
