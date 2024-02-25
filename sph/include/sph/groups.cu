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

template<class Dataset>
void computeTargetGroups(size_t startIndex, size_t endIndex, Dataset& d,
                         const cstone::Box<typename Dataset::RealType>& box)
{
    thrust::device_vector<util::array<GpuConfig::ThreadMask, TravConfig::nwt>> S;

    float tolFactor = 2.0f;
    cstone::computeGroupSplits<TravConfig::targetSize>(startIndex, endIndex, rawPtr(d.devData.x), rawPtr(d.devData.y),
                                                       rawPtr(d.devData.z), rawPtr(d.devData.h), d.treeView.leaves,
                                                       d.treeView.tree.numLeafNodes, d.treeView.layout, box, tolFactor,
                                                       S, d.devData.traversalStack, d.devData.targetGroups);
}

template void computeTargetGroups(size_t, size_t, sphexa::ParticlesData<cstone::GpuTag>& d,
                                  const cstone::Box<SphTypes::CoordinateType>&);

}
