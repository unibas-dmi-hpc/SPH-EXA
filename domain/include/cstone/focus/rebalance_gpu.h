/*
 * MIT License
 *
 * Copyright (c) 2022 CSCS, ETH Zurich
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
 * @brief Focused octree rebalance on GPUs
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include "cstone/tree/definitions.h"

namespace cstone
{

template<class KeyType>
extern void rebalanceDecisionEssentialGpu(const KeyType* prefixes,
                                          const TreeNodeIndex* childOffsets,
                                          const TreeNodeIndex* parents,
                                          const unsigned* counts,
                                          const char* macs,
                                          KeyType focusStart,
                                          KeyType focusEnd,
                                          unsigned bucketSize,
                                          TreeNodeIndex* nodeOps,
                                          TreeNodeIndex numNodes);

template<class KeyType>
extern bool protectAncestorsGpu(const KeyType*, const TreeNodeIndex*, TreeNodeIndex*, TreeNodeIndex);

template<class KeyType>
extern ResolutionStatus enforceKeysGpu(const KeyType* forcedKeys,
                                       TreeNodeIndex numForcedKeys,
                                       const KeyType* nodeKeys,
                                       const TreeNodeIndex* childOffsets,
                                       const TreeNodeIndex* parents,
                                       TreeNodeIndex* nodeOps);

} // namespace cstone