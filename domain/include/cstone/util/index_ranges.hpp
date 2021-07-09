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
 * @brief Utility class to express ranges of indices
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <cassert>
#include <tuple>
#include <vector>

#include "cstone/tree/definitions.h"
#include "util.hpp"

namespace cstone
{

template<class T>
struct IndexPair : public std::tuple<T, T>
{
    IndexPair() = default;
    IndexPair(T a, T b) : std::tuple<T, T>(a, b) {}

    T start() const { return std::get<0>(*this); }
    T end()   const { return std::get<1>(*this); }
    T count() const { return end() - start(); }
};

using TreeIndexPair = IndexPair<TreeNodeIndex>;

/*! @brief Stores ranges of local particles to be sent to another rank
 *
 * @tparam I  32- or 64-bit signed or unsigned integer to store the indices
 *
 *  Used for SendRanges with index ranges referencing elements in e.g. x,y,z,h arrays.
 */
template <class Index>
class IndexRanges
{
public:
    using IndexType = Index;

    IndexRanges()
        : totalCount_(0)
        , ranges_{}
    {
    }

    //! @brief add a local index range, infer count from difference
    void addRange(IndexType lower, IndexType upper)
    {
        assert(lower <= upper);
        ranges_.emplace_back(lower, upper);
        totalCount_ += upper - lower;
    }

    [[nodiscard]] IndexType rangeStart(size_t i) const { return ranges_[i][0]; }

    [[nodiscard]] IndexType rangeEnd(size_t i) const { return ranges_[i][1]; }

    //! @brief the number of particles in range i
    [[nodiscard]] std::size_t count(size_t i) const { return ranges_[i][1] - ranges_[i][0]; }

    //! @brief the sum of number of particles in all ranges or total send count
    [[nodiscard]] std::size_t totalCount() const { return totalCount_; }

    [[nodiscard]] std::size_t nRanges() const { return ranges_.size(); }

private:
    friend bool operator==(const IndexRanges& lhs, const IndexRanges& rhs)
    {
        return lhs.totalCount_ == rhs.totalCount_ && lhs.ranges_ == rhs.ranges_;
    }

    std::size_t totalCount_;
    std::vector<pair<IndexType>> ranges_;
};

//! @brief stores one or multiple index ranges of local particles to send out to another rank
using SendManifest = IndexRanges<LocalParticleIndex>;
//! @brief SendList will contain one manifest per rank
using SendList = std::vector<SendManifest>;

} // namespace cstone
