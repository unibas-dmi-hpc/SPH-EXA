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
 * @brief  Thrust sorting and prefix sums
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include <cstdint>

#include "cstone/tree/definitions.h"

void thrust_exclusive_scan(const cstone::TreeNodeIndex* first, const cstone::TreeNodeIndex* last,
                           cstone::TreeNodeIndex* dest);

void thrust_exclusive_scan(const unsigned* first, const unsigned* last, unsigned* dest);

void thrust_sort_by_key(cstone::TreeNodeIndex* firstKey, cstone::TreeNodeIndex* lastKey, cstone::TreeNodeIndex* value);

/*! @brief sort values by keys, first by level, then by SFC key value
 *
 * @param[inout] firstKey first SFC key prefixed with Warren-Salmon placeholder bit
 * @param[inout] lastKey  last SFC key
 * @param[inout] value    the iota sequence 0,1,2,3,4,... to record the sort order
 */
void thrust_sort_by_level_and_key(uint32_t* firstKey, uint32_t* lastKey, cstone::TreeNodeIndex* value);
void thrust_sort_by_level_and_key(uint64_t* firstKey, uint64_t* lastKey, cstone::TreeNodeIndex* value);
