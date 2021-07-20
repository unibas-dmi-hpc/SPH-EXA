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
 * @brief Test hilbert code implementation
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include <algorithm>
#include <array>
#include <random>
#include <vector>

#include "gtest/gtest.h"
#include "cstone/sfc/hilbert.hpp"

using namespace cstone;

//! @brief test the curve on the first eight octants
template<class KeyType>
void firstOrderCurve()
{
    constexpr unsigned hilbertToMorton[8] = {0, 1, 3, 2, 6, 7, 5, 4};

    for (unsigned xi = 0; xi < 2; ++xi)
    {
        for (unsigned yi = 0; yi < 2; ++yi)
        {
            for (unsigned zi = 0; zi < 2; ++zi)
            {
                unsigned L1Range = (1 << maxTreeLevel<KeyType>{}) / 2;
                unsigned mortonOctant = 4 * xi + 2 * yi + zi;

                {
                    KeyType hilbertKey = iHilbert<KeyType>(L1Range * xi, L1Range * yi, L1Range * zi);
                    unsigned hilbertOctant = octalDigit(hilbertKey, 1);
                    EXPECT_EQ(mortonOctant, hilbertToMorton[hilbertOctant]);
                }
                {
                    KeyType hilbertKey = iHilbert<KeyType>(L1Range * xi + L1Range - 1,
                                                           L1Range * yi + L1Range - 1,
                                                           L1Range * zi + L1Range - 1);
                    unsigned hilbertOctant = octalDigit(hilbertKey, 1);
                    EXPECT_EQ(mortonOctant, hilbertToMorton[hilbertOctant]);
                }
            }
        }
    }
}

TEST(HilbertCode, firstOrderCurve)
{
    firstOrderCurve<unsigned>();
    firstOrderCurve<uint64_t>();
}

//! @brief verifies continuity properties across consecutive octants at all levels
template<class KeyType>
void continuityTest()
{
    for (int level = 1; level < maxTreeLevel<KeyType>{}; ++level)
    {
        // on the highest level, we can only check 7 octant crossings
        int maxOctant = (level > 1) ? 8 : 7;

        for (int octant = 0; octant < maxOctant; ++octant)
        {
            KeyType lastKey = (octant + 1) * nodeRange<KeyType>(level) - 1;
            KeyType firstNextKey = lastKey + 1;

            unsigned lastPoint[3];
            idecodeHilbert(lastKey, lastPoint, lastPoint + 1, lastPoint + 2);

            unsigned firstNextPoint[3];
            idecodeHilbert(firstNextKey, firstNextPoint, firstNextPoint + 1, firstNextPoint + 2);

            // the points in 3D space should be right next to each other, i.e. delta == 1
            // this is a property that the Z-curve does not have
            int delta = std::abs(int(lastPoint[0]) - int(firstNextPoint[0])) +
                        std::abs(int(lastPoint[1]) - int(firstNextPoint[1])) +
                        std::abs(int(lastPoint[2]) - int(firstNextPoint[2]));

            EXPECT_EQ(delta, 1);
        }
    }
}

TEST(HilbertCode, continuity)
{
    continuityTest<unsigned>();
    continuityTest<uint64_t>();
}

//! @brief tests numKeys random 3D points for encoding/decoding consistency
template<class KeyType>
void inversionTest()
{
    int numKeys  = 1000;
    int maxCoord = (1 << maxTreeLevel<KeyType>{}) - 1;

    std::mt19937 gen;
    std::uniform_int_distribution<unsigned> distribution(0, maxCoord);

    auto getRand = [&distribution, &gen](){ return distribution(gen); };

    std::vector<unsigned> x(numKeys);
    std::vector<unsigned> y(numKeys);
    std::vector<unsigned> z(numKeys);

    std::generate(begin(x), end(x), getRand);
    std::generate(begin(y), end(y), getRand);
    std::generate(begin(z), end(z), getRand);

    for (int i = 0; i < numKeys; ++i)
    {
        KeyType hilbertKey = iHilbert<KeyType>(x[i], y[i], z[i]);

        unsigned decodedKey[3];
        idecodeHilbert(hilbertKey, decodedKey, decodedKey + 1, decodedKey + 2);
        EXPECT_EQ(x[i], decodedKey[0]) ;
        EXPECT_EQ(y[i], decodedKey[1]) ;
        EXPECT_EQ(z[i], decodedKey[2]) ;
    }
}

TEST(HilbertCode, inversion)
{
    inversionTest<unsigned>();
    inversionTest<uint64_t>();
}
