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
 * @brief Halo discovery tests
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include "gtest/gtest.h"

#include "cstone/halos/discovery.hpp"
#include "cstone/tree/octree_util.hpp"

using namespace cstone;

/*! @brief find halo test
 *
 * A regular 4x4x4 tree with 64 nodes is assigned to 2 ranks,
 * such that nodes 0-32 are on rank 0 and 32-64 on rank 1,
 * or, in x,y,z coordinates,
 *
 * nodes (0-2, 0-4, 0-4) -> rank 0
 * nodes (2-4, 0-4, 0-4) -> rank 1
 *
 * Halo search radius is less than a node edge length, so the halo nodes are
 *
 * (2, 0-4, 0-4) halos of rank 0
 * (1, 0-4, 0-4) halos of rank 1
 */
template <class KeyType>
void findHalos()
{
    // a tree with 4 subdivisions along each dimension, 64 nodes
    std::vector<KeyType> tree = makeUniformNLevelTree<KeyType>(64, 1);

    Box<double> box(0, 1);

    // size of one node is 0.25^3
    std::vector<double> interactionRadii(nNodes(tree), 0.1);

    std::vector<pair<int>> refPairs0;
    for (std::size_t i = 0; i < nNodes(tree) / 2u; ++i)
        for (std::size_t j = nNodes(tree) / 2u; j < nNodes(tree); ++j)
        {
            if (overlap(tree[i], tree[i + 1], makeHaloBox(tree[j], tree[j + 1], interactionRadii[j], box)))
            {
                refPairs0.emplace_back(i, j);
            }
        }
    std::sort(begin(refPairs0), end(refPairs0));
    EXPECT_EQ(refPairs0.size(), 100);

    {
        std::vector<pair<int>> testPairs0;
        findHalos<KeyType, double>(tree, interactionRadii, box, 0, 32, testPairs0);
        std::sort(begin(testPairs0), end(testPairs0));

        EXPECT_EQ(testPairs0.size(), 100);
        EXPECT_EQ(testPairs0, refPairs0);
    }

    auto refPairs1 = refPairs0;
    for (auto& p : refPairs1)
        std::swap(p[0], p[1]);
    std::sort(begin(refPairs1), end(refPairs1));

    {
        std::vector<pair<int>> testPairs1;
        findHalos<KeyType, double>(tree, interactionRadii, box, 32, 64, testPairs1);
        std::sort(begin(testPairs1), end(testPairs1));
        EXPECT_EQ(testPairs1.size(), 100);
        EXPECT_EQ(testPairs1, refPairs1);
    }
}

TEST(HaloDiscovery, findHalos)
{
    findHalos<unsigned>();
    findHalos<uint64_t>();
}


//! @brief an integration test between findHalos, computeSendRecvNodeList and Pbc
template<class KeyType>
void findHalosPbc()
{
    // a tree with 4 subdivisions along each dimension, 64 nodes
    std::vector<KeyType> tree = makeUniformNLevelTree<KeyType>(64, 1);

    Box<double> box(0, 1, 0, 1, 0, 1, true, true, true);

    // size of one node is 0.25^3
    std::vector<double> interactionRadii(nNodes(tree), 0.1);
    {
        std::vector<pair<TreeNodeIndex>> haloPairs;
        findHalos<KeyType, double>(tree, interactionRadii, box, 0, 32, haloPairs);

        std::vector<pair<TreeNodeIndex>> reference{
            {0, 36}, {0, 37}, {0, 38}, {0, 39}, {0, 45}, {0, 47}, {0, 54}, {0, 55}, {0, 63},
            {1, 36}, {1, 37}, {1, 38}, {1, 39}, {1, 44}, {1, 46}, {1, 54}, {1, 55}, {1, 62},
            {2, 36}, {2, 37}, {2, 38}, {2, 39}, {2, 45}, {2, 47}, {2, 52}, {2, 53}, {2, 61},
            {3, 36}, {3, 37}, {3, 38}, {3, 39}, {3, 44}, {3, 46}, {3, 52}, {3, 53}, {3, 60},
            {4, 32}, {4, 33}, {4, 34}, {4, 35}, {4, 41}, {4, 43}, {4, 50}, {4, 51}, {4, 59},
            {5, 32}, {5, 33}, {5, 34}, {5, 35}, {5, 40}, {5, 42}, {5, 50}, {5, 51}, {5, 58},
            {6, 32}, {6, 33}, {6, 34}, {6, 35}, {6, 41}, {6, 43}, {6, 48}, {6, 49}, {6, 57},
            {7, 32}, {7, 33}, {7, 34}, {7, 35}, {7, 40}, {7, 42}, {7, 48}, {7, 49}, {7, 56},
            {8, 37}, {8, 39}, {8, 44}, {8, 45}, {8, 46}, {8, 47}, {8, 55}, {8, 62}, {8, 63},
            {9, 36}, {9, 38}, {9, 44}, {9, 45}, {9, 46}, {9, 47}, {9, 54}, {9, 62}, {9, 63},
            {10, 37}, {10, 39}, {10, 44}, {10, 45}, {10, 46}, {10, 47}, {10, 53}, {10, 60}, {10, 61},
            {11, 36}, {11, 38}, {11, 44}, {11, 45}, {11, 46}, {11, 47}, {11, 52}, {11, 60}, {11, 61},
            {12, 33}, {12, 35}, {12, 40}, {12, 41}, {12, 42}, {12, 43}, {12, 51}, {12, 58}, {12, 59},
            {13, 32}, {13, 34}, {13, 40}, {13, 41}, {13, 42}, {13, 43}, {13, 50}, {13, 58}, {13, 59},
            {14, 33}, {14, 35}, {14, 40}, {14, 41}, {14, 42}, {14, 43}, {14, 49}, {14, 56}, {14, 57},
            {15, 32}, {15, 34}, {15, 40}, {15, 41}, {15, 42}, {15, 43}, {15, 48}, {15, 56}, {15, 57},
            {16, 38}, {16, 39}, {16, 47}, {16, 52}, {16, 53}, {16, 54}, {16, 55}, {16, 61}, {16, 63},
            {17, 38}, {17, 39}, {17, 46}, {17, 52}, {17, 53}, {17, 54}, {17, 55}, {17, 60}, {17, 62},
            {18, 36}, {18, 37}, {18, 45}, {18, 52}, {18, 53}, {18, 54}, {18, 55}, {18, 61}, {18, 63},
            {19, 36}, {19, 37}, {19, 44}, {19, 52}, {19, 53}, {19, 54}, {19, 55}, {19, 60}, {19, 62},
            {20, 34}, {20, 35}, {20, 43}, {20, 48}, {20, 49}, {20, 50}, {20, 51}, {20, 57}, {20, 59},
            {21, 34}, {21, 35}, {21, 42}, {21, 48}, {21, 49}, {21, 50}, {21, 51}, {21, 56}, {21, 58},
            {22, 32}, {22, 33}, {22, 41}, {22, 48}, {22, 49}, {22, 50}, {22, 51}, {22, 57}, {22, 59},
            {23, 32}, {23, 33}, {23, 40}, {23, 48}, {23, 49}, {23, 50}, {23, 51}, {23, 56}, {23, 58},
            {24, 39}, {24, 46}, {24, 47}, {24, 53}, {24, 55}, {24, 60}, {24, 61}, {24, 62}, {24, 63},
            {25, 38}, {25, 46}, {25, 47}, {25, 52}, {25, 54}, {25, 60}, {25, 61}, {25, 62}, {25, 63},
            {26, 37}, {26, 44}, {26, 45}, {26, 53}, {26, 55}, {26, 60}, {26, 61}, {26, 62}, {26, 63},
            {27, 36}, {27, 44}, {27, 45}, {27, 52}, {27, 54}, {27, 60}, {27, 61}, {27, 62}, {27, 63},
            {28, 35}, {28, 42}, {28, 43}, {28, 49}, {28, 51}, {28, 56}, {28, 57}, {28, 58}, {28, 59},
            {29, 34}, {29, 42}, {29, 43}, {29, 48}, {29, 50}, {29, 56}, {29, 57}, {29, 58}, {29, 59},
            {30, 33}, {30, 40}, {30, 41}, {30, 49}, {30, 51}, {30, 56}, {30, 57}, {30, 58}, {30, 59},
            {31, 32}, {31, 40}, {31, 41}, {31, 48}, {31, 50}, {31, 56}, {31, 57}, {31, 58}, {31, 59}
        };
        auto comp = [](auto a, auto b){ return std::tie(a[0], a[1]) < std::tie(b[0], b[1]); };
        std::sort(begin(haloPairs), end(haloPairs), comp);
        std::sort(begin(reference), end(reference), comp);
        EXPECT_EQ(haloPairs, reference);
    }
    {
        std::vector<pair<TreeNodeIndex>> haloPairs;
        findHalos<KeyType, double>(tree, interactionRadii, box, 32, 64, haloPairs);

        std::vector<pair<TreeNodeIndex>> reference{
            {32, 4}, {32, 5}, {32, 6}, {32, 7}, {32, 13}, {32, 15}, {32, 22}, {32, 23}, {32, 31},
            {33, 4}, {33, 5}, {33, 6}, {33, 7}, {33, 12}, {33, 14}, {33, 22}, {33, 23}, {33, 30},
            {34, 4}, {34, 5}, {34, 6}, {34, 7}, {34, 13}, {34, 15}, {34, 20}, {34, 21}, {34, 29},
            {35, 4}, {35, 5}, {35, 6}, {35, 7}, {35, 12}, {35, 14}, {35, 20}, {35, 21}, {35, 28},
            {36, 0}, {36, 1}, {36, 2}, {36, 3}, {36, 9}, {36, 11}, {36, 18}, {36, 19}, {36, 27},
            {37, 0}, {37, 1}, {37, 2}, {37, 3}, {37, 8}, {37, 10}, {37, 18}, {37, 19}, {37, 26},
            {38, 0}, {38, 1}, {38, 2}, {38, 3}, {38, 9}, {38, 11}, {38, 16}, {38, 17}, {38, 25},
            {39, 0}, {39, 1}, {39, 2}, {39, 3}, {39, 8}, {39, 10}, {39, 16}, {39, 17}, {39, 24}, {40, 5},
            {40, 7}, {40, 12}, {40, 13}, {40, 14}, {40, 15}, {40, 23}, {40, 30}, {40, 31},
            {41, 4}, {41, 6}, {41, 12}, {41, 13}, {41, 14}, {41, 15}, {41, 22}, {41, 30}, {41, 31},
            {42, 5}, {42, 7}, {42, 12}, {42, 13}, {42, 14}, {42, 15}, {42, 21}, {42, 28}, {42, 29},
            {43, 4}, {43, 6}, {43, 12}, {43, 13}, {43, 14}, {43, 15}, {43, 20}, {43, 28}, {43, 29},
            {44, 1}, {44, 3}, {44, 8}, {44, 9}, {44, 10}, {44, 11}, {44, 19}, {44, 26}, {44, 27},
            {45, 0}, {45, 2}, {45, 8}, {45, 9}, {45, 10}, {45, 11}, {45, 18}, {45, 26}, {45, 27},
            {46, 1}, {46, 3}, {46, 8}, {46, 9}, {46, 10}, {46, 11}, {46, 17}, {46, 24}, {46, 25},
            {47, 0}, {47, 2}, {47, 8}, {47, 9}, {47, 10}, {47, 11}, {47, 16}, {47, 24}, {47, 25},
            {48, 6}, {48, 7}, {48, 15}, {48, 20}, {48, 21}, {48, 22}, {48, 23}, {48, 29}, {48, 31},
            {49, 6}, {49, 7}, {49, 14}, {49, 20}, {49, 21}, {49, 22}, {49, 23}, {49, 28}, {49, 30},
            {50, 4}, {50, 5}, {50, 13}, {50, 20}, {50, 21}, {50, 22}, {50, 23}, {50, 29}, {50, 31},
            {51, 4}, {51, 5}, {51, 12}, {51, 20}, {51, 21}, {51, 22}, {51, 23}, {51, 28}, {51, 30},
            {52, 2}, {52, 3}, {52, 11}, {52, 16}, {52, 17}, {52, 18}, {52, 19}, {52, 25}, {52, 27},
            {53, 2}, {53, 3}, {53, 10}, {53, 16}, {53, 17}, {53, 18}, {53, 19}, {53, 24}, {53, 26},
            {54, 0}, {54, 1}, {54, 9}, {54, 16}, {54, 17}, {54, 18}, {54, 19}, {54, 25}, {54, 27},
            {55, 0}, {55, 1}, {55, 8}, {55, 16}, {55, 17}, {55, 18}, {55, 19}, {55, 24}, {55, 26},
            {56, 7}, {56, 14}, {56, 15}, {56, 21}, {56, 23}, {56, 28}, {56, 29}, {56, 30}, {56, 31},
            {57, 6}, {57, 14}, {57, 15}, {57, 20}, {57, 22}, {57, 28}, {57, 29}, {57, 30}, {57, 31},
            {58, 5}, {58, 12}, {58, 13}, {58, 21}, {58, 23}, {58, 28}, {58, 29}, {58, 30}, {58, 31},
            {59, 4}, {59, 12}, {59, 13}, {59, 20}, {59, 22}, {59, 28}, {59, 29}, {59, 30}, {59, 31},
            {60, 3}, {60, 10}, {60, 11}, {60, 17}, {60, 19}, {60, 24}, {60, 25}, {60, 26}, {60, 27},
            {61, 2}, {61, 10}, {61, 11}, {61, 16}, {61, 18}, {61, 24}, {61, 25}, {61, 26}, {61, 27},
            {62, 1}, {62, 8}, {62, 9}, {62, 17}, {62, 19}, {62, 24}, {62, 25}, {62, 26}, {62, 27},
            {63, 0}, {63, 8}, {63, 9}, {63, 16}, {63, 18}, {63, 24}, {63, 25}, {63, 26}, {63, 27},
        };
        auto comp = [](auto a, auto b){ return std::tie(a[0], a[1]) < std::tie(b[0], b[1]); };
        std::sort(begin(haloPairs), end(haloPairs), comp);
        std::sort(begin(reference), end(reference), comp);
        EXPECT_EQ(haloPairs, reference);
    }
}

TEST(HaloDiscovery, findHalosPbc)
{
    findHalosPbc<unsigned>();
    findHalosPbc<uint64_t>();
}