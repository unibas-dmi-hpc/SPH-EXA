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

/*! \file
 * \brief Testing of the naive collision detection implementation
 *
 * \author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include "gtest/gtest.h"

#include "collisions_a2a.hpp"


using namespace cstone;

/*! \brief test the naive to-all collision detection function
 *
 * @tparam I  32- or 64-bit unsigned integer
 */
template<class I>
void collide2all()
{
    auto tree = OctreeMaker<I>{}.divide().divide(0).divide(0,7).makeTree();

    // this search box intersects with neighbors in x direction and will intersect
    // with multiple smaller level 2 and level 3 nodes
    // it corresponds to the node with code codeFromIndices<I>({4}) with a dx=1 halo extension
    int r = 1u<<(maxTreeLevel<I>{} - 1);
    Box<int> haloBox{r-1, 2*r, 0, r, 0, r};

    CollisionList collisionList;
    findCollisions2All(tree, collisionList, haloBox);

    std::vector<I> collisions(collisionList.size());
    for (int i = 0; i < collisions.size(); ++i)
        collisions[i] = tree[collisionList[i]];

    // list of octree leaf morton codes that should collide
    // with the halo box
    std::vector<I> refCollisions{
        codeFromIndices<I>({0,4}),
        codeFromIndices<I>({0,5}),
        codeFromIndices<I>({0,6}),
        codeFromIndices<I>({0,7,4}),
        codeFromIndices<I>({0,7,5}),
        codeFromIndices<I>({0,7,6}),
        codeFromIndices<I>({0,7,7}),
        codeFromIndices<I>({4})
    };

    EXPECT_EQ(collisions, refCollisions);
}

TEST(Collisions, collide2all)
{
    collide2all<unsigned>();
    collide2all<uint64_t>();
}

/*! \brief test the naive all-to-all collision detection function
 *
 * @tparam I  32- or 64-bit unsigned integer
 */
template<class I, class T>
void collideAll2all()
{
    auto tree = OctreeMaker<I>{}.divide().divide(0).divide(0,7).makeTree();

    Box<T> box(0, 1);
    std::vector<T> haloRadii(nNodes(tree), 0.1);

    std::vector<CollisionList> allCollisions = findCollisionsAll2all(tree, haloRadii, box);

    // extract list of collisions for node with index 18, corresponding to {4}
    std::vector<I> n18coll(allCollisions[18].size());
    for (int i = 0; i < n18coll.size(); ++i)
        n18coll[i] = tree[allCollisions[18][i]];

    std::sort(begin(n18coll), end(n18coll));

    // reference list of collisions for node with index 18, corresponding to {4}
    std::vector<I> refCollisions{
        codeFromIndices<I>({0,4}),
        codeFromIndices<I>({0,5}),
        codeFromIndices<I>({0,6}),
        codeFromIndices<I>({0,7,4}),
        codeFromIndices<I>({0,7,5}),
        codeFromIndices<I>({0,7,6}),
        codeFromIndices<I>({0,7,7}),
        codeFromIndices<I>({1}),
        codeFromIndices<I>({2}),
        codeFromIndices<I>({3}),
        codeFromIndices<I>({4}),
        codeFromIndices<I>({5}),
        codeFromIndices<I>({6}),
        codeFromIndices<I>({7})
    };

    EXPECT_EQ(n18coll, refCollisions);
}

TEST(Collisions, collideAll2all)
{
    collideAll2all<unsigned, float>();
    collideAll2all<uint64_t, float>();
    collideAll2all<unsigned, double>();
    collideAll2all<uint64_t, double>();
}

/*! \brief test the naive all-to-all collision detection function, PBC-X case
 *
 * @tparam I  32- or 64-bit unsigned integer
 */
template<class I, class T>
void collideAll2allPbcX()
{
    auto tree = OctreeMaker<I>{}.divide().divide(0).divide(0,7).makeTree();

    Box<T> box(0, 1, 0, 1, 0, 1, true, false, false);
    std::vector<T> haloRadii(nNodes(tree), 0.1);

    std::vector<CollisionList> allCollisions = findCollisionsAll2all(tree, haloRadii, box);

    // extract list of collisions for node with index 18, corresponding to {4}
    std::vector<I> n18coll(allCollisions[18].size());
    for (int i = 0; i < n18coll.size(); ++i)
        n18coll[i] = tree[allCollisions[18][i]];

    std::sort(begin(n18coll), end(n18coll));

    // reference list of collisions for node with index 18, corresponding to {4}
    std::vector<I> refCollisions{
            codeFromIndices<I>({0,0}), // due to pbc X
            codeFromIndices<I>({0,1}), // due to pbc X
            codeFromIndices<I>({0,2}), // due to pbc X
            codeFromIndices<I>({0,3}), // due to pbc X
            codeFromIndices<I>({0,4}),
            codeFromIndices<I>({0,5}),
            codeFromIndices<I>({0,6}),
            codeFromIndices<I>({0,7,4}),
            codeFromIndices<I>({0,7,5}),
            codeFromIndices<I>({0,7,6}),
            codeFromIndices<I>({0,7,7}),
            codeFromIndices<I>({1}),
            codeFromIndices<I>({2}),
            codeFromIndices<I>({3}),
            codeFromIndices<I>({4}),
            codeFromIndices<I>({5}),
            codeFromIndices<I>({6}),
            codeFromIndices<I>({7})
    };

    EXPECT_EQ(n18coll, refCollisions);
}

TEST(Collisions, collideAll2allPbcX)
{
    collideAll2allPbcX<unsigned, float>();
    collideAll2allPbcX<uint64_t, float>();
    collideAll2allPbcX<unsigned, double>();
    collideAll2allPbcX<uint64_t, double>();
}


/*! \brief test the naive all-to-all collision detection function, PBC-XYZ case
 *
 * @tparam I  32- or 64-bit unsigned integer
 */
template<class I, class T>
void collideAll2allPbcXYZ()
{
    auto tree = OctreeMaker<I>{}.divide().divide(0).divide(0,7).divide(5).divide(6).makeTree();

    Box<T> box(0, 1, 0, 1, 0, 1, true, true, true);
    std::vector<T> haloRadii(nNodes(tree), 0.1);

    std::vector<CollisionList> allCollisions = findCollisionsAll2all(tree, haloRadii, box);

    // extract list of collisions for node with index 18, corresponding to {4}
    std::vector<I> n18coll(allCollisions[18].size());
    for (int i = 0; i < n18coll.size(); ++i)
        n18coll[i] = tree[allCollisions[18][i]];

    std::sort(begin(n18coll), end(n18coll));

    // reference list of collisions for node with index 18, corresponding to {4}
    std::vector<I> refCollisions{
            codeFromIndices<I>({0,0}), // due to pbc X
            codeFromIndices<I>({0,1}), // due to pbc X
            codeFromIndices<I>({0,2}), // due to pbc X
            codeFromIndices<I>({0,3}), // due to pbc X
            codeFromIndices<I>({0,4}),
            codeFromIndices<I>({0,5}),
            codeFromIndices<I>({0,6}),
            codeFromIndices<I>({0,7,4}),
            codeFromIndices<I>({0,7,5}),
            codeFromIndices<I>({0,7,6}),
            codeFromIndices<I>({0,7,7}),
            codeFromIndices<I>({1}),
            codeFromIndices<I>({2}),
            codeFromIndices<I>({3}),
            codeFromIndices<I>({4}),
            codeFromIndices<I>({5,0}),
            codeFromIndices<I>({5,1}), // due to pbc Z
            codeFromIndices<I>({5,2}),
            codeFromIndices<I>({5,3}), // due to pbc Z
            codeFromIndices<I>({5,4}),
            codeFromIndices<I>({5,5}), // due to pbc Z
            codeFromIndices<I>({5,6}),
            codeFromIndices<I>({5,7}), // due to pbc Z
            codeFromIndices<I>({6,0}),
            codeFromIndices<I>({6,1}),
            codeFromIndices<I>({6,2}), // due to pbc Y
            codeFromIndices<I>({6,3}), // due to pbc Y
            codeFromIndices<I>({6,4}),
            codeFromIndices<I>({6,5}),
            codeFromIndices<I>({6,6}), // due to pbc Y
            codeFromIndices<I>({6,7}), // due to pbc Y
            codeFromIndices<I>({7})
    };

    EXPECT_EQ(n18coll, refCollisions);
}

TEST(Collisions, collideAll2allPbcXYZ)
{
    collideAll2allPbcXYZ<unsigned, float>();
    collideAll2allPbcXYZ<uint64_t, float>();
    collideAll2allPbcXYZ<unsigned, double>();
    collideAll2allPbcXYZ<uint64_t, double>();
}
