
#include "gtest/gtest.h"

#include "sfc/boxoverlap.hpp"

using namespace sphexa;

/*! \brief add (binary) zeros behind a prefix
 *
 * Allows comparisons, such as number of leading common bits (cpr)
 * of the prefix with Morton codes.
 *
 * @tparam I      32- or 64-bit unsigned integer type
 * @param prefix  the bit pattern
 * @param length  number of bits in the prefix
 * @return        prefix padded out with zeros
 *
 */
template <class I>
constexpr I pad(I prefix, int length)
{
    return prefix << (3*sphexa::maxTreeLevel<I>{} - length);
}

TEST(BoxOverlap, padUtility)
{
    EXPECT_EQ(pad(0b011,   3), 0b00011 << 27);
    EXPECT_EQ(pad(0b011ul, 3), 0b0011ul << 60);
}


/*! \brief Test overlap between octree nodes and coordinate ranges
 *
 * The octree node is given as a Morton code plus number of bits
 * and the coordinates as integer ranges.
 */
template <class I>
void overlapTest()
{
    // range of a level-2 node
    int r = I(1)<<(maxTreeLevel<I>{} - 2);

    // node range: [r,2r]^3
    I prefix         = pad(I(0b000111), 6);
    int prefixLength = 6;

    I bound = pad(I(0b001), 3);

    EXPECT_EQ(prefixLength, treeLevel(bound-prefix) * 3);

    /// Each test is a separate case

    EXPECT_FALSE(overlap(prefix, prefixLength, Box<int>{0, r, 0, r, 0, r}));
    EXPECT_FALSE(overlap(prefix, bound, Box<int>{0, r, 0, r, 0, r}));

    // exact match
    EXPECT_TRUE(overlap(prefix, prefixLength, Box<int>{r, 2*r, r, 2*r, r, 2*r}));
    // contained within (1,1,1) corner of node
    EXPECT_TRUE(overlap(prefix, prefixLength, Box<int>{2*r-1, 2*r, 2*r-1, 2*r, 2*r-1, 2*r}));
    // contained and exceeding (1,1,1) corner by 1 in all dimensions
    EXPECT_TRUE(overlap(prefix, prefixLength, Box<int>{2*r-1, 2*r+1, 2*r-1, 2*r+1, 2*r-1, 2*r+1}));

    // all of these miss the (1,1,1) corner by 1 in one of the three dimensions
    EXPECT_FALSE(overlap(prefix, prefixLength, Box<int>{2*r, 2*r+1, 2*r-1, 2*r, 2*r-1, 2*r}));
    EXPECT_FALSE(overlap(prefix, prefixLength, Box<int>{2*r-1, 2*r, 2*r, 2*r+1, 2*r-1, 2*r}));
    EXPECT_FALSE(overlap(prefix, prefixLength, Box<int>{2*r-1, 2*r, 2*r-1, 2*r, 2*r, 2*r+1}));

    // contained within (0,0,0) corner of node
    EXPECT_TRUE(overlap(prefix, prefixLength, Box<int>{r, r+1, r, r+1, r, r+1}));

    // all of these miss the (0,0,0) corner by 1 in one of the three dimensions
    EXPECT_FALSE(overlap(prefix, prefixLength, Box<int>{r-1, r, r, r+1, r, r+1}));
    EXPECT_FALSE(overlap(prefix, prefixLength, Box<int>{r, r+1, r-1, r, r, r+1}));
    EXPECT_FALSE(overlap(prefix, prefixLength, Box<int>{r, r+1, r, r+1, r-1, r}));
}

TEST(BoxOverlap, overlaps)
{
    overlapTest<unsigned>();
    overlapTest<uint64_t>();
}


//! \brief check halo box ranges in all spatial dimensions
template<class I>
void makeHaloBoxXYZ()
{
    int r = I(1) << (maxTreeLevel<I>{} - 3);
    // node range: [r,2r]^3
    I nodeStart = pad(I(0b000000111), 9);
    I nodeEnd = pad(I(0b000001000), 9);

    /// internal node check
    {
        Box<int> haloBox = makeHaloBox(nodeStart, nodeEnd, 1, 0, 0);
        Box<int> refBox{r-1, 2*r+1, r, 2*r, r, 2*r};
        EXPECT_EQ(haloBox, refBox);
    }
    {
        Box<int> haloBox = makeHaloBox(nodeStart, nodeEnd, 0, 1, 0);
        Box<int> refBox{r, 2*r, r-1, 2*r+1, r, 2*r};
        EXPECT_EQ(haloBox, refBox);
    }
    {
        Box<int> haloBox = makeHaloBox(nodeStart, nodeEnd, 0, 0, 1);
        Box<int> refBox{r, 2*r, r, 2*r, r-1, 2*r+1};
        EXPECT_EQ(haloBox, refBox);
    }
}

TEST(BoxOverlap, makeHaloBoxXYZ)
{
    makeHaloBoxXYZ<unsigned>();
    makeHaloBoxXYZ<uint64_t>();
}


//! \brief underflow check
template<class I>
void makeHaloBoxUnderflow()
{
    int r = I(1) << (maxTreeLevel<I>{} - 1);
    // node range: [r,2r]^3
    I nodeStart = pad(I(0b000), 3);
    I nodeEnd = pad(I(0b001), 3);

    {
        Box<int> haloBox = makeHaloBox(nodeStart, nodeEnd, 1, 0, 0);
        Box<int> refBox{0, r+1, 0, r, 0, r};
        EXPECT_EQ(haloBox, refBox);
    }
    {
        Box<int> haloBox = makeHaloBox(nodeStart, nodeEnd, 0, 1, 0);
        Box<int> refBox{0, r, 0, r+1, 0, r};
        EXPECT_EQ(haloBox, refBox);
    }
    {
        Box<int> haloBox = makeHaloBox(nodeStart, nodeEnd, 0, 0, 1);
        Box<int> refBox{0, r, 0, r, 0, r+1};
        EXPECT_EQ(haloBox, refBox);
    }
}

TEST(BoxOverlap, makeHaloBoxUnderflow)
{
    makeHaloBoxUnderflow<unsigned>();
    makeHaloBoxUnderflow<uint64_t>();
}


//! \brief overflow check
template<class I>
void makeHaloBoxOverflow()
{
    int r = I(1) << (maxTreeLevel<I>{} - 1);
    // node range: [r,2r]^3
    I nodeStart = pad(I(0b111), 3);
    I nodeEnd   = nodeRange<I>(0);

    {
        Box<int> haloBox = makeHaloBox(nodeStart, nodeEnd, 1, 0, 0);
        Box<int> refBox{r-1, 2*r, r, 2*r, r, 2*r};
        EXPECT_EQ(haloBox, refBox);
    }
    {
        Box<int> haloBox = makeHaloBox(nodeStart, nodeEnd, 0, 1, 0);
        Box<int> refBox{r, 2*r, r-1, 2*r, r, 2*r};
        EXPECT_EQ(haloBox, refBox);
    }
    {
        Box<int> haloBox = makeHaloBox(nodeStart, nodeEnd, 0, 0, 1);
        Box<int> refBox{r, 2*r, r, 2*r, r-1, 2*r};
        EXPECT_EQ(haloBox, refBox);
    }
}

TEST(BoxOverlap, makeHaloBoxOverflow)
{
    makeHaloBoxOverflow<unsigned>();
    makeHaloBoxOverflow<uint64_t>();
}