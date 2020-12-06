#include <algorithm>
#include <numeric>

#include "gtest/gtest.h"

#include "sfc/domaindecomp.hpp"
#include "coord_samples/random.hpp"


TEST(DomainDecomposition, singleRangeSfcSplit)
{
    using CodeType = unsigned;
    {
        int nSplits = 2;
        std::vector<std::size_t> counts{5, 5, 5, 5, 5, 6};
        std::vector<CodeType>    tree{0, 1, 2, 3, 4, 5, 6};

        auto splits = sphexa::singleRangeSfcSplit(tree, counts, nSplits);

        sphexa::SpaceCurveAssignment<CodeType> ref(nSplits);
        ref[0].addRange(0,3,15);
        ref[1].addRange(3,6,16);
        EXPECT_EQ(ref, splits);
    }
    {
        int nSplits = 2;
        std::vector<std::size_t> counts{5, 5, 5, 15, 1, 0};
        std::vector<CodeType>    tree{0, 1, 2, 3, 4, 5, 6};

        auto splits = sphexa::singleRangeSfcSplit(tree, counts, nSplits);

        sphexa::SpaceCurveAssignment<CodeType> ref(nSplits);
        ref[0].addRange(0,3,15);
        ref[1].addRange(3,6,16);
        EXPECT_EQ(ref, splits);
    }
    {
        int nSplits = 2;
        std::vector<std::size_t> counts{15, 0, 1, 5, 5, 5};
        std::vector<CodeType>    tree{0, 1, 2, 3, 4, 5, 6};

        auto splits = sphexa::singleRangeSfcSplit(tree, counts, nSplits);

        sphexa::SpaceCurveAssignment<CodeType> ref(nSplits);
        ref[0].addRange(0,3,16);
        ref[1].addRange(3,6,15);
        EXPECT_EQ(ref, splits);
    }
    {
        int nSplits = 7;
        std::vector<std::size_t> counts{4, 3, 4, 3, 4, 3, 4, 3, 4, 3};
        // should be grouped |4|7|3|7|4|7|3|
        std::vector<CodeType> tree{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

        auto splits = sphexa::singleRangeSfcSplit(tree, counts, nSplits);

        sphexa::SpaceCurveAssignment<CodeType> ref(nSplits);
        ref[0].addRange(0,1,4);
        ref[1].addRange(1,3,7);
        ref[2].addRange(3,4,3);
        ref[3].addRange(4,6,7);
        ref[4].addRange(6,7,4);
        ref[5].addRange(7,9,7);
        ref[6].addRange(9,10,3);
        EXPECT_EQ(ref, splits);
    }
}


/*! \brief test SendList creation from a SFC assignment
 *
 * This test creates an array with Morton codes and an
 * SFC assignment with Morton code ranges.
 * CreateSendList then translated the code ranges into indices
 * valid for the Morton code array.
 */
template<class I>
void createSendList()
{
    int nParticles = 10;
    std::vector<I> codes(nParticles);
    std::iota(begin(codes), end(codes), 10);

    int nRanks = 2;
    sphexa::SpaceCurveAssignment<I> assignment(nRanks);
    assignment[0].addRange(9, 11, 1);   // range lower than lowest code
    assignment[0].addRange(13, 15, 2);
    assignment[1].addRange(17, 1000, 2); // range bigger than highest code

    // note: codes input needs to be sorted
    auto sendList = sphexa::createSendList(assignment, codes);

    EXPECT_EQ(sendList[0].totalCount(), 3);
    EXPECT_EQ(sendList[1].totalCount(), 3);

    sphexa::SendList refSendList(nRanks);
    refSendList[0].addRange(0, 1, 1);
    refSendList[0].addRange(3, 5, 2);
    refSendList[1].addRange(7, 10, 3);

    EXPECT_EQ(refSendList, sendList);
}

TEST(DomainDecomposition, createSendList)
{
    createSendList<unsigned>();
    createSendList<uint64_t>();
}

/*! \brief This test integrates octree generation, SFC assignment and SendList creation
 *
 * Test procedure:
 *
 * 1. create nParticles random gaussian distributed x,y,z coordinates in a box
 * 2. create (local) sfc-octree
 * 3. create sfc assignment based on octree from 2.
 * 4. create sendList from assignment
 *
 * Expected results:
 *
 * 1. assignment contains nSplit SFC ranges which all contain about nParticles/nSplit +- bucketSize particles
 * 2. each particle appears in the SendList, i.e. each particle did get assigned to some rank
 */
template<class I>
void assignSendRandomData()
{
    int nParticles = 1003;
    int bucketSize = 64;
    RandomGaussianCoordinates<double, I> coords(nParticles, {-1,1});

    auto [tree, counts] = sphexa::computeOctree(coords.mortonCodes().data(),
                                                coords.mortonCodes().data() + nParticles,
                                                bucketSize);

    int nSplits = 4;
    auto assignment = sphexa::singleRangeSfcSplit(tree, counts, nSplits);

    /// all splits except the last one should at least be assigned nParticles/nSplits
    for (int rank = 0; rank < nSplits; ++rank)
    {
        std::size_t rankCount = assignment[rank].totalCount();

        /// particles in each rank should be within avg per rank +- bucketCount
        EXPECT_LE(nParticles/nSplits - bucketSize, rankCount);
        EXPECT_LE(rankCount, nParticles/nSplits + bucketSize);
    }

    auto sendList = sphexa::createSendList(assignment, coords.mortonCodes());

    int particleRecount = 0;
    for (auto& manifest : sendList)
        for (int rangeIndex = 0; rangeIndex < manifest.nRanges(); ++rangeIndex)
            particleRecount += manifest.rangeEnd(rangeIndex) - manifest.rangeStart(rangeIndex);

    /// make sure that all particles present on the node got assigned to some rank
    EXPECT_EQ(nParticles, particleRecount);
}

TEST(DomainDecomposition, assignSendIntegration)
{
    assignSendRandomData<unsigned>();
    assignSendRandomData<uint64_t>();
}

/*! \brief Test that createSendBuffer can create the correct buffer from a source array
 *
 * @tparam I 32- or 64-bit unsigned integer
 *
 * Precondition: an array, corresponding ordering and SendManifest with valid index ranges into
 *               the array
 *
 * Expected Result: createSendBuffer extracts the elements that fall within the index ranges in the
 *                  sendManifest into the output buffer
 */
template<class I>
void createSendBuffer()
{
    int bufferSize = 64;
    // the source array from which to extract the buffer
    std::vector<double> x(bufferSize);
    std::iota(begin(x), end(x), 0);

    sphexa::SendManifest manifest;
    manifest.addRange(0, 8, 8);
    manifest.addRange(40, 42, 2);
    manifest.addRange(50, 50, 0);

    std::vector<int> ordering(bufferSize);
    std::iota(begin(ordering), end(ordering), 0);

    // non-default ordering will make x appear sorted despite two elements being swapped
    std::swap(x[0], x[1]);
    std::swap(ordering[0], ordering[1]);

    auto buffer = sphexa::createSendBuffer(manifest, x, ordering);

    // note sorted reference
    std::vector<double> ref{0,1,2,3,4,5,6,7,40,41};
    EXPECT_EQ(buffer, ref);
}

TEST(DomainDecomposition, createSendBuffer)
{
    createSendBuffer<unsigned>();
    createSendBuffer<uint64_t>();
}


TEST(DomainDecomposition, binX)
{
    using CodeType = unsigned;
    std::vector<CodeType> codes;

    constexpr unsigned nBins = 1u<<sphexa::maxTreeLevel<CodeType>{};

    for (unsigned i = 0; i < nBins/2; ++i)
    {
        codes.push_back(sphexa::detail::codeFromBox<CodeType>({i,0,0}, sphexa::maxTreeLevel<CodeType>{}));
        codes.push_back(sphexa::detail::codeFromBox<CodeType>({i,0,1}, sphexa::maxTreeLevel<CodeType>{}));
        codes.push_back(sphexa::detail::codeFromBox<CodeType>({i+512,0,1}, sphexa::maxTreeLevel<CodeType>{}));
        codes.push_back(sphexa::detail::codeFromBox<CodeType>({i+512,1,1}, sphexa::maxTreeLevel<CodeType>{}));
    }

    std::array<unsigned, nBins> xCounts{};

    std::sort(codes.data(), codes.data()+codes.size(), sphexa::CompareX<CodeType>{});
    sphexa::histogramX(codes.data(), codes.data()+codes.size(), xCounts);

    for (auto c : xCounts)
    {
        EXPECT_EQ(c, 2);
    }
}