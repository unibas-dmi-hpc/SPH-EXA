#include <algorithm>
#include <numeric>

#include "gtest/gtest.h"


/*! \brief particle count histograms along x,y,z dimensions
 *
 * This code can be used to compute histograms of particle counts
 * along x,y,z dimensions. I.e. for a 32-bit Morton SFC octree, there
 * are 2^10=1024 bins along each dimension. Computing the particle
 * counts for those bins can be used to decompose a domain into
 * n_x * n_y * n_z subdomains, with n_x * n_y * n_z being a factorization
 * of the total rank count and n_i splits along dimension i.
 *
 * This generally leads to fewer halos than assigning a single Morton code
 * range along the SFC to each rank, because the SFC has spatial discontinuities.
 * It however leads to more octree nodes being split across domains which is a
 * disadvantage for FMM schemes.
 *
 * Since this code is not currently used, it lives in the testing space for now.
 */

#include "sfc/mortoncode.hpp"

namespace sphexa
{

template <class I>
struct CompareX
{
    inline bool operator()(I a, I b) { return decodeMortonX(a) < decodeMortonX(b); }
};

template <class I>
void histogramX(const I *codesStart, const I *codesEnd, std::array<unsigned, 1u << maxTreeLevel<I>{}> &histogram)
{
    constexpr int nBins = 1u << maxTreeLevel<I>{};

    for (int bin = 0; bin < nBins; ++bin)
    {
        auto lower = std::lower_bound(codesStart, codesEnd, bin, CompareX<I>{});
        auto upper = std::upper_bound(codesStart, codesEnd, bin, CompareX<I>{});
        histogram[bin] = upper - lower;
    }
}

} // namespace sphexa


TEST(DomainDecomposition, binX)
{
    using CodeType = unsigned;
    std::vector<CodeType> codes;

    constexpr unsigned nBins = 1u<<sphexa::maxTreeLevel<CodeType>{};

    for (unsigned i = 0; i < nBins/2; ++i)
    {
        codes.push_back(sphexa::codeFromBox<CodeType>(i,0,0, sphexa::maxTreeLevel<CodeType>{}));
        codes.push_back(sphexa::codeFromBox<CodeType>(i,0,1, sphexa::maxTreeLevel<CodeType>{}));
        codes.push_back(sphexa::codeFromBox<CodeType>(i+512,0,1, sphexa::maxTreeLevel<CodeType>{}));
        codes.push_back(sphexa::codeFromBox<CodeType>(i+512,1,1, sphexa::maxTreeLevel<CodeType>{}));
    }

    std::array<unsigned, nBins> xCounts{};

    std::sort(codes.data(), codes.data()+codes.size(), sphexa::CompareX<CodeType>{});
    sphexa::histogramX(codes.data(), codes.data()+codes.size(), xCounts);

    for (auto c : xCounts)
    {
        EXPECT_EQ(c, 2);
    }
}