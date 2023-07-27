
/*! @file
 * @brief Test resizing of domain
 *
 * @author Noah Kubli <noah.kubli@uzh.ch>
 */

#include <mpi.h>

#include "cstone/domain/domain.hpp"
#include "cstone/tree/cs_util.hpp"
#include <gtest/gtest.h>

using namespace cstone;

TEST(GlobalDomainResize, resize)
{
    using T       = double;
    using KeyType = unsigned;

    int rank = 0, numRanks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    const int thisExampleRanks = 2;
    if (numRanks != thisExampleRanks) throw std::runtime_error("this test needs 2 ranks\n");
    int bucketSize = 1;
    float theta    = 1.0;
    Domain<KeyType, T> domain(rank, numRanks, bucketSize, bucketSize, theta);

    std::vector<T> x{0.55};
    std::vector<T> y{0.55};
    std::vector<T> z{0.55};
    std::vector<T> h{0.05};

    std::vector<KeyType> keys(x.size());
    std::vector<T> s1, s2, s3;

    domain.sync(keys, x, y, z, h, std::tuple{}, std::tie(s1, s2, s3));

    if (rank == 0)
    {
        EXPECT_EQ(domain.startIndex(), 0);
        EXPECT_EQ(domain.endIndex(), 2);
        EXPECT_EQ(domain.nParticlesWithHalos(), 2);
    }
    else if (rank == 1)
    {
        EXPECT_EQ(domain.startIndex(), 0);
        EXPECT_EQ(domain.endIndex(), 0);
        EXPECT_EQ(domain.nParticlesWithHalos(), 0);
    }

    if (rank == 0)
    {
        // Add one particle
        int n_new = 1;
        // If halos are large enough, no resizing required
        int add_size = n_new - int(domain.endIndex()) + int(x.size());
        // Resize
        if (add_size > 0)
        {
            size_t new_size = std::max(domain.nParticlesWithHalos(), (unsigned int)(x.size() + add_size));
            x.resize(new_size);
            y.resize(new_size);
            z.resize(new_size);
            h.resize(new_size);
            keys.resize(new_size);
            std::cout << "Rank " << rank << " resize done, new size: " << new_size << std::endl;
        }
        unsigned endBefore = domain.endIndex();
        domain.setEndIndex(domain.endIndex() + n_new);
        // Add new particle
        x.at(endBefore)           = x.at(domain.startIndex());
        y.at(endBefore)           = y.at(domain.startIndex());
        z.at(endBefore)           = z.at(domain.startIndex());
        h.at(endBefore)           = h.at(domain.startIndex());
        x.at(domain.startIndex()) = 0.59;
        y.at(domain.startIndex()) = 0.59;
        z.at(domain.startIndex()) = 0.59;
        h.at(domain.startIndex()) = 0.01;
    }

    domain.sync(keys, x, y, z, h, std::tuple{}, std::tie(s1, s2, s3));

    if (rank == 0)
    {
        EXPECT_EQ(domain.startIndex(), 0);
        EXPECT_EQ(domain.endIndex(), 2);
        EXPECT_EQ(domain.nParticlesWithHalos(), 3);
    }
    else if (rank == 1)
    {
        EXPECT_EQ(domain.startIndex(), 2);
        EXPECT_EQ(domain.endIndex(), 3);
        EXPECT_EQ(domain.nParticlesWithHalos(), 3);
    }
}
