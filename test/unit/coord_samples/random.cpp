
#include <gtest/gtest.h>

#include "random.hpp"

TEST(RandomCoordinates, coordinateContainerIsSorted)
{
    using real = double;
    using CodeType = unsigned;
    int n = 10;

    sphexa::Box<real> box{ 0, 1, -1, 2, 0, 5 };
    RandomCoordinates<real, CodeType> c(n, box);

    std::vector<CodeType> testCodes(n);
    sphexa::computeMortonCodes(begin(c.x()), end(c.x()), begin(c.y()), begin(c.z()),
                               begin(testCodes), box);

    EXPECT_EQ(testCodes, c.mortonCodes());

    std::vector<CodeType> testCodesSorted = testCodes;
    std::sort(begin(testCodesSorted), end(testCodesSorted));

    EXPECT_EQ(testCodes, testCodesSorted);
}
