#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "sfc.hpp"


TEST(SFC, mortonIndex) {

    double x = 0.00489; // 5 on the scale 0-1023
    double y = 0.00196; // 2
    double z = 0.00391; // 4

    // binary representation:
    // x = 101
    // y = 010
    // z = 100
    // Morton code is 101010100 = 340

    EXPECT_EQ(340, sphexa::morton3D(float(x), float(y), float(z)));
    EXPECT_EQ(340, sphexa::morton3D(x, y, z));
}

int main(int argc, char **argv) {

  ::testing::InitGoogleTest(&argc, argv);
  auto ret = RUN_ALL_TESTS();
  return ret;
}
