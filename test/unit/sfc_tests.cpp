#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "sfc.hpp"


TEST(SFC, mortonIndex) {

    double x = 0.1;
    double y = 0.2;
    double z = 0.3;

    EXPECT_EQ(0, sphexa::morton3D(x, y, z));
}

int main(int argc, char **argv) {

  ::testing::InitGoogleTest(&argc, argv);
  auto ret = RUN_ALL_TESTS();
  return ret;
}
