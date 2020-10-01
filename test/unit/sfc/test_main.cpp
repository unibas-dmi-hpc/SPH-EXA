#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include "gtest/gtest.h"

int main(int argc, char **argv) {

  ::testing::InitGoogleTest(&argc, argv);
  auto ret = RUN_ALL_TESTS();
  return ret;
}
