#include "cosmo/utils.hpp"
#include <cassert>
#include <iostream>
#include <vector>
#include <cmath>
#include "gtest/gtest.h"
#include "gtest/gtest-spi.h"
#include <csignal>

TEST(cosmo, romberg)
{
    using namespace cosmo;
    using T = double; 

    EXPECT_NEAR(romberg<T>([]([[maybe_unused]] T x){ return 1.0; }, 0.0, 10.0, 1e-7), 10.0, 1e-7);
    EXPECT_NEAR(romberg<T>([]([[maybe_unused]] T x){ return 1.0; }, -10, 10.0, 1e-7), 20.0, 1e-7);
    EXPECT_NEAR(romberg<T>([](T x){ return   x; }, -10, 10.0, 1e-8),  0.0, 1e-7);
    EXPECT_NEAR(romberg<T>([](T x){ return   x*x; }, -1, 1.0, 1e-7), 2.0/3.0, 1e-7);
    EXPECT_NEAR(romberg<T>([](T x){ return 1/x; }, 1, 2, 1e-7),  log(2), 1e-7);
    EXPECT_NEAR(romberg<T>([](T x){ return 1/(x*x); }, 0.01, 1, 1e-7),  99.0, 1e-7);
    EXPECT_NEAR(romberg<T>([](T x){ return 1/(x*x); }, 0.01, 9, 1e-7),  99.88888888, 1e-7);

    EXPECT_NONFATAL_FAILURE(EXPECT_NEAR(romberg<T>([](T x){ return 1/x; }, 1, 2, 1e-4),  log(2), 1e-12), "");
    EXPECT_NONFATAL_FAILURE(EXPECT_NEAR(romberg<T>([](T x){ return 1/x; }, 1, 2, 1e-5),  log(2), 1e-12), "");
    EXPECT_NONFATAL_FAILURE(EXPECT_NEAR(romberg<T>([](T x){ return 1/x; }, 1, 2, 1e-8),  log(2), 1e-15), "");

}

TEST(cosmo, newton)
{
    using namespace cosmo;
    using T = double; 

    {
    auto f  = [](const T x){ return x*x - 2.0; };
    auto df = [](const T x){ return 2.0*x; };
    EXPECT_THROW(newton<T>(f, df,  1.0, 10.0, -10.0, 1e-7), std::invalid_argument);
    }

    {
    auto f  = [](const T x){ return x*x; };
    auto df = [](const T x){ return 2.0*x; };

    EXPECT_EQ(newton<T>(f, df,  0.0, -10.0, 10.0), 0);
    }

    {
    auto f  = [](const T x){ return x*x - 2.0; };
    auto df = [](const T x){ return 2.0*x; };

    EXPECT_NEAR(newton<T>(f, df,  1.0, -10.0, 10.0, 1e-7), sqrt(2), 1e-7);
    EXPECT_NEAR(newton<T>(f, df,  1.0, -10.0, 10.0, 1e-7), sqrt(2), 1e-7);
    EXPECT_NEAR(newton<T>(f, df, -2.0, -10.0, 10.0, 1e-7), -sqrt(2), 1e-7);
    EXPECT_NEAR(newton<T>(f, df, 0, 0, 10, 1e-7), sqrt(2), 1e-7);
    EXPECT_NEAR(newton<T>(f, df, 0, -10, 10, 1e-7), sqrt(2), 1e-7);
    }

    {
    auto f  = [](const T x){ return x - 2 + log(x); };
    auto df = [](const T x){ return 1 + 1/x; };
    EXPECT_NEAR(newton<T>(f, df,  1.5, -10, 10, 1e-7), 1.5571455989976113, 1e-7);
    }

    {
    auto f  = [](const T x){ return sin(x) - (x+1)/(x-1); };
    auto df = [](const T x){ return cos(x) + 2/((x-1)*(x-1)); };
    EXPECT_NEAR(newton<T>(f, df,  -0.2, -10, 10, 1e-7), -0.42036240721563467, 1e-7);
    }

}

