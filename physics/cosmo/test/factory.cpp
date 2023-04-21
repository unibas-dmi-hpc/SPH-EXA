#include "gtest/gtest.h"

#include "cosmo/cosmology_data.hpp"
#include "cosmo/factory.hpp"

TEST(cosmo, factory)
{
    using T = double; 
    using namespace cosmo;

    EXPECT_THROW(cosmologyFactory<T>(std::string("a-file-name")), std::runtime_error);
    EXPECT_THROW(cosmologyFactory<T>("a-file-name"), std::runtime_error);

    EXPECT_NE(dynamic_cast<StaticUniverse<T>*>(cosmologyFactory<T>(0,0,0,0).get()), nullptr);
    EXPECT_NE(dynamic_cast<CDM<T>*>(cosmologyFactory<T>(1,1).get()), nullptr);
    EXPECT_NE(dynamic_cast<LambdaCDM<T>*>(cosmologyFactory<T>(1,1,1).get()), nullptr);
    EXPECT_NE(dynamic_cast<LambdaCDM<T>*>(cosmologyFactory<T>(LambdaCDM<T>::Planck2018).get()), nullptr);

    EXPECT_THROW(cosmologyFactory<T>(0,1), std::domain_error);
    EXPECT_THROW(cosmologyFactory<T>(-1,0,0,0), std::domain_error);

}

