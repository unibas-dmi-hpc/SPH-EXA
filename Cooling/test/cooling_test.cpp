//
// Created by Noah Kubli on 28.02.22.
//
#include "gtest/gtest.h"
#include "cooling.hpp"
#include "sph/particles_data.hpp"
#include <iostream>
using namespace sphexa;

TEST(cooling_grackle, test1a)
{
#ifdef USE_CUDA
    using AccType = cstone::GpuTag;
#else
    using AccType = cstone::CpuTag;
#endif
    grackle_verbose = 0;
    initGrackle();

    using Real    = double;
    using KeyType = uint64_t;
    using Dataset = ParticlesData<Real, KeyType, AccType>;

    Dataset  d;
    double   dt          = 3.15e7 * 1e6; // / grackle_units.time_units;
    gr_float tiny_number = 1.e-20;

    d.dt  = std::vector<Real>{dt};
    d.x   = std::vector<Real>{0.};
    d.y   = std::vector<Real>{0.};
    d.z   = std::vector<Real>{0.};
    d.rho = std::vector<Real>{1.0};
#define mh 1.67262171e-24
#define kboltz 1.3806504e-16
    double temperature_units =
        mh * pow(grackle_units.a_units * grackle_units.length_units / grackle_units.time_units, 2.) / kboltz;
    d.u              = std::vector<Real>{1000. / temperature_units};
    d.HI_fraction    = std::vector<Real>{0.76};
    d.HII_fraction   = std::vector<Real>{tiny_number};
    d.HeI_fraction   = std::vector<Real>{0.24};
    d.HeII_fraction  = std::vector<Real>{tiny_number};
    d.HeIII_fraction = std::vector<Real>{tiny_number};
    d.e_fraction     = std::vector<Real>{tiny_number};
    d.HM_fraction    = std::vector<Real>{tiny_number};
    d.H2I_fraction   = std::vector<Real>{tiny_number};
    d.H2II_fraction  = std::vector<Real>{tiny_number};
    d.DI_fraction    = std::vector<Real>{2.0 * 3.4e-5};
    d.DII_fraction   = std::vector<Real>{tiny_number};
    d.HDI_fraction   = std::vector<Real>{tiny_number};
    d.metal_fraction = std::vector<Real>{0.01295};
    d.n              = 1;

    std::cout << d.HI_fraction[0] << std::endl;
    std::cout << d.HeI_fraction[0] << std::endl;
    std::cout << d.metal_fraction[0] << std::endl;

    cool_particle<Real, Dataset>(d.dt[0],
                                 d.rho[0],
                                 d.u[0],
                                 d.HI_fraction[0],
                                 d.HII_fraction[0],
                                 d.HeI_fraction[0],
                                 d.HeII_fraction[0],
                                 d.HeIII_fraction[0],
                                 d.e_fraction[0],
                                 d.HM_fraction[0],
                                 d.H2I_fraction[0],
                                 d.H2II_fraction[0],
                                 d.DI_fraction[0],
                                 d.DII_fraction[0],
                                 d.HDI_fraction[0],
                                 d.metal_fraction[0]
    );

    std::cout << d.HI_fraction[0] << std::endl;

    EXPECT_NEAR(d.HI_fraction[0], 0.630705, 1e-6);
    EXPECT_NEAR(d.u[0], 2.95159e+35, 1e30);

    cleanGrackle();
}