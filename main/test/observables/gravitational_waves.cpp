/*
* MIT License
*
* Copyright (c) 2021 CSCS, ETH Zurich
*               2021 University of Basel
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in all
* copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
* SOFTWARE.
*/

/*!@brief small test for gravitational waves observable
 *
 * @author Lukas Schmidt
 */

#include <algorithm>
#include "gtest/gtest.h"
#include "cstone/domain/domain.hpp"
#include "observables/gravitational_waves.hpp"
#include "sph/particles_data.hpp"

using namespace sphexa;
using AccType = cstone::CpuTag;
using T = double;
using KeyType = uint64_t;
using Dataset = ParticlesData<T, KeyType, AccType>;



TEST(observables, grav)
{
    MPI_Init(NULL, NULL);
    int rank = 0, numRanks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    Dataset d;

    std::vector<T> x = {123.0,   234.0,  -345.0, 456.0,  -567.0};
    std::vector<T> y = {-678.0, 789.0,  -890.0, -901.0, 112.0};
    std::vector<T> z = {122.0,  -233.0, 543.0,  765.0,  -234.0};

    std::vector<T> vx = {723.0e-1,  244.0,    345.0e2,    406.0e-1,   -267.0e2};
    std::vector<T> vy = {-778.0,    489.0e-1,   -810.0,     900.0,      122.0};
    std::vector<T> vz = {-722.0e2,  -234.0,     -541.0e2,   705.0,      232.0e-1};

    std::vector<T> gradP_x = {-123.0e-3, 334.0,     345.0,      136.0e-3,  -167.0};
    std::vector<T> gradP_y = {-678.0e-3, -789.0,    -891.0e-3,  -901.0,    -132.0e-3};
    std::vector<T> gradP_z = {-122.0e-3, 333.0e-3,  543.0e-3,   -765.0,    214.0};
    std::vector<T> m = {1.0e5,  1.0e5,  1.0e5,  1.0e5,  1.0e5,};

    d.x = x;
    d.y = y;
    d.z = z;
    d.vx = vx;
    d.vy = vy;
    d.vz = vz;
    d.grad_P_x = gradP_x;
    d.grad_P_y = gradP_y;
    d.grad_P_z = gradP_z;
    d.m = m;

    T viewTheta = 0.545;
    T viewPhi   = 1.421;

    std::array<T, 8> test = gravRad(d, 0, 5 , viewTheta, viewPhi);

    EXPECT_NEAR(test[0], 2.69459728766579961e-58, 1e-72); //httplus
    EXPECT_EQ(test[1], -1.26588487735439141e-57);//httcross

    EXPECT_EQ(test[2], -2.89095364643866687e14);   //ixx
    EXPECT_EQ(test[3], -6.69346245332466625e14);   //iyy
    EXPECT_EQ(test[4], 9.58441609976333125e14);    //izz
    EXPECT_EQ(test[5], -6.17629053030000000e12);   //ixy
    EXPECT_EQ(test[6], -3.74431432361500000e14);   //ixz
    EXPECT_EQ(test[7], 2.01029844058000000e13);    //iyz

}