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

#include "gtest/gtest.h"
#include "observables/grav_waves_calculations.hpp"

using namespace sphexa;
using T = double;

TEST(grav_observable, quadpoleMomentum)
{

    std::vector<T> x = {123.0,   234.0,  -345.0, 456.0,  -567.0};
    std::vector<T> y = {-678.0, 789.0,  -890.0, -901.0, 112.0};
    std::vector<T> z = {122.0,  -233.0, 543.0,  765.0,  -234.0};

    std::vector<T> vx = {723.0e-1,  244.0,    345.0e2,    406.0e-1,   -267.0e2};
    std::vector<T> vy = {-778.0,    489.0e-1,   -810.0,     900.0,      122.0};
    std::vector<T> vz = {-722.0e2,  -234.0,     -541.0e2,   705.0,      232.0e-1};

    std::vector<T> ax = {123.0e-3, -334.0,     -345.0,      -136.0e-3,  167.0};
    std::vector<T> ay = {678.0e-3, 789.0,    891.0e-3,  901.0,    132.0e-3};
    std::vector<T> az = {122.0e-3, -333.0e-3,  -543.0e-3,   765.0,    -214.0};
    std::vector<T> m = {1.0e5,  1.0e5,  1.0e5,  1.0e5,  1.0e5,};

    T ixx = d2QuadpoleMomentum<T>(0, 5, 0, 0, x.data(), y.data(), z.data(), vx.data(), vy.data(), vz.data(),
                               ax.data(), ay.data(), az.data(), m.data());
    T iyy = d2QuadpoleMomentum<T>(0, 5, 1, 1, x.data(), y.data(), z.data(), vx.data(), vy.data(), vz.data(),
                                  ax.data(), ay.data(), az.data(), m.data());
    T izz = d2QuadpoleMomentum<T>(0, 5, 2, 2, x.data(), y.data(), z.data(), vx.data(), vy.data(), vz.data(),
                                  ax.data(), ay.data(), az.data(), m.data());
    T ixy = d2QuadpoleMomentum<T>(0, 5, 0, 1, x.data(), y.data(), z.data(), vx.data(), vy.data(), vz.data(),
                                  ax.data(), ay.data(), az.data(), m.data());
    T ixz = d2QuadpoleMomentum<T>(0, 5, 0, 2, x.data(), y.data(), z.data(), vx.data(), vy.data(), vz.data(),
                                  ax.data(), ay.data(), az.data(), m.data());
    T iyz = d2QuadpoleMomentum<T>(0, 5, 1, 2, x.data(), y.data(), z.data(), vx.data(), vy.data(), vz.data(),
                          ax.data(), ay.data(), az.data(), m.data());


    EXPECT_EQ(ixx, -2.89095364643866687e14);   //ixx
    EXPECT_EQ(iyy, -6.69346245332466625e14);   //iyy
    EXPECT_EQ(izz, 9.58441609976333125e14);    //izz
    EXPECT_EQ(ixy, -6.17629053030000000e12);   //ixy
    EXPECT_EQ(ixz, -3.74431432361500000e14);   //ixz
    EXPECT_EQ(iyz, 2.01029844058000000e13);    //iyz

}
TEST(grav_observable, httcalc)
{
    T viewTheta = 0.545;
    T viewPhi = 1.421;
    T httplus;
    T httcross;
    std::array<T, 6> quadpole = {-2.89095364643866687e14, -6.69346245332466625e14, 9.58441609976333125e14,
                                 -6.17629053030000000e12, -3.74431432361500000e14, 2.01029844058000000e13};
    sphexa::computeHtt(quadpole, viewTheta, viewPhi, &httplus, &httcross);

    EXPECT_NEAR(2.69459728766579961E-58, httplus, 1e-72);
    EXPECT_NEAR(-1.26588487735439141E-57, httcross, 1e-72);
}