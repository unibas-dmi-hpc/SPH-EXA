/*
 * MIT License
 *
 * Copyright (c) 2022 CSCS, ETH Zurich
 *               2022 University of Basel
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

/*! @file
 * @brief Tests for compile-time-string tuple getters
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include "gtest/gtest.h"

#include "cstone/util/value_list.hpp"

using namespace util;

/*

 // see comment in source code

TEST(ConstexprString, OperatorEq)
{
    constexpr StructuralString a("id");
    constexpr StructuralString b("id");

    static_assert(a == b);

    StructuralString c("id");
    StructuralString d("id");
    EXPECT_EQ(c, d);

    StructuralString e("id2");
    EXPECT_NE(d, e);
}

TEST(ValueList, element)
{
    using TestList = require_gcc_12::ValueList<1, 4, 2, 5, 3>;

    static_assert(require_gcc_12::ValueListElement<1, TestList>::value == 4);
}

TEST(ValueList, find)
{
    using TestList = require_gcc_12::ValueList<1, 4, 2, 5, 3>;

    static_assert(require_gcc_12::FindIndex<1, TestList>{} == 0);
    static_assert(require_gcc_12::FindIndex<4, TestList>{} == 1);
    static_assert(require_gcc_12::FindIndex<8, TestList>{} == 5);
}
*/

TEST(ValueList, nameGet)
{
    auto tup         = std::make_tuple(0, "alpha", 3.14);
    using FieldNames = FieldList<"id", "description", "number">;

    EXPECT_EQ(FieldListSize<FieldNames>{}, 3);

    {
        auto i = vl_detail::FindIndex<StructuralString("id"), FieldNames>{};
        EXPECT_EQ(i, 0);
    }
    {
        auto i = vl_detail::FindIndex<StructuralString("description"), FieldNames>{};
        EXPECT_EQ(i, 1);
    }
    {
        auto i = vl_detail::FindIndex<StructuralString("number"), FieldNames>{};
        EXPECT_EQ(i, 2);
    }

    auto f_id          = get<"id", FieldNames>(tup);
    auto f_description = get<"description", FieldNames>(tup);
    auto f_number      = get<"number", FieldNames>(tup);

    EXPECT_EQ(f_id, 0);
    EXPECT_EQ(f_description, "alpha");
    EXPECT_EQ(f_number, 3.14);
}

struct Testset
{
    inline static constexpr std::array fieldNames{"a", "b", "c"};
};

TEST(ValueList, MakeFieldList)
{
    auto tup = std::make_tuple(0, "alpha", 1.0);
    using FL = MakeFieldList<Testset>::Fields;
    auto e_a = util::get<"a", FL>(tup);
    EXPECT_EQ(e_a, 0);
}
