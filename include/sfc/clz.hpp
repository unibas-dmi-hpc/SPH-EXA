#pragma once

#include <vector>

namespace detail
{

static int clz32(uint32_t x)
{
    static const int debruijn32[32] = {0, 31, 9, 30, 3, 8,  13, 29, 2,  5,  7,  21, 12, 24, 28, 19,
                                       1, 10, 4, 14, 6, 22, 25, 20, 11, 15, 23, 26, 16, 27, 17, 18};
    x |= x >> 1u;
    x |= x >> 2u;
    x |= x >> 4u;
    x |= x >> 8u;
    x |= x >> 16u;
    x++;
    return debruijn32[x * 0x076be629 >> 27];
}

static int clz64(uint64_t x)
{
    static const int debruijn64[64] = {0,  47, 1,  56, 48, 27, 2,  60, 57, 49, 41, 37, 28, 16, 3,  61, 54, 58, 35, 52, 50, 42,
                                       21, 44, 38, 32, 29, 23, 17, 11, 4,  62, 46, 55, 26, 59, 40, 36, 15, 53, 34, 51, 20, 43,
                                       31, 22, 10, 45, 25, 39, 14, 33, 19, 30, 9,  24, 13, 18, 8,  12, 7,  6,  5,  63};

    x |= x >> 1u;
    x |= x >> 2u;
    x |= x >> 4u;
    x |= x >> 8u;
    x |= x >> 16u;
    x |= x >> 32u;

    return 63 - debruijn64[x * 0x03f79d71b4cb0a89ul >> 58u];
}

} // namespace detail

template<class I>
int clz(I x)
{
    // with GCC and clang, we can use the builtin implementation
    // this also works with the intel compiler, which also defines __GNUC__
#if defined(__GNUC__) || defined(__clang__)

    // if the target architecture is Haswell or later,
    // __builtin_clz(l) is implemented with the LZCNT instruction
    // which returns the number of bits for an input of zero,
    // so this check is not required in that case (flag: -march=haswell)
    if (x == 0) return sizeof(I) * 8;

    if constexpr (sizeof(I) == 8)
    {
        return __builtin_clzl(x);
    }
    else
    {
        return __builtin_clz(x);
    }
#else
    if (x == 0) return sizeof(I) * 8;

    if constexpr (sizeof(I) == 8)
    {
        return detail::clz64(x);
    }
    else
    {
        return detail::clz32(x);
    }
#endif
}