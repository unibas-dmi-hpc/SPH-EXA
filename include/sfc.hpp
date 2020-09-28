#pragma once

namespace sphexa
{

namespace detail {

// Expands a 10-bit integer into 30 bits
// by inserting 2 zeros after each bit.
unsigned int expandBits(unsigned int v)
{
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

}

// Calculates a 30-bit Morton code for the
// given 3D point located within the unit cube [0,1].
unsigned int morton3D(double x, double y, double z)
{
    // normalize floating point numbers
    // 1024 = 2^10, so we can take the
    // 10 most significant bits from the mantissa
    x = std::min(std::max(x * 1024.0, 0.0), 1023.0);
    y = std::min(std::max(y * 1024.0, 0.0), 1023.0);
    z = std::min(std::max(z * 1024.0, 0.0), 1023.0);
    unsigned int xx = detail::expandBits((unsigned int)x);
    unsigned int yy = detail::expandBits((unsigned int)y);
    unsigned int zz = detail::expandBits((unsigned int)z);
    return xx * 4 + yy * 2 + zz;
}

}
