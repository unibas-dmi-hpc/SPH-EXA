#pragma once

#include <array>
#include <cassert>

namespace sphexa
{

namespace detail
{

//! \brief Expands a 10-bit integer into 30 bits by inserting 2 zeros after each bit.
inline unsigned expandBits(unsigned v)
{
    v &= 0x000003ffu; // discard bit higher 10
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

/*! \brief Compacts a 30-bit integer into 10 bits by selecting only bits divisible by 3
 *         this inverts expandBits
 */
inline unsigned compactBits(unsigned v)
{
    // Inverse of Part1By2 - "delete" all bits not at positions divisible by 3
    v &= 0x09249249u;                   // x = ---- 9--8 --7- -6-- 5--4 --3- -2-- 1--0
    v = (v ^ (v >>  2u)) & 0x030c30c3u; // x = ---- --98 ---- 76-- --54 ---- 32-- --10
    v = (v ^ (v >>  4u)) & 0x0300f00fu; // x = ---- --98 ---- ---- 7654 ---- ---- 3210
    v = (v ^ (v >>  8u)) & 0xff0000ffu; // x = ---- --98 ---- ---- ---- ---- 7654 3210
    v = (v ^ (v >> 16u)) & 0x000003ffu; // x = ---- ---- ---- ---- ---- --98 7654 3210
    return v;
}


//! \brief Expands a 21-bit integer into 63 bits by inserting 2 zeros after each bit.
inline std::size_t expandBits(std::size_t v)
{
    std::size_t x = v & 0x1fffffu; // discard bits higher 21
    x = (x | x << 32u) & 0x001f00000000fffflu;
    x = (x | x << 16u) & 0x001f0000ff0000fflu;
    x = (x | x << 8u)  & 0x100f00f00f00f00flu;
    x = (x | x << 4u)  & 0x10c30c30c30c30c3lu;
    x = (x | x << 2u)  & 0x1249249249249249lu;
    return x;
}

/*! \brief Compacts a 63-bit integer into 21 bits by selecting only bits divisible by 3
 *         this inverts expandBits
 */
inline std::size_t compactBits(std::size_t v)
{
    v &= 0x1249249249249249lu;
    v = (v ^ (v >>  2u)) & 0x10c30c30c30c30c3lu;
    v = (v ^ (v >>  4u)) & 0x100f00f00f00f00flu;
    v = (v ^ (v >>  8u)) & 0x001f0000ff0000fflu;
    v = (v ^ (v >> 16u)) & 0x001f00000000fffflu;
    v = (v ^ (v >> 32u)) & 0x00000000001ffffflu;
    return v;
}

/*! \brief Calculates a 30-bit Morton code for a 3D point
 *
 * \param[in] x,y,z input coordinates within the unit cube [0,1]^3
 */
template <class T>
unsigned int morton3D_(T x, T y, T z, [[maybe_unused]] unsigned tag)
{
    assert(x >= 0.0 && x <= 1.0);
    assert(y >= 0.0 && y <= 1.0);
    assert(z >= 0.0 && z <= 1.0);

    // normalize floating point numbers
    // 1024 = 2^10, so we map the floating point numbers
    // in [0,1] to [0,1023] and convert to integers
    x = std::min(std::max(x * T(1024.0), T(0.0)), T(1023.0));
    y = std::min(std::max(y * T(1024.0), T(0.0)), T(1023.0));
    z = std::min(std::max(z * T(1024.0), T(0.0)), T(1023.0));
    unsigned int xx = detail::expandBits((unsigned int)x);
    unsigned int yy = detail::expandBits((unsigned int)y);
    unsigned int zz = detail::expandBits((unsigned int)z);

    // interleave the x, y, z components
    return xx * 4 + yy * 2 + zz;
}

/*! \brief Calculates a 63-bit Morton code for a 3D point
 *
 * \param[in] x,y,z input coordinates within the unit cube [0,1]^3
 */
template <class T>
std::size_t morton3D_(T x, T y, T z, [[maybe_unused]] std::size_t tag)
{
    assert(x >= 0.0 && x <= 1.0);
    assert(y >= 0.0 && y <= 1.0);
    assert(z >= 0.0 && z <= 1.0);

    // normalize floating point numbers
    // 2097152 = 2^21, so we map the floating point numbers
    // in [0,1] to [0,2097152-1] and convert to integers
    x = std::min(std::max(x * T(2097152.0), T(0.0)), T(2097151.0));
    y = std::min(std::max(y * T(2097152.0), T(0.0)), T(2097151.0));
    z = std::min(std::max(z * T(2097152.0), T(0.0)), T(2097151.0));
    std::size_t xx = detail::expandBits((std::size_t)x);
    std::size_t yy = detail::expandBits((std::size_t)y);
    std::size_t zz = detail::expandBits((std::size_t)z);

    // interleave the x, y, z components
    return xx * 4 + yy * 2 + zz;
}

} // namespace detail

/*! \brief Calculates a Morton code for a 3D point
 *
 * \tparam I specify either a 32 or 64 bit unsigned integer to select
 *           the precision.
 *           Note: I needs to be specified explicitly.
 *           Note: not specifying an unsigned type results in a compilation error
 *
 * \param[in] x,y,z input coordinates within the unit cube [0,1]^3
 */
template <class I, class T>
inline std::enable_if_t<std::is_unsigned<I>{}, I> morton3D(T x, T y, T z)
{
    return detail::morton3D_(x,y,z, I{});
}

//! \brief extract X component from a morton code
template<class I>
inline std::enable_if_t<std::is_unsigned<I>{}, I> decodeMortonX(I code)
{
    return detail::compactBits(code >> 2);
}

//! \brief extract Y component from a morton code
template<class I>
inline std::enable_if_t<std::is_unsigned<I>{}, I> decodeMortonY(I code)
{
    return detail::compactBits(code >> 1);
}

//! \brief extract Z component from a morton code
template<class I>
inline std::enable_if_t<std::is_unsigned<I>{}, I> decodeMortonZ(I code)
{
    return detail::compactBits(code);
}

namespace detail {

//! \brief cut down the input morton code to the start code of the enclosing box at <treeLevel>
template<class I>
inline std::enable_if_t<std::is_unsigned<I>{}, I> enclosingBoxCode(I code, unsigned treeLevel)
{
    // total usable bits in the morton code, 30 or 63
    constexpr unsigned nBits = 3 * ((sizeof(I) * 8) / 3);

    // number of bits to discard, counting from lowest bit
    unsigned discardedBits = nBits - 3 * treeLevel;
    //return code & ( ~((I(1u)<<discardedBits)-I(1u)) );
    code = code >> discardedBits;
    return code << discardedBits;
}

}

template<class I>
inline std::enable_if_t<std::is_unsigned<I>{}, I>
mortonNeighbor(I code, unsigned treeLevel, int dx, int dy, int dz)
{
    // number of bits per dimension
    constexpr unsigned nBits = (sizeof(I) * 8) / 3;
    unsigned shiftLevel = nBits - treeLevel;

    // zero out lower tree levels
    code = detail::enclosingBoxCode(code, treeLevel);

    //using SignedInt = std::make_signed_t<I>;
    I x = decodeMortonX(code);
    I y = decodeMortonY(code);
    I z = decodeMortonZ(code);

    x += dx * (I(1) << shiftLevel);
    // prevent overflow (non-PBC)
    x = std::min(x, (I(1)<<nBits)-I(1));

    y += dy * (I(1) << shiftLevel);
    y = std::min(y, (I(1)<<nBits)-I(1));

    z += dz * (I(1) << shiftLevel);
    z = std::min(z, (I(1)<<nBits)-I(1));

    return detail::expandBits(x) * I(4)
         + detail::expandBits(y) * I(2)
         + detail::expandBits(z);
}


/*! \brief transfer a series of octree indices into a morton code
 *
 * \param indices indices[0] contains the octree index 0-7 for the top-level,
 *                 indices[1] reference to the first subdivision, etc
 *                 a 32-bit integer can resolve up to 10 layers
 * \return the morton code
 */
static unsigned mortonFromIndices(std::array<unsigned char, 10> indices)
{
    unsigned ret = 0;
    ret += indices[0] << 27u;
    ret += indices[1] << 24u;
    ret += indices[2] << 21u;
    ret += indices[3] << 18u;
    ret += indices[4] << 15u;
    ret += indices[5] << 12u;
    ret += indices[6] << 9u;
    ret += indices[7] << 6u;
    ret += indices[8] << 3u;
    ret += indices[9];

    return ret;
}

namespace detail {

template<class T>
static inline T normalize(T d, T min, T max) { return (d - min) / (max - min); }

} // namespace detail

/*! \brief compute the Morton codes for the input coordinate arrays
 *
 * \param[in]  [x,y,z][Begin, End] (const) input iterators for coordinate arrays
 * \param[out] order[Begin, End]  output for morton codes
 * \param[in]  [x,y,z][min, max]  coordinate bounding box
 */
template<class InputIterator, class OutputIterator, class T>
void computeMortonCodes(InputIterator  xBegin,
                        InputIterator  xEnd,
                        InputIterator  yBegin,
                        InputIterator  zBegin,
                        OutputIterator codesBegin,
                        T xmin, T xmax, T ymin, T ymax, T zmin, T zmax)
{
    using detail::normalize;
    using Integer = std::decay_t<decltype(*codesBegin)>;

    while (xBegin != xEnd)
    {
        *codesBegin++ = morton3D<Integer>(normalize(*xBegin++, xmin, xmax),
                                          normalize(*yBegin++, ymin, ymax),
                                          normalize(*zBegin++, zmin, zmax));
    }
}

} // namespace sphexa
