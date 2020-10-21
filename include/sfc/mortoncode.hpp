#pragma once

#include <array>
#include <cassert>

#include "box.hpp"
#include "clz.hpp"

namespace sphexa
{

//! \brief number of unused leading zeros in a 32-bit Morton code
template<class I>
struct unusedBits : std::integral_constant<unsigned, 2> {};

//! \brief number of unused leading zeros in a 64-bit Morton code
template<>
struct unusedBits<uint64_t> : std::integral_constant<unsigned, 1> {};

template<class I>
struct maxTreeLevel : std::integral_constant<unsigned, 10> {};

template<>
struct maxTreeLevel<uint64_t> : std::integral_constant<unsigned, 21> {};

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
    v &= 0x09249249u;
    v = (v ^ (v >>  2u)) & 0x030c30c3u;
    v = (v ^ (v >>  4u)) & 0x0300f00fu;
    v = (v ^ (v >>  8u)) & 0xff0000ffu;
    v = (v ^ (v >> 16u)) & 0x000003ffu;
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

template <class I, class T>
inline I toNBitInt(T x)
{
    // spatial resolution in bits per dimension
    constexpr unsigned nBits = (sizeof(I) * 8) / 3;

    // [0,1] to [0,1023] and convert to integer (32-bit) or
    // [0,1] to [0,2097151] and convert to integer (64-bit)
    return std::min(std::max(x * T(1u<<nBits), T(0.0)), T((1u<<nBits)-1u));
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
    assert(x >= 0.0 && x <= 1.0);
    assert(y >= 0.0 && y <= 1.0);
    assert(z >= 0.0 && z <= 1.0);

    // normalize floating point numbers
    I xi = detail::toNBitInt<I>(x);
    I yi = detail::toNBitInt<I>(y);
    I zi = detail::toNBitInt<I>(z);

    I xx = detail::expandBits(xi);
    I yy = detail::expandBits(yi);
    I zz = detail::expandBits(zi);

    // interleave the x, y, z components
    return xx * 4 + yy * 2 + zz;
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
    code = code >> discardedBits;
    return code << discardedBits;
}

/*! \brief Decode a morton code into x,y,z and convert coordinates
 *
 * \tparam I         32- or 64-bit unsigned integer
 * \param code       input Morton code
 * \param treeLevel  octree subdivision level
 * \return           array with x,y,z in the range [0, 2^treeLevel-1]
 */
template<class I>
std::array<unsigned, 3> boxFromCode(I code, unsigned treeLevel)
{
    // 10 or 21 bits per dimension
    constexpr unsigned nBits = (sizeof(I) * 8) / 3;

    // number of bits to discard
    unsigned discardedBits = nBits - treeLevel;

    code = enclosingBoxCode(code, treeLevel);

    I x = decodeMortonX(code);
    I y = decodeMortonY(code);
    I z = decodeMortonZ(code);

    return { unsigned(x >> discardedBits), unsigned(y >> discardedBits), unsigned(z >> discardedBits)} ;
}

/*! \brief Calculate the morton code corresponding to integer input box coordinates
 *         at a given tree subdivision level. Inverts boxCoordinates.
 *
 * \tparam I         32- or 64-bit unsigned integer
 * \param xyz        input integer box coordinates, must be in the range [0, 2^treeLevel-1]
 * \param treeLevel  octree subdivison level
 * \return           the morton code
 */
template<class I>
I codeFromBox(std::array<unsigned, 3> xyz, unsigned treeLevel)
{
    constexpr unsigned nBits = (sizeof(I) * 8) / 3;
    unsigned shifts = nBits - treeLevel;

    assert( xyz[0] < (1u << treeLevel));
    assert( xyz[1] < (1u << treeLevel));
    assert( xyz[2] < (1u << treeLevel));

    I xx = detail::expandBits(I(xyz[0]) << shifts);
    I yy = detail::expandBits(I(xyz[1]) << shifts);
    I zz = detail::expandBits(I(xyz[2]) << shifts);

    // interleave the x, y, z components
    return xx * 4 + yy * 2 + zz;
}

/*! \brief transfer a series of hierarchical octree indices into a morton code
 *
 * \tparam I       32- or 64-bit unsigned integer
 * \param indices  indices[0] contains the octree index 0-7 for the top-level,
 *                 indices[1] refers to the first subdivision, etc
 *                 a 32-bit integer can resolve up to 10 layers, while
 *                 a 64-bit integer can resolve 21 layers
 *
 *                 Note: all indices must be in the range [0-7]!
 *
 * \return         the morton code
 */
template<class I>
inline I codeFromIndices(std::array<unsigned char, 21> indices)
{
    constexpr unsigned nLevels = (sizeof(I) * 8) / 3;

    I ret = 0;
    for(unsigned idx = 0; idx < nLevels; ++idx)
    {
        unsigned treeLevel = nLevels - idx - 1;
        ret += I(indices[idx]) << (3*treeLevel);
    }

    return ret;
}

}

/*! \brief compute the maximum range of an octree node at a given subdivision level
 *
 * \tparam I         32- or 64-bit unsigned integer type
 * \param treeLevel  octree subdivision level
 * \return           the range
 *
 * At treeLevel 0, the range is the entire 30 or 63 bits used in the Morton code.
 * After that, the range decreases by 3 bits for each level.
 *
 */
template<class I>
inline std::enable_if_t<std::is_unsigned<I>{}, I>
nodeRange(unsigned treeLevel)
{
    // 10 or 21 bits per dimension
    constexpr unsigned nBits = (sizeof(I) * 8) / 3;

    I ret = I(1) << (3 * (nBits - treeLevel));
    return ret;
}

/*! \brief return octree subdivision level corresponding to codeRange
 *
 * \tparam I         32- or 64-bit unsigned integer type
 * \param codeRange  input Morton code range
 * \return           octree subdivision level 0-10 (32-bit) or 0-21 (64-bit)
 */
template<class I>
inline unsigned treeLevel(I codeRange)
{
    // check that codeRange is a power of 8
    assert( (countLeadingZeros(codeRange - 1) - unusedBits<I>{}) % 3 == 0);

    return (countLeadingZeros(codeRange - 1) - unusedBits<I>{}) / 3;
}

/*! \brief return the node index between 0-7 of the input code in the parent node
 *
 * \tparam I    32- or 64-bit unsigned integer type
 * \param code  input code corresponding to an octree node
 * \param level octree subdivision level to fully specify the octree node together with @a code
 * \return      the index between 0 and 7 that locates @a code in its enclosing parent node
 *              at level - 1. For the root node at level 0 which has no parent, the return value
 *              is 0.
 */
template<class I>
inline unsigned parentIndex(I code, unsigned level)
{
    return (code >> (3u*(maxTreeLevel<I>{} - level))) & 7u;
}

/*! \brief compute an enclosing envelope corresponding to the smallest possible
 *         octree node for two input Morton codes
 *
 * \tparam I              32- or 64-bit unsigned integer type
 * \param[in] firstCode   lower Morton code
 * \param[in] secondCode  upper Morton code
 *
 * \return                two morton codes that delineate the start and end of
 *                        the smallest octree node that contains both input codes
 */
template<class I>
inline std::tuple<I, I> smallestCommonBox(I firstCode, I secondCode)
{
    assert(firstCode <= secondCode);

    // XOR for superposition, followed by leading zero count
    // yields number of identical bits, counting from MSB to first differing bit
    unsigned commonBits = countLeadingZeros(firstCode ^ secondCode);

    unsigned commonLevel = (commonBits - unusedBits<I>{}) / 3;
    I        nodeStart   = detail::enclosingBoxCode(firstCode, commonLevel);

    return std::make_tuple(nodeStart, nodeStart + nodeRange<I>(commonLevel));
}

/*! \brief compute morton codes corresponding to neighboring octree nodes
 *         for a given input code and tree level
 *
 * \tparam I        32- or 64-bit unsigned integer type
 * \param code      input Morton code
 * \param treeLevel octree subdivision level, 0-10 for 32-bit, and 0-21 for 64-bit
 * \param dx        neighbor offset in x direction
 * \param dy        neighbor offset in y direction
 * \param dz        neighbor offset in z direction
 * \return          morton neighbor start code
 *
 * Note that the end of the neighbor range is given by the start code + nodeRange(treeLevel)
 */
template<class I>
inline std::enable_if_t<std::is_unsigned<I>{}, I>
mortonNeighbor(I code, unsigned treeLevel, int dx, int dy, int dz)
{
    // spatial resolution in bits per dimension
    constexpr unsigned nBits = (sizeof(I) * 8) / 3;
    // maximum coordinate value per dimension 2^nBits-1
    constexpr int maxCoord = int((1u << nBits) - 1u);

    unsigned shiftBits  = nBits - treeLevel;
    int shiftValue = int(1u << shiftBits);

    // zero out lower tree levels
    code = detail::enclosingBoxCode(code, treeLevel);

    int x = decodeMortonX(code);
    int y = decodeMortonY(code);
    int z = decodeMortonZ(code);

    // handle under and overflow (non-PBC)
    int newX = x + dx * shiftValue;
    x = (newX < 0 || newX > maxCoord) ? x : newX;
    int newY = y + dy * shiftValue;
    y = (newY < 0 || newY > maxCoord) ? y : newY;
    int newZ = z + dz * shiftValue;
    z = (newZ < 0 || newZ > maxCoord) ? z : newZ;

    return detail::expandBits(I(x)) * I(4)
         + detail::expandBits(I(y)) * I(2)
         + detail::expandBits(I(z));
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
                        const Box<T>& box)
{
    using detail::normalize;
    using Integer = std::decay_t<decltype(*codesBegin)>;

    while (xBegin != xEnd)
    {
        *codesBegin++ = morton3D<Integer>(normalize(*xBegin++, box.xmin(), box.xmax()),
                                          normalize(*yBegin++, box.ymin(), box.ymax()),
                                          normalize(*zBegin++, box.zmin(), box.zmax()));
    }
}

} // namespace sphexa
