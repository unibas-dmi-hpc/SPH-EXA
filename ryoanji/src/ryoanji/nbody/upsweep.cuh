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

/*! @file
 * @brief  Upsweep for multipole and source center computation
 *
 * @author Rio Yokota <rioyokota@gsic.titech.ac.jp>
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <chrono>

#include "types.h"

namespace ryoanji
{

template<class Tc, class Tm, class Tf, class MType>
void computeLeafMultipoles(const Tc* x, const Tc* y, const Tc* z, const Tm* m, const TreeNodeIndex* leafToInternal,
                           TreeNodeIndex numLeaves, const LocalIndex* layout, const Vec4<Tf>* centers,
                           MType* multipoles);

#define RYOANJI_LEAF_MULTIPOLES(Tc, Tm, Tf, MType)                                                                     \
    void computeLeafMultipoles(const Tc*            x,                                                                 \
                               const Tc*            y,                                                                 \
                               const Tc*            z,                                                                 \
                               const Tm*            m,                                                                 \
                               const TreeNodeIndex* leafToInternal,                                                    \
                               TreeNodeIndex        numLeaves,                                                         \
                               const LocalIndex*    layout,                                                            \
                               const Vec4<Tf>*      centers,                                                           \
                               MType*               multipoles)

extern template RYOANJI_LEAF_MULTIPOLES(double, double, double, SphericalHexadecapole<double>);
extern template RYOANJI_LEAF_MULTIPOLES(double, float, double, SphericalHexadecapole<double>);
extern template RYOANJI_LEAF_MULTIPOLES(double, float, double, SphericalHexadecapole<float>);
extern template RYOANJI_LEAF_MULTIPOLES(float, float, float, SphericalHexadecapole<float>);

/*! @brief perform multipole upward sweep for one tree level
 *
 * launch config: one thread per cell of the current level
 *
 * @param[in]  firstCell        first cell to process
 * @param[in]  lastCell         last cell to process
 * @param[in]  childOffsets     cell index of first child of each node
 * @param[in]  centers          source expansion (mass) centers
 * @param[out] multipoles       output multipole of each cell
 */
template<class T, class MType>
void upsweepMultipoles(TreeNodeIndex firstCell, TreeNodeIndex lastCell, const TreeNodeIndex* childOffsets,
                       const Vec4<T>* centers, MType* multipoles);

#define RYOANJI_UPSWEEP_MULTIPOLES(T, MType)                                                                           \
    void upsweepMultipoles(TreeNodeIndex        firstCell,                                                             \
                           TreeNodeIndex        lastCell,                                                              \
                           const TreeNodeIndex* childOffsets,                                                          \
                           const Vec4<T>*       centers,                                                               \
                           MType*               multipoles)

extern template RYOANJI_UPSWEEP_MULTIPOLES(double, SphericalHexadecapole<double>);
extern template RYOANJI_UPSWEEP_MULTIPOLES(double, SphericalHexadecapole<float>);
extern template RYOANJI_UPSWEEP_MULTIPOLES(float, SphericalHexadecapole<float>);

template<class Tc, class Tm, class Th, class Tf>
void computeLeafCenters(const Tc* x, const Tc* y, const Tc* z, const Tm* m, const Th* h,
                        const TreeNodeIndex* leafToInternal, TreeNodeIndex numLeaves, const LocalIndex* layout,
                        Vec4<Tf>* centers, Vec4<Tc>* cellXmin, Vec4<Tc>* cellXmax);

#define RYOANJI_LEAF_CENTERS(Tc, Tm, Th, Tf)                                                                           \
    void computeLeafCenters(const Tc*            x,                                                                    \
                            const Tc*            y,                                                                    \
                            const Tc*            z,                                                                    \
                            const Tm*            m,                                                                    \
                            const Th*            h,                                                                    \
                            const TreeNodeIndex* leafToInternal,                                                       \
                            TreeNodeIndex        numLeaves,                                                            \
                            const LocalIndex*    layout,                                                               \
                            Vec4<Tf>*            centers,                                                              \
                            Vec4<Tc>*            cellXmin,                                                             \
                            Vec4<Tc>*            cellXmax)

extern template RYOANJI_LEAF_CENTERS(double, double, double, double);
extern template RYOANJI_LEAF_CENTERS(double, float, float, double);
extern template RYOANJI_LEAF_CENTERS(double, float, float, float);
extern template RYOANJI_LEAF_CENTERS(float, float, float, float);

/*! @brief perform source expansion center upward sweep for one tree level
 *
 * launch config: one thread per cell of the current level
 *
 * @param[in]  firstCell        first cell to process
 * @param[in]  lastCell         last cell to process
 * @param[in]  childOffsets     cell index of first child of each node
 * @param[out] centers          source expansion (mass) centers
 * @param[out] cellXmin         minimum coordinate of any body in the cell
 * @param[out] cellXmax         maximum coordinate of any body in the cell
 */
template<class Tf, class Tc>
void computeUpsweepCenters(TreeNodeIndex firstCell, TreeNodeIndex lastCell, const TreeNodeIndex* childOffsets,
                           Vec4<Tf>* centers, Vec4<Tc>* cellXmin, Vec4<Tc>* cellXmax);

#define RYOANJI_UPSWEEP_CENTERS(TF, TC)                                                                                \
    void computeUpsweepCenters(TreeNodeIndex        firstCell,                                                         \
                               TreeNodeIndex        lastCell,                                                          \
                               const TreeNodeIndex* childOffsets,                                                      \
                               Vec4<TF>*            centers,                                                           \
                               Vec4<TC>*            cellXmin,                                                          \
                               Vec4<TC>*            cellXmax)

extern template RYOANJI_UPSWEEP_CENTERS(double, double);
extern template RYOANJI_UPSWEEP_CENTERS(float, float);
extern template RYOANJI_UPSWEEP_CENTERS(float, double);

#define RYOANJI_UPSWEEP(T, MType)                                                                                      \
    void upsweep(int                  numSources,                                                                      \
                 int                  numLeaves,                                                                       \
                 int                  numLevels,                                                                       \
                 T                    theta,                                                                           \
                 const int2*          levelRange,                                                                      \
                 const T*             x,                                                                               \
                 const T*             y,                                                                               \
                 const T*             z,                                                                               \
                 const T*             m,                                                                               \
                 const T*             h,                                                                               \
                 const LocalIndex*    layout,                                                                          \
                 const TreeNodeIndex* childOffsets,                                                                    \
                 const TreeNodeIndex* leafToInternal,                                                                  \
                 Vec4<T>*             centers,                                                                         \
                 MType*               Multipole)

template<class T, class MType>
RYOANJI_UPSWEEP(T, MType);

extern template RYOANJI_UPSWEEP(float, SphericalHexadecapole<float>);
extern template RYOANJI_UPSWEEP(double, SphericalHexadecapole<double>);

} // namespace ryoanji
