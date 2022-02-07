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

#pragma once

/*! @file
 * @brief  EXA-FMM multipole kernels
 *
 * @author Rio Yokota <rioyokota@gsic.titech.ac.jp>
 */

namespace
{

HOST_DEVICE_FUN DEVICE_INLINE float inverseSquareRoot(float x)
{
#if defined(__HIP_DEVICE_COMPILE__) || defined (__CUDA_ARCH__)
    return rsqrtf(x);
#else
    return 1.0f / std::sqrt(x);
#endif
}

HOST_DEVICE_FUN DEVICE_INLINE double inverseSquareRoot(double x)
{
#if defined(__HIP_DEVICE_COMPILE__) || defined (__CUDA_ARCH__)
    return rsqrt(x);
#else
    return 1.0 / std::sqrt(x);
#endif
}

template<int nx, int ny, int nz>
struct Index
{
    static constexpr int I      = Index<nx, ny + 1, nz - 1>::I + 1;
    static constexpr uint64_t F = Index<nx, ny, nz - 1>::F * nz;
    template<class T>
    static HOST_DEVICE_FUN DEVICE_INLINE T power(const Vec3<T>& dX)
    {
        return Index<nx, ny, nz - 1>::power(dX) * dX[2];
    }
};

template<int nx, int ny>
struct Index<nx, ny, 0>
{
    static constexpr int I      = Index<nx + 1, 0, ny - 1>::I + 1;
    static constexpr uint64_t F = Index<nx, ny - 1, 0>::F * ny;

    template<class T>
    static HOST_DEVICE_FUN DEVICE_INLINE T power(const Vec3<T>& dX)
    {
        return Index<nx, ny - 1, 0>::power(dX) * dX[1];
    }
};

template<int nx>
struct Index<nx, 0, 0>
{
    static constexpr int I      = Index<0, 0, nx - 1>::I + 1;
    static constexpr uint64_t F = Index<nx - 1, 0, 0>::F * nx;

    template<class T>
    static HOST_DEVICE_FUN DEVICE_INLINE T power(const Vec3<T>& dX)
    {
        return Index<nx - 1, 0, 0>::power(dX) * dX[0];
    }
};

template<>
struct Index<0, 0, 0>
{
    static constexpr int I      = 0;
    static constexpr uint64_t F = 1;

    template<class T>
    static HOST_DEVICE_FUN DEVICE_INLINE T power(const Vec3<T>&) { return T(1.0); }
};

template<int n>
struct DerivativeTerm
{
    static constexpr int c = 1 - 2 * n;

    template<class T>
    static HOST_DEVICE_FUN DEVICE_INLINE void invR(T* invRN, const T& invR2)
    {
        DerivativeTerm<n - 1>::invR(invRN, invR2);
        invRN[n] = c * invRN[n - 1] * invR2;
    }
};

template<>
struct DerivativeTerm<0>
{
    template<class T>
    static HOST_DEVICE_FUN DEVICE_INLINE void invR(T*, const T&) {}
};

template<int depth, int nx, int ny, int nz, int flag>
struct DerivativeSum
{
    static constexpr int cx = nx * (nx - 1) / 2;
    static constexpr int cy = ny * (ny - 1) / 2;
    static constexpr int cz = nz * (nz - 1) / 2;
    static constexpr int n  = nx + ny + nz;
    static constexpr int d  = depth > 0 ? depth : 1;

    template<class T>
    static HOST_DEVICE_FUN DEVICE_INLINE T loop(const T* invRN, const Vec3<T>& dX)
    {
        return Index<nx, ny, nz>::power(dX) * invRN[n + depth] / d +
               cx * DerivativeSum<depth + 1, nx - 2, ny, nz, (nx > 3) * 4 + 3>::loop(invRN, dX) +
               cy * DerivativeSum<depth + 1, nx, ny - 2, nz, (ny > 3) * 2 + 5>::loop(invRN, dX) +
               cz * DerivativeSum<depth + 1, nx, ny, nz - 2, (nz > 3) + 6>::loop(invRN, dX);
    }
};

template<int depth, int nx, int ny, int nz>
struct DerivativeSum<depth, nx, ny, nz, 6>
{
    static constexpr int cx = nx * (nx - 1) / 2;
    static constexpr int cy = ny * (ny - 1) / 2;
    static constexpr int n  = nx + ny + nz;
    static constexpr int d  = depth > 0 ? depth : 1;

    template<class T>
    static HOST_DEVICE_FUN DEVICE_INLINE T loop(const T* invRN, const Vec3<T>& dX)
    {
        return Index<nx, ny, nz>::power(dX) * invRN[n + depth] / d +
               cx * DerivativeSum<depth + 1, nx - 2, ny, nz, (nx > 3) * 4 + 2>::loop(invRN, dX) +
               cy * DerivativeSum<depth + 1, nx, ny - 2, nz, (ny > 3) * 2 + 4>::loop(invRN, dX);
    }
};

template<int depth, int nx, int ny, int nz>
struct DerivativeSum<depth, nx, ny, nz, 5>
{
    static constexpr int cx = nx * (nx - 1) / 2;
    static constexpr int cz = nz * (nz - 1) / 2;
    static constexpr int n  = nx + ny + nz;
    static constexpr int d  = depth > 0 ? depth : 1;

    template<class T>
    static HOST_DEVICE_FUN DEVICE_INLINE T loop(const T* invRN, const Vec3<T>& dX)
    {
        return Index<nx, ny, nz>::power(dX) * invRN[n + depth] / d +
               cx * DerivativeSum<depth + 1, nx - 2, ny, nz, (nx > 3) * 4 + 1>::loop(invRN, dX) +
               cz * DerivativeSum<depth + 1, nx, ny, nz - 2, (nz > 3) + 4>::loop(invRN, dX);
    }
};

template<int depth, int nx, int ny, int nz>
struct DerivativeSum<depth, nx, ny, nz, 4>
{
    static constexpr int cx = nx * (nx - 1) / 2;
    static constexpr int n  = nx + ny + nz;
    static constexpr int d  = depth > 0 ? depth : 1;

    template<class T>
    static HOST_DEVICE_FUN DEVICE_INLINE T loop(const T* invRN, const Vec3<T>& dX)
    {
        return Index<nx, ny, nz>::power(dX) * invRN[n + depth] / d +
               cx * DerivativeSum<depth + 1, nx - 2, ny, nz, (nx > 3) * 4>::loop(invRN, dX);
    }
};

template<int depth, int nx, int ny, int nz>
struct DerivativeSum<depth, nx, ny, nz, 3>
{
    static constexpr int cy = ny * (ny - 1) / 2;
    static constexpr int cz = nz * (nz - 1) / 2;
    static constexpr int n  = nx + ny + nz;
    static constexpr int d  = depth > 0 ? depth : 1;

    template<class T>
    static HOST_DEVICE_FUN DEVICE_INLINE T loop(const T* invRN, const Vec3<T>& dX)
    {
        return Index<nx, ny, nz>::power(dX) * invRN[n + depth] / d +
               cy * DerivativeSum<depth + 1, nx, ny - 2, nz, (ny > 3) * 2 + 1>::loop(invRN, dX) +
               cz * DerivativeSum<depth + 1, nx, ny, nz - 2, (nz > 3) + 2>::loop(invRN, dX);
    }
};

template<int depth, int nx, int ny, int nz>
struct DerivativeSum<depth, nx, ny, nz, 2>
{
    static constexpr int cy = ny * (ny - 1) / 2;
    static constexpr int n  = nx + ny + nz;
    static constexpr int d  = depth > 0 ? depth : 1;

    template<class T>
    static HOST_DEVICE_FUN DEVICE_INLINE T loop(const T* invRN, const Vec3<T>& dX)
    {
        return Index<nx, ny, nz>::power(dX) * invRN[n + depth] / d +
               cy * DerivativeSum<depth + 1, nx, ny - 2, nz, (ny > 3) * 2>::loop(invRN, dX);
    }
};

template<int depth, int nx, int ny, int nz>
struct DerivativeSum<depth, nx, ny, nz, 1>
{
    static constexpr int cz = nz * (nz - 1) / 2;
    static constexpr int n  = nx + ny + nz;
    static constexpr int d  = depth > 0 ? depth : 1;

    template<class T>
    static HOST_DEVICE_FUN DEVICE_INLINE T loop(const T* invRN, const Vec3<T>& dX)
    {
        return Index<nx, ny, nz>::power(dX) * invRN[n + depth] / d +
               cz * DerivativeSum<depth + 1, nx, ny, nz - 2, (nz > 3)>::loop(invRN, dX);
    }
};

template<int depth, int nx, int ny, int nz>
struct DerivativeSum<depth, nx, ny, nz, 0>
{
    static constexpr int n = nx + ny + nz;
    static constexpr int d = depth > 0 ? depth : 1;

    template<class T>
    static HOST_DEVICE_FUN DEVICE_INLINE T loop(const T* invRN, const Vec3<T>& dX)
    {
        return Index<nx, ny, nz>::power(dX) * invRN[n + depth] / d;
    }
};

template<int nx, int ny, int nz, int kx = nx, int ky = ny, int kz = nz>
struct MultipoleSum
{
    template<class T, class MType>
    static HOST_DEVICE_FUN DEVICE_INLINE T kernel(const Vec3<T>& dX, const MType& M)
    {
        return MultipoleSum<nx, ny, nz, kx, ky, kz - 1>::kernel(dX, M) + Index<nx - kx, ny - ky, nz - kz>::power(dX) /
                                                                             Index<nx - kx, ny - ky, nz - kz>::F *
                                                                             M[Index<kx, ky, kz>::I];
    }
};

template<int nx, int ny, int nz, int kx, int ky>
struct MultipoleSum<nx, ny, nz, kx, ky, 0>
{
    template<class T, class MType>
    static HOST_DEVICE_FUN DEVICE_INLINE T kernel(const Vec3<T>& dX, const MType& M)
    {
        return MultipoleSum<nx, ny, nz, kx, ky - 1, nz>::kernel(dX, M) +
               Index<nx - kx, ny - ky, nz>::power(dX) / Index<nx - kx, ny - ky, nz>::F * M[Index<kx, ky, 0>::I];
    }
};

template<int nx, int ny, int nz, int kx>
struct MultipoleSum<nx, ny, nz, kx, 0, 0>
{
    template<class T, class MType>
    static HOST_DEVICE_FUN DEVICE_INLINE T kernel(const Vec3<T>& dX, const MType& M)
    {
        return MultipoleSum<nx, ny, nz, kx - 1, ny, nz>::kernel(dX, M) +
               Index<nx - kx, ny, nz>::power(dX) / Index<nx - kx, ny, nz>::F * M[Index<kx, 0, 0>::I];
    }
};

template<int nx, int ny, int nz>
struct MultipoleSum<nx, ny, nz, 0, 0, 0>
{
    template<class T, class MType>
    static HOST_DEVICE_FUN DEVICE_INLINE T kernel(const Vec3<T>& dX, const MType& M)
    {
        return Index<nx, ny, nz>::power(dX) / Index<nx, ny, nz>::F * M[Index<0, 0, 0>::I];
    }
};

template<int nx, int ny, int nz>
struct Kernels
{
    static constexpr int n    = nx + ny + nz;
    static constexpr int x    = nx > 0;
    static constexpr int y    = ny > 0;
    static constexpr int z    = nz > 0;
    static constexpr int flag = (nx > 1) * 4 + (ny > 1) * 2 + (nz > 1);

    template<class T, class MType>
    static HOST_DEVICE_FUN DEVICE_INLINE void P2M(MType& M, const Vec3<T>& dX)
    {
        Kernels<nx, ny + 1, nz - 1>::P2M(M, dX);
        M[Index<nx, ny, nz>::I] = Index<nx, ny, nz>::power(dX) / Index<nx, ny, nz>::F * M[0];
    }

    template<class T, class MType>
    static HOST_DEVICE_FUN DEVICE_INLINE void M2M(MType& MI, const Vec3<T>& dX, const MType& MJ)
    {
        Kernels<nx, ny + 1, nz - 1>::M2M(MI, dX, MJ);
        MI[Index<nx, ny, nz>::I] += MultipoleSum<nx, ny, nz>::kernel(dX, MJ);
    }

    template<class T, class MType>
    static HOST_DEVICE_FUN DEVICE_INLINE void M2P(Vec4<T>& TRG, T* invRN, const Vec3<T>& dX, const MType& M)
    {
        Kernels<nx, ny + 1, nz - 1>::M2P(TRG, invRN, dX, M);
        T C = DerivativeSum<0, nx, ny, nz, flag>::loop(invRN, dX);
        TRG[0] -= M[Index<nx, ny, nz>::I] * C;
        TRG[1] += M[Index<(nx - 1) * x, ny, nz>::I] * C * x;
        TRG[2] += M[Index<nx, (ny - 1) * y, nz>::I] * C * y;
        TRG[3] += M[Index<nx, ny, (nz - 1) * z>::I] * C * z;
    }
};

template<int nx, int ny>
struct Kernels<nx, ny, 0>
{
    static constexpr int n    = nx + ny;
    static constexpr int x    = nx > 0;
    static constexpr int y    = ny > 0;
    static constexpr int flag = (nx > 1) * 4 + (ny > 1) * 2;

    template<class T, class MType>
    static HOST_DEVICE_FUN DEVICE_INLINE void P2M(MType& M, const Vec3<T>& dX)
    {
        Kernels<nx + 1, 0, ny - 1>::P2M(M, dX);
        M[Index<nx, ny, 0>::I] = Index<nx, ny, 0>::power(dX) / Index<nx, ny, 0>::F * M[0];
    }

    template<class T, class MType>
    static HOST_DEVICE_FUN DEVICE_INLINE void M2M(MType& MI, const Vec3<T>& dX, const MType& MJ)
    {
        Kernels<nx + 1, 0, ny - 1>::M2M(MI, dX, MJ);
        MI[Index<nx, ny, 0>::I] += MultipoleSum<nx, ny, 0>::kernel(dX, MJ);
    }

    template<class T, class MType>
    static HOST_DEVICE_FUN DEVICE_INLINE void M2P(Vec4<T>& TRG, T* invRN, const Vec3<T>& dX, const MType& M)
    {
        Kernels<nx + 1, 0, ny - 1>::M2P(TRG, invRN, dX, M);
        T C = DerivativeSum<0, nx, ny, 0, flag>::loop(invRN, dX);
        TRG[0] -= M[Index<nx, ny, 0>::I] * C;
        TRG[1] += M[Index<(nx - 1) * x, ny, 0>::I] * C * x;
        TRG[2] += M[Index<nx, (ny - 1) * y, 0>::I] * C * y;
    }
};

template<int nx>
struct Kernels<nx, 0, 0>
{
    static constexpr int n    = nx;
    static constexpr int flag = (nx > 1) * 4;

    template<class T, class MType>
    static HOST_DEVICE_FUN DEVICE_INLINE void P2M(MType& M, const Vec3<T>& dX)
    {
        Kernels<0, 0, nx - 1>::P2M(M, dX);
        M[Index<nx, 0, 0>::I] = Index<nx, 0, 0>::power(dX) / Index<nx, 0, 0>::F * M[0];
    }

    template<class T, class MType>
    static HOST_DEVICE_FUN DEVICE_INLINE void M2M(MType& MI, const Vec3<T>& dX, const MType& MJ)
    {
        Kernels<0, 0, nx - 1>::M2M(MI, dX, MJ);
        MI[Index<nx, 0, 0>::I] += MultipoleSum<nx, 0, 0>::kernel(dX, MJ);
    }

    template<class T, class MType>
    static HOST_DEVICE_FUN DEVICE_INLINE void M2P(Vec4<T>& TRG, T* invRN, const Vec3<T> dX, const MType& M)
    {
        Kernels<0, 0, nx - 1>::M2P(TRG, invRN, dX, M);
        const float C = DerivativeSum<0, nx, 0, 0, flag>::loop(invRN, dX);
        TRG[0] -= M[Index<nx, 0, 0>::I] * C;
        TRG[1] += M[Index<nx - 1, 0, 0>::I] * C;
    }
};

template<>
struct Kernels<0, 0, 2>
{
    template<class T, class MType>
    static HOST_DEVICE_FUN DEVICE_INLINE void P2M(MType& M, const Vec3<T>& dX)
    {
        Kernels<0, 1, 1>::P2M(M, dX);
        M[Index<0, 0, 2>::I] = Index<0, 0, 2>::power(dX) / Index<0, 0, 2>::F * M[0];
    }

    template<class T, class MType>
    static HOST_DEVICE_FUN DEVICE_INLINE void M2M(MType& MI, const Vec3<T>& dX, const MType& MJ)
    {
        Kernels<0, 1, 1>::M2M(MI, dX, MJ);
        MI[Index<0, 0, 2>::I] += MultipoleSum<0, 0, 2>::kernel(dX, MJ);
    }

    template<class T, class MType>
    static HOST_DEVICE_FUN DEVICE_INLINE void M2P(Vec4<T>& TRG, T* invRN, const Vec3<T>& dX, const MType& M)
    {
        TRG[0] -= invRN[0] + invRN[1] * (M[4] + M[7] + M[9]) +
                  invRN[2] * (M[4] * dX[0] * dX[0] + M[5] * dX[0] * dX[1] + M[6] * dX[0] * dX[2] +
                              M[7] * dX[1] * dX[1] + M[8] * dX[1] * dX[2] + M[9] * dX[2] * dX[2]);
        TRG[1] += dX[0] * invRN[1];
        TRG[2] += dX[1] * invRN[1];
        TRG[3] += dX[2] * invRN[1];
    }
};

template<>
struct Kernels<0, 0, 0>
{
    template<class T, class MType>
    static HOST_DEVICE_FUN DEVICE_INLINE void P2M(MType& M, const Vec3<T>& dX)
    {
    }

    template<class T, class MType>
    static HOST_DEVICE_FUN DEVICE_INLINE void M2M(MType& MI, const Vec3<T>& dX, const MType& MJ)
    {
        MI[Index<0, 0, 0>::I] += MultipoleSum<0, 0, 0>::kernel(dX, MJ);
    }

    template<class T, class MType>
    static HOST_DEVICE_FUN DEVICE_INLINE void M2P(Vec4<T>& TRG, T* invRN, const Vec3<T>&, const MType&)
    {
        TRG[0] -= invRN[0];
    }
};

} // namespace

namespace ryoanji
{

/*! @brief calculate multipole from particles
 *
 * @param[in]  begin    first particle index of bodyPos
 * @param[in]  end      last particles index of bodyPos
 * @param[in]  center   multipole center of gravity
 * @param[in]  bodyPos  body position and mass
 * @param[out] Mi       output multipole to add contributions to
 *
 */
template<class T, class MType>
HOST_DEVICE_FUN DEVICE_INLINE void P2M(int begin, int end, const Vec4<T>& center, const Vec4<T>* bodyPos, MType& Mi)
{
    constexpr int P = ExpansionOrder<MType{}.size()>{};

    for (int i = begin; i < end; i++)
    {
        fvec4 body = bodyPos[i];
        fvec3 dX   = make_fvec3(center - body);
        MType M;
        M[0] = body[3];
        Kernels<0, 0, P - 1>::P2M(M, dX);
        Mi += M;
    }
}

template<class T, class MType>
HOST_DEVICE_FUN DEVICE_INLINE void M2M(int begin, int end, const Vec4<T>& Xi, Vec4<T>* sourceCenter,
                                             fvec4* Multipole, MType& Mi)
{
    constexpr int P = ExpansionOrder<MType{}.size()>{};

    for (int i = begin; i < end; i++)
    {
        MType Mj = *(MType*)&Multipole[NVEC4 * i];
        fvec4 Xj = sourceCenter[i];
        fvec3 dX = make_fvec3(Xi - Xj);
        Kernels<0, 0, P - 1>::M2M(Mi, dX, Mj);
    }
}

/*! @brief interaction between two particles
 *
 * @param acc     acceleration to add to
 * @param pos_i
 * @param pos_j
 * @param q_j
 * @param EPS2
 * @return        input acceleration plus contribution from this call
 */
template<class T>
HOST_DEVICE_FUN DEVICE_INLINE Vec4<T> P2P(Vec4<T> acc, const Vec3<T>& pos_i, const Vec3<T>& pos_j, T q_j, T EPS2)
{
    Vec3<T> dX = pos_j - pos_i;
    T R2 = norm2(dX) + EPS2;
    T invR = inverseSquareRoot(R2);
    T invR2 = invR * invR;
    T invR1 = q_j * invR;

    dX *= invR1 * invR2;

    acc[0] -= invR1;
    acc[1] += dX[0];
    acc[2] += dX[1];
    acc[3] += dX[2];

    return acc;
}

/*! @brief apply multipole to particle
 *
 * @param acc     acceleration to add to
 * @param pos_i   target particle coordinate
 * @param pos_j   multipole source coordinate
 * @param M       the multipole
 * @param EPS2    plummer softening parameter
 * @return        input acceleration plus contribution from this call
 */
template<class T, class MType>
HOST_DEVICE_FUN DEVICE_INLINE Vec4<T> M2P(Vec4<T> acc, const Vec3<T>& pos_i, const Vec3<T>& pos_j, MType& M,
                                                T EPS2)
{
    constexpr int P = ExpansionOrder<MType{}.size()>{};

    Vec3<T> dX = pos_i - pos_j;
    T R2 = norm2(dX) + EPS2;
    T invR = inverseSquareRoot(R2);
    T invR2 = invR * invR;

    T invRN[P];
    invRN[0] = M[0] * invR;
    DerivativeTerm<P - 1>::invR(invRN, invR2);

    T M0 = M[0];
    M[0] = 1;
    Kernels<0, 0, P - 1>::M2P(acc, invRN, dX, M);
    M[0] = M0;

    return acc;
}

} // namespace ryoanji
