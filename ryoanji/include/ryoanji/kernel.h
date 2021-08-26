#pragma once

namespace {
  template<int nx, int ny, int nz>
  struct Index {
    static const int      I = Index<nx,ny+1,nz-1>::I + 1;
    static const uint64_t F = Index<nx,ny,nz-1>::F * nz;
    static __host__ __device__ __forceinline__
    float power(const fvec3 &dX) {
      return Index<nx,ny,nz-1>::power(dX) * dX[2];
    }
  };

  template<int nx, int ny>
  struct Index<nx,ny,0> {
    static const int      I = Index<nx+1,0,ny-1>::I + 1;
    static const uint64_t F = Index<nx,ny-1,0>::F * ny;
    static __host__ __device__ __forceinline__
    float power(const fvec3 &dX) {
      return Index<nx,ny-1,0>::power(dX) * dX[1];
    }
  };

  template<int nx>
  struct Index<nx,0,0> {
    static const int      I = Index<0,0,nx-1>::I + 1;
    static const uint64_t F = Index<nx-1,0,0>::F * nx;
    static __host__ __device__ __forceinline__
    float power(const fvec3 &dX) {
      return Index<nx-1,0,0>::power(dX) * dX[0];
    }
  };

  template<>
  struct Index<0,0,0> {
    static const int      I = 0;
    static const uint64_t F = 1;
    static __host__ __device__ __forceinline__
    float power(const fvec3&) { return 1.0f; }
  };


  template<int n>
  struct DerivativeTerm {
    static const int c = 1 - 2 * n;
    static __host__ __device__ __forceinline__
    void invR(float *invRN, const float &invR2) {
      DerivativeTerm<n-1>::invR(invRN,invR2);
      invRN[n] = c * invRN[n-1] * invR2;
    }
  };

  template<>
  struct DerivativeTerm<0> {
    static __host__ __device__ __forceinline__
    void invR(float*, const float&) {}
  };


  template<int depth, int nx, int ny, int nz, int flag>
  struct DerivativeSum {
    static const int cx = nx * (nx - 1) / 2;
    static const int cy = ny * (ny - 1) / 2;
    static const int cz = nz * (nz - 1) / 2;
    static const int n = nx + ny + nz;
    static const int d = depth > 0 ? depth : 1;
    static __host__ __device__ __forceinline__
    float loop(float *invRN, const fvec3 &dX) {
      return Index<nx,ny,nz>::power(dX) * invRN[n+depth] / d
	+ cx*DerivativeSum<depth+1,nx-2,ny,nz,(nx>3)*4+3>::loop(invRN,dX)
	+ cy*DerivativeSum<depth+1,nx,ny-2,nz,(ny>3)*2+5>::loop(invRN,dX)
	+ cz*DerivativeSum<depth+1,nx,ny,nz-2,(nz>3)  +6>::loop(invRN,dX);
    }
  };

  template<int depth, int nx, int ny, int nz>
  struct DerivativeSum<depth,nx,ny,nz,6> {
    static const int cx = nx * (nx - 1) / 2;
    static const int cy = ny * (ny - 1) / 2;
    static const int n = nx + ny + nz;
    static const int d = depth > 0 ? depth : 1;
    static __host__ __device__ __forceinline__
    float loop(float *invRN, const fvec3 &dX) {
      return Index<nx,ny,nz>::power(dX) * invRN[n+depth] / d
	+ cx*DerivativeSum<depth+1,nx-2,ny,nz,(nx>3)*4+2>::loop(invRN,dX)
	+ cy*DerivativeSum<depth+1,nx,ny-2,nz,(ny>3)*2+4>::loop(invRN,dX);
    }
  };

  template<int depth, int nx, int ny, int nz>
  struct DerivativeSum<depth,nx,ny,nz,5> {
    static const int cx = nx * (nx - 1) / 2;
    static const int cz = nz * (nz - 1) / 2;
    static const int n = nx + ny + nz;
    static const int d = depth > 0 ? depth : 1;
    static __host__ __device__ __forceinline__
    float loop(float *invRN, const fvec3 &dX) {
      return Index<nx,ny,nz>::power(dX) * invRN[n+depth] / d
	+ cx*DerivativeSum<depth+1,nx-2,ny,nz,(nx>3)*4+1>::loop(invRN,dX)
	+ cz*DerivativeSum<depth+1,nx,ny,nz-2,(nz>3)  +4>::loop(invRN,dX);
    }
  };

  template<int depth, int nx, int ny, int nz>
  struct DerivativeSum<depth,nx,ny,nz,4> {
    static const int cx = nx * (nx - 1) / 2;
    static const int n = nx + ny + nz;
    static const int d = depth > 0 ? depth : 1;
    static __host__ __device__ __forceinline__
    float loop(float *invRN, const fvec3 &dX) {
      return Index<nx,ny,nz>::power(dX) * invRN[n+depth] / d
	+ cx*DerivativeSum<depth+1,nx-2,ny,nz,(nx>3)*4>::loop(invRN,dX);
    }
  };

  template<int depth, int nx, int ny, int nz>
  struct DerivativeSum<depth,nx,ny,nz,3> {
    static const int cy = ny * (ny - 1) / 2;
    static const int cz = nz * (nz - 1) / 2;
    static const int n = nx + ny + nz;
    static const int d = depth > 0 ? depth : 1;
    static __host__ __device__ __forceinline__
    float loop(float *invRN, const fvec3 &dX) {
      return Index<nx,ny,nz>::power(dX) * invRN[n+depth] / d
	+ cy*DerivativeSum<depth+1,nx,ny-2,nz,(ny>3)*2+1>::loop(invRN,dX)
	+ cz*DerivativeSum<depth+1,nx,ny,nz-2,(nz>3)  +2>::loop(invRN,dX);
    }
  };

  template<int depth, int nx, int ny, int nz>
  struct DerivativeSum<depth,nx,ny,nz,2> {
    static const int cy = ny * (ny - 1) / 2;
    static const int n = nx + ny + nz;
    static const int d = depth > 0 ? depth : 1;
    static __host__ __device__ __forceinline__
    float loop(float *invRN, const fvec3 &dX) {
      return Index<nx,ny,nz>::power(dX) * invRN[n+depth] / d
	+ cy*DerivativeSum<depth+1,nx,ny-2,nz,(ny>3)*2>::loop(invRN,dX);
    }
  };

  template<int depth, int nx, int ny, int nz>
  struct DerivativeSum<depth,nx,ny,nz,1> {
    static const int cz = nz * (nz - 1) / 2;
    static const int n = nx + ny + nz;
    static const int d = depth > 0 ? depth : 1;
    static __host__ __device__ __forceinline__
    float loop(float *invRN, const fvec3 &dX) {
      return Index<nx,ny,nz>::power(dX) * invRN[n+depth] / d
	+ cz*DerivativeSum<depth+1,nx,ny,nz-2,(nz>3)>::loop(invRN,dX);
    }
  };

  template<int depth, int nx, int ny, int nz>
  struct DerivativeSum<depth,nx,ny,nz,0> {
    static const int n = nx + ny + nz;
    static const int d = depth > 0 ? depth : 1;
    static __host__ __device__ __forceinline__
    float loop(float *invRN, const fvec3 &dX) {
      return Index<nx,ny,nz>::power(dX) * invRN[n+depth] / d;
    }
  };


  template<int nx, int ny, int nz, int kx=nx, int ky=ny, int kz=nz>
  struct MultipoleSum {
    static __host__ __device__ __forceinline__
    float kernel(const fvec3 &dX, const fvecP &M) {
      return MultipoleSum<nx,ny,nz,kx,ky,kz-1>::kernel(dX,M)
	+ Index<nx-kx,ny-ky,nz-kz>::power(dX)
	/ Index<nx-kx,ny-ky,nz-kz>::F
	* M[Index<kx,ky,kz>::I];
    }
  };

  template<int nx, int ny, int nz, int kx, int ky>
  struct MultipoleSum<nx,ny,nz,kx,ky,0> {
    static __host__ __device__ __forceinline__
    float kernel(const fvec3 &dX, const fvecP &M) {
      return MultipoleSum<nx,ny,nz,kx,ky-1,nz>::kernel(dX,M)
	+ Index<nx-kx,ny-ky,nz>::power(dX)
	/ Index<nx-kx,ny-ky,nz>::F
	* M[Index<kx,ky,0>::I];
    }
  };

  template<int nx, int ny, int nz, int kx>
  struct MultipoleSum<nx,ny,nz,kx,0,0> {
    static __host__ __device__ __forceinline__
    float kernel(const fvec3 &dX, const fvecP &M) {
      return MultipoleSum<nx,ny,nz,kx-1,ny,nz>::kernel(dX,M)
	+ Index<nx-kx,ny,nz>::power(dX)
	/ Index<nx-kx,ny,nz>::F
	* M[Index<kx,0,0>::I];
    }
  };

  template<int nx, int ny, int nz>
  struct MultipoleSum<nx,ny,nz,0,0,0> {
    static __host__ __device__ __forceinline__
    float kernel(const fvec3 &dX, const fvecP &M) {
      return Index<nx,ny,nz>::power(dX)
	/ Index<nx,ny,nz>::F
	* M[Index<0,0,0>::I];
    }
  };


  template<int nx, int ny, int nz>
  struct Kernels {
    static const int n = nx + ny + nz;
    static const int x = nx > 0;
    static const int y = ny > 0;
    static const int z = nz > 0;
    static const int flag = (nx>1)*4+(ny>1)*2+(nz>1);
    static __host__ __device__ __forceinline__
    void P2M(fvecP &M, const fvec3 &dX) {
      Kernels<nx,ny+1,nz-1>::P2M(M,dX);
      M[Index<nx,ny,nz>::I] = Index<nx,ny,nz>::power(dX) / Index<nx,ny,nz>::F * M[0];
    }
    static __host__ __device__ __forceinline__
    void M2M(fvecP &MI, const fvec3 &dX, const fvecP &MJ) {
      Kernels<nx,ny+1,nz-1>::M2M(MI,dX,MJ);
      MI[Index<nx,ny,nz>::I] += MultipoleSum<nx,ny,nz>::kernel(dX,MJ);
    }
    static __host__ __device__ __forceinline__
    void M2P(fvec4 &TRG, float *invRN, const fvec3 &dX, const fvecP &M) {
      Kernels<nx,ny+1,nz-1>::M2P(TRG,invRN,dX,M);
      const float C = DerivativeSum<0,nx,ny,nz,flag>::loop(invRN,dX);
      TRG[0] -= M[Index<nx,ny,nz>::I] * C;
      TRG[1] += M[Index<(nx-1)*x,ny,nz>::I] * C * x;
      TRG[2] += M[Index<nx,(ny-1)*y,nz>::I] * C * y;
      TRG[3] += M[Index<nx,ny,(nz-1)*z>::I] * C * z;
    }
  };

  template<int nx, int ny>
  struct Kernels<nx,ny,0> {
    static const int n = nx + ny;
    static const int x = nx > 0;
    static const int y = ny > 0;
    static const int flag = (nx>1)*4+(ny>1)*2;
    static __host__ __device__ __forceinline__
    void P2M(fvecP &M, const fvec3 &dX) {
      Kernels<nx+1,0,ny-1>::P2M(M,dX);
      M[Index<nx,ny,0>::I] = Index<nx,ny,0>::power(dX) / Index<nx,ny,0>::F * M[0];
    }
    static __host__ __device__ __forceinline__
    void M2M(fvecP &MI, const fvec3 &dX, const fvecP &MJ) {
      Kernels<nx+1,0,ny-1>::M2M(MI,dX,MJ);
      MI[Index<nx,ny,0>::I] += MultipoleSum<nx,ny,0>::kernel(dX,MJ);
    }
    static __host__ __device__ __forceinline__
    void M2P(fvec4 &TRG, float *invRN, const fvec3 &dX, const fvecP &M) {
      Kernels<nx+1,0,ny-1>::M2P(TRG,invRN,dX,M);
      const float C = DerivativeSum<0,nx,ny,0,flag>::loop(invRN,dX);
      TRG[0] -= M[Index<nx,ny,0>::I] * C;
      TRG[1] += M[Index<(nx-1)*x,ny,0>::I] * C * x;
      TRG[2] += M[Index<nx,(ny-1)*y,0>::I] * C * y;
    }
  };

  template<int nx>
  struct Kernels<nx,0,0> {
    static const int n = nx;
    static const int flag = (nx>1)*4;
    static __host__ __device__ __forceinline__
    void P2M(fvecP &M, const fvec3 &dX) {
      Kernels<0,0,nx-1>::P2M(M,dX);
      M[Index<nx,0,0>::I] = Index<nx,0,0>::power(dX) / Index<nx,0,0>::F * M[0];
    }
    static __host__ __device__ __forceinline__
    void M2M(fvecP &MI, const fvec3 &dX, const fvecP &MJ) {
      Kernels<0,0,nx-1>::M2M(MI,dX,MJ);
      MI[Index<nx,0,0>::I] += MultipoleSum<nx,0,0>::kernel(dX,MJ);
    }
    static __host__ __device__ __forceinline__
    void M2P(fvec4 &TRG, float *invRN, const fvec3 &dX, const fvecP &M) {
      Kernels<0,0,nx-1>::M2P(TRG,invRN,dX,M);
      const float C = DerivativeSum<0,nx,0,0,flag>::loop(invRN,dX);
      TRG[0] -= M[Index<nx,0,0>::I] * C;
      TRG[1] += M[Index<nx-1,0,0>::I] * C;
    }
  };

#if MASS
  template<>
  struct Kernels<0,0,2> {
    static __host__ __device__ __forceinline__
    void P2M(fvecP &M, const fvec3 &dX) {
      Kernels<0,1,1>::P2M(M,dX);
      M[Index<0,0,2>::I] = Index<0,0,2>::power(dX) / Index<0,0,2>::F * M[0];
    }
    static __host__ __device__ __forceinline__
    void M2M(fvecP &MI, const fvec3 &dX, const fvecP &MJ) {
      Kernels<0,1,1>::M2M(MI,dX,MJ);
      MI[Index<0,0,2>::I] += MultipoleSum<0,0,2>::kernel(dX,MJ);
    }
    static __host__ __device__ __forceinline__
    void M2P(fvec4 &TRG, float *invRN, const fvec3 &dX, const fvecP &M) {
      TRG[0] -= invRN[0] + invRN[1] * (M[4] + M[7] + M[9])
	+ invRN[2] * (M[4] * dX[0] * dX[0] + M[5] * dX[0] * dX[1] + M[6] * dX[0] * dX[2]
		    + M[7] * dX[1] * dX[1] + M[8] * dX[1] * dX[2] + M[9] * dX[2] * dX[2]);
      TRG[1] += dX[0] * invRN[1];
      TRG[2] += dX[1] * invRN[1];
      TRG[3] += dX[2] * invRN[1];
    }
  };
#endif

  template<>
  struct Kernels<0,0,0> {
    static __host__ __device__ __forceinline__
    void P2M(fvecP &M, const fvec3 &dX) {}
    static __host__ __device__ __forceinline__
    void M2M(fvecP &MI, const fvec3 &dX, const fvecP &MJ) {
      MI[Index<0,0,0>::I] += MultipoleSum<0,0,0>::kernel(dX,MJ);
    }
    static __host__ __device__ __forceinline__
    void M2P(fvec4 &TRG, float *invRN, const fvec3&, const fvecP&) {
      TRG[0] -= invRN[0];
    }
  };


  __device__ __forceinline__
  void P2M(const int begin,
	   const int end,
	   const fvec4 center,
	   fvecP & Mi) {
    for (int i=begin; i<end; i++) {
      fvec4 body = tex1Dfetch(texBody,i);
      fvec3 dX = make_fvec3(center - body);
      fvecP M;
      M[0] = body[3];
      Kernels<0,0,P-1>::P2M(M,dX);
      Mi += M;
    }
  }

  __device__ __forceinline__
  void M2M(const int begin,
	   const int end,
	   const fvec4 Xi,
	   fvec4 * sourceCenter,
	   fvec4 * Multipole,
	   fvecP & Mi) {
    for (int i=begin; i<end; i++) {
      fvecP Mj = *(fvecP*)&Multipole[NVEC4*i];
      fvec4 Xj = sourceCenter[i];
      fvec3 dX = make_fvec3(Xi - Xj);
      Kernels<0,0,P-1>::M2M(Mi,dX,Mj);
    }
  }

  __device__ __forceinline__
  fvec4 P2P(fvec4 acc,
	    const fvec3 pos_i,
	    const fvec3 pos_j,
	    const float q_j,
	    const float EPS2) {
    fvec3 dX = pos_j - pos_i;
    const float R2 = norm(dX) + EPS2;
    const float invR = rsqrtf(R2);
    const float invR2 = invR * invR;
    const float invR1 = q_j * invR;
    dX *= invR1 * invR2;
    acc[0] -= invR1;
    acc[1] += dX[0];
    acc[2] += dX[1];
    acc[3] += dX[2];
    return acc;
  }

  __device__ __forceinline__
  fvec4 M2P(fvec4 acc,
	    const fvec3 & pos_i,
	    const fvec3 & pos_j,
	    fvecP & M,
	    float EPS2) {
    const fvec3 dX = pos_i - pos_j;
    const float R2 = norm(dX) + EPS2;
    const float invR = rsqrtf(R2);
    const float invR2 = invR * invR;
    float invRN[P];
    invRN[0] = M[0] * invR;
    DerivativeTerm<P-1>::invR(invRN,invR2);
    const float M0 = M[0];
    M[0] = 1;
    Kernels<0,0,P-1>::M2P(acc,invRN,dX,M);
    M[0] = M0;
    return acc;
  }
}
