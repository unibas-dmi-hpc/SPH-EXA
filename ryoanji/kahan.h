#ifndef kahan_h
#define kahan_h
#include <iostream>
#ifndef __CUDACC__
#define __host__
#define __device__
#define __forceinline__
#endif
//! Operator overloading for Kahan summation
template<typename T>
struct kahan {
  T s;
  T c;
  __host__ __device__ __forceinline__
  kahan(){}                                                     // Default constructor
  __host__ __device__ __forceinline__
  kahan(const T &v) {                                           // Copy constructor (scalar)
    s = v;
    c = 0;
  }
  __host__ __device__ __forceinline__
  kahan(const kahan &v) {                                       // Copy constructor (structure)
    s = v.s;
    c = v.c;
  }
  __host__ __device__ __forceinline__
  ~kahan(){}                                                    // Destructor
  __host__ __device__ __forceinline__
  const kahan &operator=(const T v) {                           // Scalar assignment
    s = v;
    c = 0;
    return *this;
  }
  __host__ __device__ __forceinline__
  const kahan &operator+=(const T v) {                          // Scalar compound assignment (add)
    T y = v - c;
    T t = s + y;
    c = (t - s) - y;
    s = t;
    return *this;
  }
  __host__ __device__ __forceinline__
  const kahan &operator-=(const T v) {                          // Scalar compound assignment (subtract)
    T y = - v - c;
    T t = s + y;
    c = (t - s) - y;
    s = t; 
    return *this;
  }
  __host__ __device__ __forceinline__
  const kahan &operator*=(const T v) {                          // Scalar compound assignment (multiply)
    c *= v;
    s *= v; 
    return *this;
  }
  __host__ __device__ __forceinline__
  const kahan &operator/=(const T v) {                          // Scalar compound assignment (divide)
    c /= v;
    s /= v; 
    return *this;
  }
  __host__ __device__ __forceinline__
  const kahan &operator=(const kahan &v) {                      // Vector assignment
    s = v.s;
    c = v.c;
    return *this;
  }
  __host__ __device__ __forceinline__
  const kahan &operator+=(const kahan &v) {                     // Vector compound assignment (add)
    T y = v.s - c;
    T t = s + y;
    c = (t - s) - y;
    s = t;
    y = v.c - c;
    t = s + y;
    c = (t - s) - y;
    s = t;
    return *this;
  }
  __host__ __device__ __forceinline__
  const kahan &operator-=(const kahan &v) {                     // Vector compound assignment (subtract)
    T y = - v.s - c;
    T t = s + y;
    c = (t - s) - y;
    s = t;
    y = - v.c - c;
    t = s + y;
    c = (t - s) - y;
    s = t;
    return *this;
  }
  __host__ __device__ __forceinline__
  const kahan &operator*=(const kahan &v) {                     // Vector compound assignment (multiply)
    c *= (v.c + v.s);
    s *= (v.c + v.s); 
    return *this;
  }
  __host__ __device__ __forceinline__
  const kahan &operator/=(const kahan &v) {                     // Vector compound assignment (divide)
    c /= (v.c + v.s);
    s /= (v.c + v.s); 
    return *this;
  }
  __host__ __device__ __forceinline__
  kahan operator-() const {                                     // Vector arithmetic (negation)
    kahan temp;
    temp.s = -s;
    temp.c = -c;
    return temp;
  }
  __host__ __device__ __forceinline__
  operator       T ()       {return s+c;}                       // Type-casting (lvalue)
  __host__ __device__ __forceinline__
  operator const T () const {return s+c;}                       // Type-casting (rvalue) 
  friend std::ostream &operator<<(std::ostream &s, const kahan &v) {// Output stream
    s << (v.s + v.c);
    return s;
  }
  friend std::istream &operator>>(std::istream &s, kahan &v) {  // Input stream
    s >> v.s;
    v.c = 0;
    return s;
  }
};
#endif
