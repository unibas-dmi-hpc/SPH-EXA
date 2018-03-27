#ifndef BOUNDINGBOX_H
#define BOUNDINGBOX_H

#include <array>
#include <limits>
#include <cmath>
#include <iostream>

// An axis aligned bounding boxed (AABB)
struct BoundingBox {
  BoundingBox() = default;
  template <typename InputIterator>
  BoundingBox(InputIterator begin, InputIterator end);
  BoundingBox(std::initializer_list<double> l);
  BoundingBox(BoundingBox&& rhs);
  BoundingBox(const BoundingBox&) = default;

  BoundingBox& operator=(BoundingBox&) = default;
  BoundingBox& operator=(BoundingBox&& rhs);
  BoundingBox& operator=(const BoundingBox& rhs);

  ~BoundingBox() = default;

  bool contains(const BoundingBox& other) const;
  bool contains(const std::array<double, 3>& point) const;

  bool overlap(const BoundingBox& other, BoundingBox* out) const;

  std::array<BoundingBox, 8> partition() const;

  bool operator==(const BoundingBox& rhs) const;

  bool operator!=(const BoundingBox& rhs) const;

  friend
  std::ostream& operator<<(std::ostream& stream, const BoundingBox& rhs);

  double xhi, xlo, yhi, ylo, zhi, zlo;
};

const BoundingBox initial = BoundingBox{
    std::numeric_limits<double>::min(), std::numeric_limits<double>::max(),
    std::numeric_limits<double>::min(), std::numeric_limits<double>::max(),
    std::numeric_limits<double>::min(), std::numeric_limits<double>::max()
};

const BoundingBox invalid = BoundingBox{
    std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN(), 
    std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN(), 
    std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN()
};

template <typename InputIterator>
BoundingBox::BoundingBox(InputIterator begin, InputIterator end) : BoundingBox(initial) {
  for (; begin != end; ++begin) {
    const std::array<double, 3>& point = *begin;

    if (point[0] < xlo) {
      xlo = point[0];
    } else if (point[0] > xhi) {
      xhi = point[0];
    }

    if (point[1] < ylo) {
      ylo = point[1];
    } else if (point[1] > yhi) {
      yhi = point[1];
    }

    if (point[2] < zlo) {
      zlo = point[2];
    } else if (point[2] > zhi) {
      zhi = point[2];
    }
  }
}

#endif // defined BOUNDINGBOX_H



// #ifndef BOUNDINGBOX_H
// #define BOUNDINGBOX_H

// #include <array>
// #include <limits>
// #include <cmath>
// #include <iostream>
// #include "Particle.hpp"

// struct Point3D {
//     double x;
//     double y;
//     double z;

//     bool isNaN() const;
//     bool operator == (const Point3D& other) const;
// };

// struct BoundingBox {
//     Point3D mins;
//     Point3D maxs;

//     bool contains(const BoundingBox& other) const;
//     bool contains(const Point3D& point) const;

//     bool overlap(const BoundingBox& other, BoundingBox* out) const;
//     std::array<BoundingBox, 8> partition() const;

//     bool operator == (const BoundingBox& rhs) const;
//     bool operator != (const BoundingBox& rhs) const;
// };

// template <typename InputIterator>
// BoundingBox makeBoundingBox(InputIterator begin, InputIterator end);
// std::ostream& operator << (std::ostream& stream, const BoundingBox& rhs);

// #endif // defined BOUNDINGBOX_H