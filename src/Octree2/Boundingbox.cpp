#include "boundingbox.h"

#include <algorithm>
#include <array>
#include <limits>
#include <cmath>
#include <iostream>
#include <utility>

using std::array;
using std::initializer_list;

BoundingBox::BoundingBox(BoundingBox&& rhs) {
  xhi = std::move(rhs.xhi);
  xlo = std::move(rhs.xlo);
  yhi = std::move(rhs.yhi);
  ylo = std::move(rhs.ylo);
  zhi = std::move(rhs.zhi);
  zlo = std::move(rhs.zlo);
}

BoundingBox::BoundingBox(initializer_list<double> l) {
  std::copy(l.begin(), l.end(), &xhi);
}

BoundingBox& BoundingBox::operator=(const BoundingBox& rhs) {
  std::copy(&rhs.xhi, &rhs.xhi + 6, &xhi);
  return *this;
}

BoundingBox& BoundingBox::operator=(BoundingBox&& rhs) {
  xhi = std::move(rhs.xhi);
  xlo = std::move(rhs.xlo);
  yhi = std::move(rhs.yhi);
  ylo = std::move(rhs.ylo);
  zhi = std::move(rhs.zhi);
  zlo = std::move(rhs.zlo);
  return *this;
}

bool BoundingBox::contains(const BoundingBox& other) const {
  return xlo <= other.xlo && xhi >= other.xhi &&
         ylo <= other.ylo && yhi >= other.yhi &&
         zlo <= other.zlo && zhi >= other.zhi;
}

bool BoundingBox::contains(const std::array<double, 3>& point) const {
  return xlo <= point[0] && xhi > point[0] &&
         ylo <= point[1] && yhi > point[1] &&
         zlo <= point[2] && zhi > point[2];
}

bool BoundingBox::overlap(const BoundingBox& other, BoundingBox* out) const {
  // trivial cases
  if (contains(other)) {
    *out = other;
    return true;
  } else if (other.contains(*this)) {
    *out = *this;
    return true;
  } 

  // Check if there is no intersection
  if (xhi < other.xlo || xlo > other.xhi ||
      yhi < other.ylo || ylo > other.yhi ||
      zhi < other.zlo || zlo > other.zhi) {
    *out = invalid;
    return false;
  }

  // Actually calculate the bounds
  double upperX = std::min(xhi, other.xhi);
  double upperY = std::min(yhi, other.yhi);
  double upperZ = std::min(zhi, other.zhi);

  double lowerX = std::max(xlo, other.xlo);
  double lowerY = std::max(ylo, other.ylo);
  double lowerZ = std::max(zlo, other.zlo);

  *out = BoundingBox{upperX, lowerX, upperY, lowerY, upperZ, lowerZ};
  return true;
}

array<BoundingBox, 8> BoundingBox::partition() const {
  double xmid = (xhi - xlo) / 2.;
  double ymid = (yhi - ylo) / 2.;
  double zmid = (zhi - zlo) / 2.;

  std::array<BoundingBox, 8> ret{{
    BoundingBox{xmid, xlo, ymid, ylo, zmid, zlo}, // bottom left front
    BoundingBox{xhi, xmid, ymid, ylo, zmid, zlo}, // bottom right front
    BoundingBox{xmid, xlo, yhi, ymid, zmid, zlo}, // bottom left back
    BoundingBox{xhi, xmid, yhi, ymid, zmid, zlo}, // bottom right back
    BoundingBox{xmid, xlo, ymid, ylo, zhi, zmid}, // top left front
    BoundingBox{xhi, xmid, ymid, ylo, zhi, zmid}, // top right front
    BoundingBox{xmid, xlo, yhi, ymid, zhi, zmid}, // top left back
    BoundingBox{xhi, xmid, yhi, ymid, zhi, zmid}  // top right back
  }};
  return ret;
}

bool BoundingBox::operator==(const BoundingBox& rhs) const {
  // They're all equal, or they're all NaNs
  return (xhi == rhs.xhi && xlo == rhs.xlo &&
          yhi == rhs.yhi && ylo == rhs.ylo &&
          zhi == rhs.zhi && zlo == rhs.zlo) ||
         (std::isnan(xhi) && std::isnan(rhs.xhi) &&
          std::isnan(xlo) && std::isnan(rhs.xlo) &&
          std::isnan(yhi) && std::isnan(rhs.yhi) &&
          std::isnan(ylo) && std::isnan(rhs.ylo) &&
          std::isnan(zhi) && std::isnan(rhs.zhi) &&
          std::isnan(zlo) && std::isnan(rhs.zlo));
}

// bool BoundingBox::operator == (const BoundingBox& rhs) const {
//     const bool allEqual = (mins == rhs.mins && maxs == rhs.maxs); #TODO check here: https://stackoverflow.com/questions/17333/what-is-the-most-effective-way-for-float-and-double-comparison
//     if (allEqual) return true;

//     const bool allNaN = (mins.isNaN() && rhs.mins.isNaN() && 
//                          maxs.isNaN() && rhs.maxs.isNaN());
//     if (allNaN) return true;

//     return false;
// }

bool BoundingBox::operator!=(const BoundingBox& rhs) const {
  return !operator==(rhs);
}

std::ostream& operator<<(std::ostream& stream, const BoundingBox& rhs) {
  stream << "{" << rhs.xhi << ", " << rhs.xlo << ", "
                << rhs.yhi << ", " << rhs.ylo << ", "
                << rhs.zhi << ", " << rhs.zlo << ", "
         << "}";
  return stream;
}