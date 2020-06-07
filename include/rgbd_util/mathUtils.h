/*
 * mathUtils
 *
 * define RGBD_MATHUTILS_DISABLE_CXX0X to include this file from another that can't use c++0x (eg something being compiled by the CUDA compiler, nvcc)
 *
 * Evan Herbst
 * 3 / 3 / 10
 */

#ifndef EX_RGBD_MATH_UTILS_H
#define EX_RGBD_MATH_UTILS_H

#include <vector>
#ifndef RGBD_MATHUTILS_DISABLE_CXX0X
#include <tuple>
#endif
#include <utility>
#include <boost/functional/hash.hpp>

template <typename T> T sqr(T d) {return d * d;}

/*
 * absolute value
 */
double dabs(double d);

/*
 * linear interpolation
 *
 * alpha: in [0, 1]
 */
double linterp(const double v0, const double v1, const double alpha);

/****************************************************************************************
 * hash functions for containers
 */

#ifndef RGBD_MATHUTILS_DISABLE_CXX0X
namespace std
{

template <typename T1, typename T2>
struct hash<std::pair<T1, T2> > : public boost::hash<std::pair<T1, T2> > {};

} //namespace
#endif

#include "mathUtils.ipp"

#endif //header
