/*
 * range_adaptor: allow eigen Vectors to satisfy boost::RandomAccessRangeConcept
 *
 * Evan Herbst
 * 2 / 5 / 11
 */

#ifndef EX_EIGEN_RANGE_ADAPTOR_H
#define EX_EIGEN_RANGE_ADAPTOR_H

#include <boost/range.hpp>
#include "rgbd_util/eigen/Geometry"

#define MAKE_RANGE_STUFF(EigenVectorT)\
namespace boost\
{\
template <>\
struct range_mutable_iterator<EigenVectorT> {typedef EigenVectorT::Scalar* type;};\
template <>\
struct range_const_iterator<EigenVectorT> {typedef const EigenVectorT::Scalar* type;};\
}/*namespace*/\
namespace Eigen3{\
boost::range_iterator<EigenVectorT>::type range_begin(EigenVectorT& v) {return v.data();}\
boost::range_iterator<EigenVectorT>::type range_end(EigenVectorT& v) {return v.data() + v.size();}\
boost::range_iterator<const EigenVectorT>::type range_begin(const EigenVectorT& v) {return v.data();}\
boost::range_iterator<const EigenVectorT>::type range_end(const EigenVectorT& v) {return v.data() + v.size();}\
}/*namespace*/

MAKE_RANGE_STUFF(rgbd::eigen::VectorXf)
MAKE_RANGE_STUFF(rgbd::eigen::VectorXd)
//TODO add fixed-length types as nec
#undef MAKE_RANGE_STUFF

#endif //header
