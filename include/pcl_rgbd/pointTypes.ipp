/*
 * pointTypes: point structs for use with pcl
 *
 * Evan Herbst
 * 6 / 17 / 10
 */

#include <cassert>
#include "rgbd_util/assert.h"

namespace rgbd
{

template <typename T>
boost::array<T, 3> unpackRGB(const float rgb)
{
	ASSERT_ALWAYS(false && "please specialize for your result type");
}

/*
 * provide point info in various Eigen formats via partial specialization
 */

template <typename EigenPointT, typename PointT>
struct ptX2eigenFunctor
{
	EigenPointT operator () (const PointT& p) const {ASSERT_ALWAYS(false && "please specialize for your result type");}
};
template <typename PointT>
struct ptX2eigenFunctor<rgbd::eigen::Vector3f, PointT>
{
	rgbd::eigen::Vector3f operator () (const PointT& p) const {return rgbd::eigen::Vector3f(p.x, p.y, p.z);}
};
template <typename PointT>
struct ptX2eigenFunctor<rgbd::eigen::Vector4f, PointT>
{
	rgbd::eigen::Vector4f operator () (const PointT& p) const {return rgbd::eigen::Vector4f(p.x, p.y, p.z, 1);}
};

template <typename EigenPointT, typename PointT>
EigenPointT ptX2eigen(const PointT& p)
{
	ptX2eigenFunctor<EigenPointT, PointT> f;
	return f(p);
}

template <typename EigenPointT, typename PointT>
struct ptNormal2eigenFunctor
{
	EigenPointT operator () (const PointT& p) const {ASSERT_ALWAYS(false && "please specialize for your result type");}
};
template <typename PointT>
struct ptNormal2eigenFunctor<rgbd::eigen::Vector3f, PointT>
{
	rgbd::eigen::Vector3f operator () (const PointT& p) const {return rgbd::eigen::Vector3f(p.normal[0], p.normal[1], p.normal[2]);}
};
template <typename PointT>
struct ptNormal2eigenFunctor<rgbd::eigen::Vector4f, PointT>
{
	rgbd::eigen::Vector4f operator () (const PointT& p) const {return rgbd::eigen::Vector4f(p.normal[0], p.normal[1], p.normal[2], 0);}
};

template <typename EigenPointT, typename PointT>
EigenPointT ptNormal2eigen(const PointT& p)
{
	ptNormal2eigenFunctor<EigenPointT, PointT> f;
	return f(p);
}

} //namespace
