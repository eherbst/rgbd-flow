/*
 * pointTypes: point structs for use with pcl
 *
 * Evan Herbst
 * 6 / 15 / 10
 */

#ifndef EX_PCL_POINT_TYPES_H
#define EX_PCL_POINT_TYPES_H

#include <boost/array.hpp>
#include <boost/mpl/bool.hpp>
#include <pcl/point_types.h>
#include <pcl/register_point_struct.h>
#include "rgbd_util/eigen/Core"
#include "rgbd_util/eigen/Geometry"

namespace rgbd
{

/*
 * ***** CAVEAT: pcl uses the c++ offsetof() macro on these point types, so they must be POD -- EVH 20100615
 */

/*
 * most of the fields we use
 */
struct EIGEN_ALIGN16 pt
{
  PCL_ADD_POINT4D;    // This adds the members x,y,z which can also be accessed using the point (which is float[4])
  union //if we defined this above and instantiated it as a field here, it'd have to be named
  {
		float rgb; //packed, BGRA
		struct
		{
			uint8_t b, g, r, a;
		};
  };
  PCL_ADD_NORMAL4D;   // This adds the member normal[3] which can also be accessed using the point (which is float[4])
  float curvature;
  uint32_t imgX, imgY;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
inline std::ostream& operator << (std::ostream& os, const pt& p)
{
  os << "(" << p.x << "," << p.y << "," << p.z << " - " << p.rgb << " - " << p.normal[0] << "," << p.normal[1] << "," << p.normal[2] << " - " << p.curvature << " (" << p.imgX << "," << p.imgY << ") " << ")";
  return (os);
}

} //namespace

/*
 * must register structs in the global namespace so the macros can use other namespace names
 */

POINT_CLOUD_REGISTER_POINT_STRUCT(
	rgbd::pt,
	(float, x, x)
	(float, y, y)
	(float, z, z)
	(float, rgb, rgb)
	(float, normal[0], nx)
	(float, normal[1], ny)
	(float, normal[2], nz)
	(float, curvature, curvature)
	(uint32_t, imgX, imgX)
	(uint32_t, imgY, imgY)
);

namespace rgbd
{

/**********************************************************************************************
 * type traits for points
 */

template <typename PointT> struct pointHasImgXY {typedef boost::mpl::bool_<false> type;};
template <> struct pointHasImgXY<rgbd::pt> {typedef boost::mpl::bool_<true> type;};

/**********************************************************************************************
 * convenience functions for dealing with the ridiculously inconvenient types above
 */

float packRGB(const unsigned char r, const unsigned char g, const unsigned char b);
float packRGB(const boost::array<unsigned char, 3> rgb);
float packRGB(const boost::array<float, 3> rgb); //args should be in [0, 1)
float packRGB(const rgbd::eigen::Vector3f& rgb); //args should be in [0, 1)

template <typename T>
boost::array<T, 3> unpackRGB(const float rgb);
template <>
boost::array<unsigned char, 3> unpackRGB(const float rgb);
template <>
boost::array<float, 3> unpackRGB(const float rgb); //outputs will be in [0, 1)
rgbd::eigen::Vector3f unpackRGB2eigen(const float rgb); //outputs will be in [0, 1)

template <typename PointT>
void setImgCoords(PointT& p, const int x, const int y) {} //by default assume the point type doesn't have img{X,Y}
template <>
inline void setImgCoords<rgbd::pt>(rgbd::pt& p, const int x, const int y) {p.imgX = x; p.imgY = y;}

//p.xyz = q
template <typename PointT>
void eigen2ptX(PointT& p, const rgbd::eigen::Vector3f& q)
{
	p.x = q.x();
	p.y = q.y();
	p.z = q.z();
}
template <typename PointT>
void eigen2ptX(PointT& p, const rgbd::eigen::Vector4f& q)
{
	p.x = q.x();
	p.y = q.y();
	p.z = q.z();
}

//p.n{xyz} = q
template <typename PointT>
void eigen2ptNormal(PointT& p, const rgbd::eigen::Vector3f& q)
{
	p.normal[0] = q.x();
	p.normal[1] = q.y();
	p.normal[2] = q.z();
}
template <typename PointT>
void eigen2ptNormal(PointT& p, const rgbd::eigen::Vector4f& q)
{
	p.normal[0] = q.x();
	p.normal[1] = q.y();
	p.normal[2] = q.z();
}

/*
 * EigenPointT should be Vector3f or Vector4f
 */
template <typename EigenPointT, typename PointT>
EigenPointT ptX2eigen(const PointT& p);
template <typename EigenPointT, typename PointT>
EigenPointT ptNormal2eigen(const PointT& p);

} //namespace

#include "pointTypes.ipp"

#endif //header
