/*
 * pointTypes: point structs for use with pcl
 *
 * Evan Herbst
 * 6 / 16 / 10
 */

#include "pcl_rgbd/pointTypes.h"

namespace rgbd
{

float packRGB(const unsigned char r, const unsigned char g, const unsigned char b)
{
	const uint32_t rgbi = ((uint32_t)r << 16) | ((uint32_t)g << 8) | (uint32_t)b;
	return *reinterpret_cast<const float*>(&rgbi);
}
float packRGB(const boost::array<unsigned char, 3> rgb)
{
	const uint32_t rgbi = ((uint32_t)rgb[0] << 16) | ((uint32_t)rgb[1] << 8) | (uint32_t)rgb[2];
	return *reinterpret_cast<const float*>(&rgbi);
}
float packRGB(const boost::array<float, 3> rgb) //args should be in [0, 1)
{
	const uint32_t rgbi = ((uint32_t)(255 * rgb[0]) << 16) | ((uint32_t)(255 * rgb[1]) << 8) | (uint32_t)(255 * rgb[2]);
	return *reinterpret_cast<const float*>(&rgbi);
}
float packRGB(const rgbd::eigen::Vector3f& rgb) //args should be in [0, 1)
{
	const uint32_t rgbi = ((uint32_t)(255 * rgb[0]) << 16) | ((uint32_t)(255 * rgb[1]) << 8) | (uint32_t)(255 * rgb[2]);
	return *reinterpret_cast<const float*>(&rgbi);
}

template <>
boost::array<unsigned char, 3> unpackRGB(const float rgb)
{
	const uint32_t rgbi = *reinterpret_cast<const uint32_t*>(&rgb);
	const boost::array<unsigned char, 3> a = {{(unsigned char)((rgbi >> 16) & 0xff), (unsigned char)((rgbi >> 8) & 0xff), (unsigned char)(rgbi & 0xff)}};
	return a;
}
template <>
boost::array<float, 3> unpackRGB(const float rgb) //outputs will be in [0, 1)
{
	const uint32_t rgbi = *reinterpret_cast<const uint32_t*>(&rgb);
	const boost::array<float, 3> a = {{(float)((rgbi >> 16) & 0xff) / 255, (float)((rgbi >> 8) & 0xff) / 255, (float)(rgbi & 0xff) / 255}};
	return a;
}
rgbd::eigen::Vector3f unpackRGB2eigen(const float rgb) //outputs will be in [0, 1)
{
	const uint32_t rgbi = *reinterpret_cast<const uint32_t*>(&rgb);
	const rgbd::eigen::Vector3f a((float)((rgbi >> 16) & 0xff) / 255, (float)((rgbi >> 8) & 0xff) / 255, (float)(rgbi & 0xff) / 255);
	return a;
}

} //namespace
