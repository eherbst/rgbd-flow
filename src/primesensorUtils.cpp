/*
 * primesensorUtils: for working with the primesensor specifically
 *
 * Evan Herbst
 * 9 / 22 / 10
 */

#include <cmath> //M_PI, sin(), fabs()
#include <stdexcept>
#include "rgbd_util/mathUtils.h" //sqr()
#include <Eigen/Core>
#include "rgbd_util/primesensorUtils.h"
using Eigen::Vector3f;

namespace primesensor
{

/*
 * be sure the camera you ask about is actually an rgb camera
 */
rgbd::CameraParams getColorCamParams(const rgbd::cameraID cam)
{
	rgbd::CameraParams camParams;
	switch(cam)
	{
		case rgbd::KINECT_640_DEFAULT:
			camParams.xRes = 640;
			camParams.yRes = 480;
			camParams.centerX = camParams.xRes / 2;
			camParams.centerY = camParams.yRes / 2;
			camParams.focalLength = 525.0f;
			camParams.fovX = camParams.fovY = 60 * M_PI / 180;
			camParams.stereoBaseline = .072;
			break;
		case rgbd::KINECT_320_DEFAULT:
			camParams.xRes = 320;
			camParams.yRes = 240;
			camParams.centerX = camParams.xRes / 2;
			camParams.centerY = camParams.yRes / 2;
			camParams.focalLength = 525.0f / (float) 2;
			camParams.fovX = camParams.fovY = 60 * M_PI / 180;
			camParams.stereoBaseline = .072;
			break;

		default: throw std::invalid_argument("unsupported cam ID");
	}
	return camParams;
}

/*
 * return the stereo depth error (difference in depth for a 1-pixel disparity difference) at depth z meters
 */
double stereoError(const double z)
{
	return sqr(z) / (1 / .00285/* from some modeling by Hao, Oct '10 */ + z);
}
/*
 * return the ratio of the stereo depth error at depth z meters to the error at depth 1 meter
 */
double stereoErrorRatio(const double z)
{
	static const double err1 = stereoError(1);
	return stereoError(z) / err1;
}
/*
 * return: 1-meter equivalent depth difference
 */
double normalizedDepthDistance(const double z1, const double z2)
{
	return fabs(z1 - z2) / stereoErrorRatio((z1 + z2) / 2);
}

} //namespace
