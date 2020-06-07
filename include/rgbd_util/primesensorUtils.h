/*
 * primesensorUtils: for working with the primesensor specifically
 *
 * Evan Herbst
 * 9 / 22 / 10
 */

#ifndef EX_PRIMESENSOR_UTILS_H
#define EX_PRIMESENSOR_UTILS_H

#include <string>
#include "rgbd_util/CameraParams.h"

namespace rgbd
{

enum cameraID
{
	INVALID_CAM, //meant to always throw errors if used at runtime
	//rgbd cameras
	KINECT_640_DEFAULT,
	KINECT_320_DEFAULT
};

struct cameraSetup
{
	cameraSetup() : cam(INVALID_CAM), hiresCam(INVALID_CAM), hiresRig(INVALID_CAM)
	{}

	/*
	 * required
	 */
	cameraID cam;
	/*
	 * optional: high-resolution cam
	 */
	cameraID hiresCam, hiresRig;
};

} //namespace

namespace primesensor
{

/*
 * be sure the camera you ask about is actually an rgb camera
 */
rgbd::CameraParams getColorCamParams(const rgbd::cameraID cam);

/*
 * return the stereo depth error (difference in depth for a 1-pixel disparity difference) at depth z meters
 */
double stereoError(const double z);
/*
 * return the ratio of the stereo depth error at depth z meters to the error at depth 1 meter
 */
double stereoErrorRatio(const double z);
/*
 * return: 1-meter equivalent depth difference
 */
double normalizedDepthDistance(const double z1, const double z2);

} //namespace

#endif //header
