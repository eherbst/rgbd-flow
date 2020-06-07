/*
 * CameraParams
 *
 * Evan Herbst
 * 11 / 2 / 11
 */

#include <algorithm> //fill()
#include <sensor_msgs/CameraInfo.h>
#include "rgbd_util/CameraParams.h"

namespace rgbd
{

sensor_msgs::CameraInfo camParams2cameraInfo(const CameraParams& p)
{
	sensor_msgs::CameraInfo info;
	info.header.seq = 0;
	info.header.frame_id = "<invalid>";
	info.width = p.xRes;
	info.height = p.yRes;
	std::fill(info.D.begin(),info.D.end(),0);
	std::fill(info.K.begin(),info.K.end(),0);
	std::fill(info.R.begin(),info.R.end(),0);
	std::fill(info.P.begin(),info.P.end(),0);
	info.K[0] = info.K[4] = info.P[0] = info.P[5] = p.focalLength;
	info.K[2] = info.P[2] = p.centerX;
	info.K[5] = info.P[6] = p.centerY;
	info.K[8] = info.R[0] = info.R[4] = info.R[8] = info.P[10] = 1.0;
	return info;
}

} //namespace
