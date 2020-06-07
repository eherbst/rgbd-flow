/*
 * CameraParams.h
 *
 *  Created on: 2010-06-07
 *      Author: mkrainin
 */

#ifndef CAMERAPARAMS_H_
#define CAMERAPARAMS_H_

namespace rgbd
{

/*
 * params for a single (rgb, ir or other) camera
 */
struct CameraParams
{
	unsigned int xRes; //in pixels
	unsigned int yRes; //in pixels
	float centerX; //in pixels
	float centerY; //in pixels
	float focalLength; //in pixels

	float stereoBaseline; //for a depth camera, distance between IR projector and IR camera, in meters

	float fovX, fovY; //radians
};

} //namespace

#endif
