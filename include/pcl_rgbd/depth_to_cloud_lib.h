/*
 * depth_to_cloud_lib.h
 *
 *  Created on: Jan 15, 2010
 *      Author: peter
 */

#ifndef EX_PCL_DEPTH_TO_CLOUD_LIB_H_
#define EX_PCL_DEPTH_TO_CLOUD_LIB_H_

#include <opencv2/core/core.hpp>
#include <pcl/point_cloud.h>
#include "rgbd_util/CameraParams.h"
#include "rgbd_util/eigen/Geometry"

/**
 * cloud holds the result
 *
 * When matching depth maps and images are received, the conversion from projective
 * coordinates to real-world coordinates is performed, RGB information
 * is looked up in the image map and associated with the points, and the
 * resulting PointCloud is returned.
 *
 * @param include_xy_channels whether to include image_{x,y} channels in the cloud (you want to do this if you're creating an unorganized cloud but will convert it to an organized later)
 * @param publish_all_points whether to include in the cloud points with negative depth (if yes, the cloud is "organized"; if not, it's "unorganized")
 *
 * if the cloud is organized, points will appear in row-major order
 *
 * @return true
 *
 * throw on any error
 */
template <typename PointT>
void depth_to_cloud(
		const cv::Mat_<float>& depthImg,
		bool include_xy_channels,
		bool publish_all_points,
		pcl::PointCloud<PointT>& cloud,
		const rgbd::CameraParams& depthCamParams);

#include "depth_to_cloud_lib.ipp"

#endif //header
