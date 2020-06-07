/*
 * depth_to_cloud_lib
 */

#include <exception>
#include "rgbd_util/eigen/Geometry"
#include "pcl_rgbd/pointTypes.h"

/*
 * throw on any error
 */
template <typename PointT>
void depth_to_cloud(
		const cv::Mat_<float>& depthImg,
		bool include_xy_channels,
		bool publish_all_points,
		pcl::PointCloud<PointT>& cloud,
		const rgbd::CameraParams& depthCamParams)
{
	//presize the points vector
	unsigned int numPoints;
	if(publish_all_points) numPoints = depthImg.cols * depthImg.rows;
	else
	{
		numPoints = 0;
		for(int row=0, i = 0; row < depthImg.rows; row++)
		  for(int col=0; col < depthImg.cols; col++, i++)
		  {
				const float depth = depthImg(row, col);
				if(depth > 0) numPoints++;
		  }
	}
	cloud.points.resize(numPoints);

	for(int row=0, i = 0, j = 0; row < depthImg.rows; row++) {
	  for(int col=0; col < depthImg.cols; col++, j++) {
		  const float depth = depthImg(row, col);
			if(depth > 0 || publish_all_points)
			{
				PointT& p = cloud.points[i];
				p.x = (((float)col - depthCamParams.centerX) * depth / depthCamParams.focalLength);
				p.y = (((float)row - depthCamParams.centerY) * depth / depthCamParams.focalLength);
				p.z = depth;
				p.rgb = 0; //black
				if(include_xy_channels) rgbd::setImgCoords<PointT>(p, col, row);

				i++;
			}
	  }
	}

	if (publish_all_points) {
		cloud.width = depthImg.cols;
		cloud.height = depthImg.rows;
	}
	else {
		cloud.width = cloud.points.size();
		cloud.height = 1;
	}
}
