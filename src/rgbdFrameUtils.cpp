/*
 * rgbdFrameUtils
 *
 * Evan Herbst
 * 10 / 12 / 12
 */

#include <opencv2/imgproc/imgproc.hpp>
#include "rgbd_flow/rgbdFrameUtils.h"

void downsizeFrame(cv::Mat& img, cv::Mat_<float>& depth)
{
	cv::Mat imgSmall;
	cv::resize(img, imgSmall, cv::Size(img.cols / 2, img.rows / 2));

//enable if your depth maps don't have invalid values
#ifdef SMOOTH_DEPTH
	cv::Mat depthImg(depth.height, depth.width, cv::DataType<float>::type);
	for(int i = 0, l = 0; i < depth.height; i++)
		for(int j = 0; j < depth.width; j++, l++)
		{
			depthImg.at<float>(i, j) = depth.float_data[l];
		}
	cv::Mat depthImgSmall;
	cv::resize(depthImg, depthImgSmall, cv::Size(img.cols / 2, img.rows / 2));
#endif

	cv::Mat_<float> depthSmall(depth.rows / 2, depth.cols / 2);
	for(int i = 0, ll = 0; i < depthSmall.rows; i++)
		for(int j = 0; j < depthSmall.cols; j++, ll++)
		{
#ifdef SMOOTH_DEPTH
			depthSmall.float_data[ll] = depthImgSmall.at<float>(i, j);
#else
			//without smoothing
			depthSmall(i, j) = depth(i * 2, j * 2);
#endif
		}

	img = imgSmall;
	depth = depthSmall;
}
