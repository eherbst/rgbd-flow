/*
 * heuristicProbBoundary: cues for spatial regularization weights
 *
 * Evan Herbst
 * 8 / 29 / 12
 */

#include "boundary_probs_heuristic/heuristicBoundaries.h"
#include "celiu_flow/heuristicProbBoundary.h"

/*
 * not Malik's Pb, but a very heuristic rough boundary-prob map to identify all possible boundary pts in an rgbd frame
 *
 * img: first 3 channels should be rgb
 *
 * depth should not have been renormalized (eg to [0, 1])
 *
 * return: p(boundary) for each pixel pair in row-major order
 * (first two output dimensions are y, x; third is [right, down] nbrs)
 */
boost::multi_array<float, 3> computeHeuristicPb(const DImage& img, const DImage& depth)
{
	//compute params specific to the kinect; TODO parameterize?
	rgbd::CameraParams camParams;
	camParams.xRes = img.width();
	camParams.yRes = img.height();
	camParams.centerX = camParams.xRes / 2;
	camParams.centerY = camParams.yRes / 2;
	camParams.focalLength = 525.0 * camParams.xRes / 640;

	cv::Mat_<cv::Vec3b> cvImg(img.height(), img.width());
	for(uint32_t i = 0, l = 0; i < img.height(); i++)
		for(uint32_t j = 0; j < img.width(); j++, l++)
			cvImg(i, j) = cv::Vec3b(255 * img.data()[l * img.nchannels() + 2], 255 * img.data()[l * img.nchannels() + 1], 255 * img.data()[l * img.nchannels() + 0]);
	cv::Mat_<float> cvDepth(depth.height(), depth.width());
	for(uint32_t i = 0, l = 0; i < img.height(); i++)
		for(uint32_t j = 0; j < img.width(); j++, l++)
			cvDepth(i, j) = depth.data()[l];

	return computeHeuristicPb(camParams, cvImg, cvDepth);
}
