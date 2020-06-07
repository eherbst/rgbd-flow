/*
 * runFlowTwoFrames: choose from a large set of flow algorithms
 *
 * Evan Herbst
 * 9 / 5 / 12
 */

#ifndef EX_RUN_TWO_FRAME_FLOW_H
#define EX_RUN_TWO_FRAME_FLOW_H

#include <boost/optional.hpp>
#include <boost/filesystem/path.hpp>
#include <opencv2/core/core.hpp>
#include "rgbd_util/CameraParams.h"
#include "celiu_flow/OpticalFlow.h"
namespace fs = boost::filesystem;

/*
 * Ce Liu's optical flow code
 */
cv::Mat_<cv::Vec2f> runFlowTwoFramesCeLiu(const cv::Mat& prevImg, const cv::Mat_<float>& prevDepth, const cv::Mat& curImg, const cv::Mat_<float>& curDepth,
	const double smoothnessWeight, const boost::optional<fs::path>& outdirPlusFilebase);

/*
 * celiu optical flow on depth maps
 */
cv::Mat_<cv::Vec2f> runFlowTwoFramesCeLiuOnDepth(const cv::Mat& prevImg, const cv::Mat_<float>& prevDepth, const cv::Mat& curImg, const cv::Mat_<float>& curDepth,
	const double smoothnessWeight, const boost::optional<fs::path>& outdirPlusFilebase);

/*
 * celiu optical flow with my fiddling
 */
cv::Mat_<cv::Vec2f> runFlowTwoFramesCeliuExperiments(const cv::Mat& prevImg, const cv::Mat_<float>& prevDepth, const cv::Mat& curImg, const cv::Mat_<float>& curDepth,
	const double smoothnessWeight, const boost::optional<fs::path>& outdirPlusFilebase);

/*
 * run 2-d flow, then sample the depth map to get 3-d flow
 */
cv::Mat_<cv::Vec3f> runFlowTwoFramesCeliu2dto3d(const cv::Mat& prevImg, const cv::Mat_<float>& prevDepth, const cv::Mat& curImg, const cv::Mat_<float>& curDepth,
	const double smoothnessWeight, const boost::optional<fs::path>& outdirPlusFilebase);

/*
 * current best scene flow algorithm
 *
 * estFlow: initial estimate
 *
 * if estFlow, initialize with an existing flow and run at only the largest pyramid level
 */
cv::Mat_<cv::Vec3f> runFlowTwoFramesSceneFlowExperiments(const rgbd::CameraParams& fullSizeCamParams, const cv::Mat& prevImg, const cv::Mat_<float>& prevDepth, const cv::Mat& curImg, const cv::Mat_<float>& curDepth,
	const double smoothnessWeight, const double depthVsColorDataWeight, const OpticalFlow::sceneFlowRegularizationType regularizationType, const uint32_t maxIters,
	const boost::optional<cv::Mat_<cv::Vec3f>>& estFlow, const boost::optional<fs::path>& outdirPlusFilebase);

cv::Mat_<float> computeSceneFlowDataTermEnergy(const rgbd::CameraParams& fullSizeCamParams, const cv::Mat& prevImg, const cv::Mat_<float>& prevDepth, const cv::Mat& curImg, const cv::Mat_<float>& curDepth,
	const double smoothnessWeight, const double depthVsColorDataWeight, const OpticalFlow::sceneFlowRegularizationType regularizationType, const cv::Mat_<cv::Vec3f>& flow, const uint32_t maxIters);

#endif //header
