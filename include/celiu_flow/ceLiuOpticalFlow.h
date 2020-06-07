/*
 * ceLiuOpticalFlow: wrapper around Liu's code to provide opencv data structures
 *
 * Evan Herbst
 * 7 / 24 / 12
 */

#ifndef EX_CE_LIU_FLOW_H
#define EX_CE_LIU_FLOW_H

#include <boost/optional.hpp>
#include <boost/filesystem/path.hpp>
#include <opencv2/core/core.hpp>
#include "celiu_flow/OpticalFlow.h"
namespace fs = boost::filesystem;

cv::Mat_<cv::Vec2f> ceLiuOpticalFlow(const cv::Mat& img1, const cv::Mat& img2, const OpticalFlow::opticalFlowParams& params);

/*
 * run on depth imgs (we'll do the normalization)
 */
cv::Mat_<cv::Vec2f> ceLiuOpticalFlowDepth(const cv::Mat_<float>& img1, const cv::Mat_<float>& img2, const OpticalFlow::opticalFlowParams& params);

/*
 * compute scene flow by computing optical flow, then following it and sampling the second depth map
 */
cv::Mat_<cv::Vec3f> ceLiuSceneFlowFrom2DFlow(const cv::Mat& img1, const cv::Mat_<float>& depth1, const cv::Mat& img2, const cv::Mat_<float>& depth2, const OpticalFlow::opticalFlowParams& params);

/*
 * estFlow: initial estimate; if given, used to initialize
 */
cv::Mat_<cv::Vec3f> ceLiuSceneFlow(const cv::Mat& img1, const cv::Mat_<float>& depth1, const cv::Mat& img2, const cv::Mat_<float>& depth2, const boost::optional<cv::Mat_<cv::Vec3f>>& estFlow, const OpticalFlow::sceneFlowParams& params,
	const boost::optional<fs::path>& outdirPlusFilebase);

/*
 * compute per-pixel energy of a flow field
 */
cv::Mat_<float> computeSceneFlowDataTermEnergy(const cv::Mat& img1, const cv::Mat_<float>& depth1, const cv::Mat& img2, const cv::Mat_<float>& depth2, const OpticalFlow::sceneFlowParams& params, const cv::Mat_<cv::Vec3f>& flow);

#endif //header
