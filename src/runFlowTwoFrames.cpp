/*
 * runFlowTwoFrames: choose from a large set of flow algorithms
 *
 * Evan Herbst
 * 9 / 5 / 12
 */

#include <opencv2/imgproc/imgproc.hpp>
#include "rgbd_frame_common/staticDepthNoiseModeling.h"
#include "celiu_flow/ceLiuOpticalFlow.h"
#include "rgbd_flow/runFlowTwoFrames.h"
using std::string;

rgbd::CameraParams getResizedCamParams(const rgbd::CameraParams& initialParams, const cv::Size& sz)
{
	rgbd::CameraParams camParams = initialParams;
	const double sizeFactor = (double)sz.width / initialParams.xRes;
	camParams.xRes = rint(camParams.xRes * sizeFactor);
	camParams.yRes = rint(camParams.yRes * sizeFactor);
	camParams.centerX = rint(camParams.centerX * sizeFactor);
	camParams.centerY = rint(camParams.centerY * sizeFactor);
	camParams.focalLength *= sizeFactor;
	return camParams;
}

/*************************************************************************************************************************************************/

/*
 * Ce Liu's optical flow code
 */
cv::Mat_<cv::Vec2f> runFlowTwoFramesCeLiu(const cv::Mat& prevImg, const cv::Mat_<float>& prevDepth, const cv::Mat& curImg, const cv::Mat_<float>& curDepth,
	const double smoothnessWeight, const boost::optional<fs::path>& outdirPlusFilebase)
{
	cv::Mat_<cv::Vec2f> flow;

	OpticalFlow::opticalFlowParams params;
	params.variant = OpticalFlow::opticalFlowVariant::CELIU;
	params.alpha = smoothnessWeight;
	params.nOuterFPIterations = 10;
	params.nInnerFPIterations = 1;
	params.nSORIterations = 400;
	flow = ceLiuOpticalFlow(prevImg, curImg, params);

	return flow;
}

/*
 * celiu optical flow on depth maps
 */
cv::Mat_<cv::Vec2f> runFlowTwoFramesCeLiuOnDepth(const cv::Mat& prevImg, const cv::Mat_<float>& prevDepth, const cv::Mat& curImg, const cv::Mat_<float>& curDepth,
	const double smoothnessWeight, const boost::optional<fs::path>& outdirPlusFilebase)
{
	cv::Mat_<cv::Vec2f> flow;

	OpticalFlow::opticalFlowParams params;
	params.variant = OpticalFlow::opticalFlowVariant::CELIU;
	params.alpha = smoothnessWeight;
	params.nOuterFPIterations = 10;
	params.nInnerFPIterations = 1;
	params.nSORIterations = 400;
	flow = ceLiuOpticalFlowDepth(prevDepth, curDepth, params);

	return flow;
}

void setUpSceneFlowParams(const rgbd::CameraParams& camParams, const cv::Mat& prevImg, const cv::Mat_<float>& prevDepth, const cv::Mat& curImg, const cv::Mat_<float>& curDepth,
	OpticalFlow::sceneFlowParams& params, const double smoothnessWeight, const double depthVsColorDataWeight = 1, const OpticalFlow::sceneFlowRegularizationType regularizationType = OpticalFlow::sceneFlowRegularizationType::HPB,
	const uint32_t maxIters = 10000)
{
	params.regularizationType = regularizationType;
	params.camParams = camParams;
	params.minWidth = 60;//prevDepth.cols / 3; //TODO ?; now I'm using a non-fixed one because oversmoothing in the pyramid can cause bizarre boundary effects like large areas of points leaping around in 3-d
	params.colorDataWeight = 1; //ensure 1 when not experimenting
	params.depthVsColorDataWeight = depthVsColorDataWeight; //1 works much better than 10 on twoObjsMove1 and robotPushMultiobj1; 10 better on whiteboard and jaech4
	params.alpha = smoothnessWeight;
	params.gamma = 1e-5;//0; //TODO parameterize; 20121219 I have this turned on to deal with small bits of valid depth at edges of frame surrounded by a wall of hpb; these areas tend to get large flow
	params.nOuterFPIterations = 50;//100;//20; //run to convergence or to a max of this many relinearizations; TODO parameterize; 20130312 too many relinearizations causes instability! -- ??
	params.nInnerFPIterations = 1;
	params.nSORIterations = maxIters; //a max; 20130828 I need thousands for really good results but 100 or 1000 maybe sometimes enough for approx stuff like for use with moseg; TODO parameterize
#if 0 //to use nonrobust penalties
	params.variant = OpticalFlow::sceneFlowVariant::NONROBUST;
#endif

	params.depthMap = prevDepth;

	/*
	 * depth uncertainty
	 */
	boost::multi_array<float, 2> prevDepthSigmas, curDepthSigmas;
	computeDepthMapStdevsPointwise(prevDepth, prevDepthSigmas, 4/* distance to invalid */, true/* multithread */);
	computeDepthMapStdevsPointwise(curDepth, curDepthSigmas, 4/* distance to invalid */, true/* multithread */);
	params.depthSigma1.create(prevDepth.rows, prevDepth.cols);
	params.depthSigma2.create(prevDepth.rows, prevDepth.cols);
	for(int32_t i = 0; i < prevDepth.rows; i++)
		for(int32_t j = 0; j < prevDepth.cols; j++)
		{
			params.depthSigma1(i, j) = prevDepthSigmas[i][j];
			params.depthSigma2(i, j) = curDepthSigmas[i][j];
		}
}

/*
 * current best scene flow algorithm
 *
 * estFlow: initial estimate
 *
 * if estFlow, initialize with an existing flow and run at only the largest pyramid level
 */
cv::Mat_<cv::Vec3f> runFlowTwoFramesSceneFlowExperiments(const rgbd::CameraParams& fullSizeCamParams, const cv::Mat& prevImg, const cv::Mat_<float>& prevDepth, const cv::Mat& curImg, const cv::Mat_<float>& curDepth,
	const double smoothnessWeight, const double depthVsColorDataWeight, const OpticalFlow::sceneFlowRegularizationType regularizationType, const uint32_t maxIters, const boost::optional<cv::Mat_<cv::Vec3f>>& estFlow, const boost::optional<fs::path>& outdirPlusFilebase)
{
	const rgbd::CameraParams camParams = getResizedCamParams(fullSizeCamParams, prevImg.size());
	cv::Mat_<cv::Vec3f> flow;

	OpticalFlow::sceneFlowParams params;
	setUpSceneFlowParams(camParams, prevImg, prevDepth, curImg, curDepth, params, smoothnessWeight, depthVsColorDataWeight, regularizationType, maxIters);

	flow = ceLiuSceneFlow(prevImg, prevDepth, curImg, curDepth, estFlow, params, outdirPlusFilebase);

	return flow;
}
