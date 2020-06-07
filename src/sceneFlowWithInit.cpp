#include "rgbd_util/primesensorUtils.h"
#include "celiu_flow/heuristicProbBoundary.h"

/*
 * compute 3-d flow at just the largest scale given an initialization
 */
void OpticalFlow::sceneFlowWithInit(const DImage& Im1, const DImage& dm1, const DImage& Im2, const DImage& dm2, const sceneFlowParams& params, const DImage& estFlow, DImage& u, DImage& v, DImage& w, const boost::optional<fs::path>& outdirPlusFilebase)
{
	ASSERT_ALWAYS(estFlow.nchannels() == 3);

	u.allocate(estFlow.width(), estFlow.height());
	v.allocate(estFlow.width(), estFlow.height());
	w.allocate(estFlow.width(), estFlow.height());
	for(int32_t i = 0, l = 0; i < Im1.height(); i++)
		for(int32_t j = 0; j < Im1.width(); j++, l++)
		{
			u.data()[l] = estFlow.data()[l * 3 + 0];
			v.data()[l] = estFlow.data()[l * 3 + 1];
			w.data()[l] = estFlow.data()[l * 3 + 2];
		}

	DImage depth1 = dm1, depth2 = dm2;
	//fill in invalid depths
	recursiveMedianFilter(depth1);
	recursiveMedianFilter(depth2);

	//at each pixel, a weight for the depth data constraint (basically a depth uncertainty)
	DImage depthDataTermWeights;
	calculateDepthDataTermWeights(params, depthDataTermWeights);

	//at each pixel, a weight for the depth smoothness term (basically a depth uncertainty)
	DImage depthSmoothnessTermWeights;
	calculateDepthSmoothnessTermWeights(params, depthSmoothnessTermWeights);

	DImage Image1,Image2,WarpImage2;
	DImage WarpDepth2;

	/*
	 * warp im1/dm1 with the initial flow
	 */
	im2feature(Image1, Im1, constancyType::RGB_WITH_GRADIENT);
	im2feature(Image2, Im2, constancyType::RGB_WITH_GRADIENT);
	warpBy3DFlow(Image1, depth1, Image2, depth2, WarpImage2, WarpDepth2, u, v, w);

	/*
	 * regularization weights
	 */
	const boost::multi_array<float, 3> hpb = computeHeuristicPb(Im1, dm1);
	DImage regularizationWeights;
	calculateRegularizationWeights(Image1, depth1, params, hpb, regularizationWeights);

#if 0
//	if(outdirPlusFilebase)
	{
		cv::Mat_<uint8_t> hpbHorizImg(Im1.height(), Im1.width()), hpbVertImg(Im1.height(), Im1.width());
		for(int32_t i = 0; i < Im1.height(); i++)
			for(int32_t j = 0; j < Im1.width(); j++)
			{
				hpbHorizImg(i, j) = (uint8_t)clamp(255.0 * hpb[i][j][0], 0.0, 255.0);
				hpbVertImg(i, j) = (uint8_t)clamp(255.0 * hpb[i][j][1], 0.0, 255.0);
			}
		cv::imwrite(/*outdirPlusFilebase.get().string() + */(boost::format("-hpbHorizReopt.png")).str(), hpbHorizImg);
		cv::imwrite(/*outdirPlusFilebase.get().string() + */(boost::format("-hpbVertReopt.png")).str(), hpbVertImg);
	}
#endif

	//depth-component-specific weights for flow magnitude penalty
	DImage depthMagnitudeTermWeights; //to make xy- and z-flow weighted equally in depth-magnitude penalty term
	calculateDepthMagnitudeTermWeights(depth1, params, depthMagnitudeTermWeights);

	/*
	 * refine at one scale
	 */
	const bool allowEarlyTermination = false; //TODO ?
	sceneFlowSingleScale(params.camParams, Image1, depth1, Image2, depth2, WarpImage2, WarpDepth2, u, v, w,
		params.colorDataWeight, depthDataTermWeights,
		params.alpha, depthSmoothnessTermWeights,
		params.gamma, depthMagnitudeTermWeights,
		regularizationWeights,
		params.nOuterFPIterations, params.nInnerFPIterations, params.nSORIterations, allowEarlyTermination);
}
