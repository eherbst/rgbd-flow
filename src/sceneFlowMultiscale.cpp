#include <cstdlib>
#include <array>
#include <iostream>
#include <fstream>
#include <opencv2/highgui/highgui.hpp>
#include "rgbd_util/timer.h"
#include "rgbd_util/primesensorUtils.h"
#include "rgbd_util/threadPool.h"
#include "optical_flow_utils/sceneFlowIO.h"
#include "celiu_flow/depthMapPyramid.h"
#include "celiu_flow/heuristicProbBoundary.h"
using std::cout;
using std::endl;
using std::ofstream;

rgbd::CameraParams getDownsizedCamParams(rgbd::CameraParams camParams, const cv::Size& sz)
{
	const double sizeFactor = (double)sz.width / camParams.xRes;
	camParams.xRes = rint(camParams.xRes * sizeFactor);
	camParams.yRes = rint(camParams.yRes * sizeFactor);
	camParams.centerX = rint(camParams.centerX * sizeFactor);
	camParams.centerY = rint(camParams.centerY * sizeFactor);
	camParams.focalLength *= sizeFactor;
	return camParams;
}

/*
 * compute 3-d flow over the pyramid
 */
void OpticalFlow::sceneFlowMultiscale(const DImage& Im1, const DImage& dm1, const DImage& Im2, const DImage& dm2, const sceneFlowParams& params, DImage& u, DImage& v, DImage& w, const boost::optional<fs::path>& outdirPlusFilebase)
{
	rgbd::timer t;
	const double ratio = params.pyramidRatio;
	GaussianPyramid GPyramidIm1, GPyramidIm2;
	GaussianPyramid gPyramidDepth1, gPyramidDepth2;
	GaussianPyramid gPyramidDepthDataTermWeights, gPyramidDepthSmoothnessTermWeights; //TODO use depthMapPyramids for these?
	DepthMapPyramid rawDepthPyramid; //for regularization weights

	//compute each pyramid in a separate thread
{
	rgbd::threadGroup tg(7);
	tg.addTask([&]()
		{
			GPyramidIm1.ConstructPyramid(Im1, ratio, params.minWidth);
		});
	tg.addTask([&]()
		{
			GPyramidIm2.ConstructPyramid(Im2, ratio, params.minWidth);
		});
	tg.addTask([&]()
{
	DImage depth1 = dm1; //deep copy
	//fill in invalid depths
	recursiveMedianFilter(depth1);
	gPyramidDepth1.ConstructPyramid(depth1,ratio,params.minWidth);
});
	tg.addTask([&]()
{
	DImage depth2 = dm2; //deep copy
	//fill in invalid depths
	recursiveMedianFilter(depth2);
	gPyramidDepth2.ConstructPyramid(depth2,ratio,params.minWidth);
});
	tg.addTask([&]()
{
	//depthDataTermWeights: at each pixel, a weight for the depth data constraint (basically a depth uncertainty)
	DImage depthDataTermWeights;
	calculateDepthDataTermWeights(params, depthDataTermWeights);
	gPyramidDepthDataTermWeights.ConstructPyramid(depthDataTermWeights, ratio, params.minWidth);
});
	tg.addTask([&]()
{
	//depthSmoothnessTermWeights: at each pixel, a weight for the depth smoothness term (basically a depth uncertainty)
	DImage depthSmoothnessTermWeights;
	calculateDepthSmoothnessTermWeights(params, depthSmoothnessTermWeights);
	gPyramidDepthSmoothnessTermWeights.ConstructPyramid(depthSmoothnessTermWeights, ratio, params.minWidth);
});
	tg.addTask([&]()
{
	DImage dm1Filtered = dm1; //deep copy
	recursiveMedianFilterSmallInvalidRegionsUsingColor(dm1Filtered, Im1, 60/* largest invalid region to fill in */);
	rawDepthPyramid.ConstructPyramid(dm1Filtered, ratio, params.minWidth);
});
	tg.wait();
}
	t.stop("compute pyramids");

	/*
	 * iterate over pyramid (main loop)
	 */
	t.restart();
	DImage Image1,Image2,WarpImage2;
	DImage Depth1, Depth2, WarpDepth2;
	for(int k=GPyramidIm1.numLevels()-1;k>=0;k--)
	{
		if(IsDisplay)
			cout<<"Pyramid level "<<k << endl;
		const int width=GPyramidIm1.image(k).width();
		const int height=GPyramidIm1.image(k).height();

		rgbd::timer t;
		im2feature(Image1, GPyramidIm1.image(k), constancyType::RGB_WITH_GRADIENT);
		im2feature(Image2, GPyramidIm2.image(k), constancyType::RGB_WITH_GRADIENT);
		//depth2feature equivalent
		Depth1.copyData(gPyramidDepth1.image(k));
		Depth2.copyData(gPyramidDepth2.image(k));
		t.stop("copy frame");

		const rgbd::CameraParams camParams = getDownsizedCamParams(params.camParams, cv::Size(Image1.width(), Image1.height()));

		/*
		 * regularization weights
		 */

		t.restart();
		const boost::multi_array<float, 3> hpb = computeHeuristicPb(GPyramidIm1.image(k), rawDepthPyramid.image(k));
		t.stop("compute hpb");

		t.restart();
		if(k==GPyramidIm1.numLevels()-1) // if at the smallest scale
		{
			u.allocate(width,height);
			v.allocate(width,height);
			w.allocate(width, height);
			WarpImage2.copyData(Image2);
			WarpDepth2.copyData(Depth2);
		}
		else
		{
			u.imresize(width,height);
			v.imresize(width,height);
			w.imresize(width, height);
			u.Multiplywith(1/ratio);
			v.Multiplywith(1/ratio);
			warpBy3DFlow(Image1, Depth1, Image2, Depth2, WarpImage2, WarpDepth2, u, v, w);
			//TODO also create a depth-uncertainty map for WarpDepth2, and include those values in the depth data weights?
		}
		t.stop("resize pyramid");

		t.restart();

		//depth-component-specific weights for flow magnitude penalty
		DImage depthMagnitudeTermWeights; //to make xy- and z-flow weighted equally in depth-magnitude penalty term
		calculateDepthMagnitudeTermWeights(Depth1, params, depthMagnitudeTermWeights);

		DImage regularizationWeights;
		calculateRegularizationWeights(Image1, Depth1, params, hpb, regularizationWeights);

		t.stop("create pyr-level weights");

#if 0
		if(outdirPlusFilebase)
		{
			cv::Mat_<uint8_t> hpbHorizImg(height, width), hpbVertImg(height, width);
			for(int32_t i = 0; i < height; i++)
				for(int32_t j = 0; j < width; j++)
				{
					hpbHorizImg(i, j) = (uint8_t)clamp(255.0 * hpb[i][j][0], 0.0, 255.0);
					hpbVertImg(i, j) = (uint8_t)clamp(255.0 * hpb[i][j][1], 0.0, 255.0);
				}
			cv::imwrite(outdirPlusFilebase.get().string() + (boost::format("-hpbHoriz%1%.png") % k).str(), hpbHorizImg);
			cv::imwrite(outdirPlusFilebase.get().string() + (boost::format("-hpbVert%1%.png") % k).str(), hpbVertImg);
		}
#endif

		t.restart();
		switch(params.variant)
		{
			case sceneFlowVariant::FULL:
			{
				/*
				 * 20130307: at least when using SOR, I've found that on the smallest pyramid level, convergence statistics are unreliable because first the optimization has to get away from zero toward a mode;
				 * on further pyramid levels it's merely refining, so the statistic decreases monotonically and we can do convergence testing
				 */
				const bool allowEarlyTermination = (k < GPyramidIm1.numLevels() - 1);

				sceneFlowSingleScale(camParams, Image1, Depth1, Image2, Depth2, WarpImage2, WarpDepth2, u, v, w,
					params.colorDataWeight, gPyramidDepthDataTermWeights.image(k),
					params.alpha, gPyramidDepthSmoothnessTermWeights.image(k),
					params.gamma, depthMagnitudeTermWeights,
					regularizationWeights,
					params.nOuterFPIterations, params.nInnerFPIterations, params.nSORIterations, allowEarlyTermination);
				break;
			}
			default: ASSERT_ALWAYS(false && "bad scene flow variant");
		}
		t.stop("run sceneFlowSingleScale");

#if 0
		if(outdirPlusFilebase)
		{
			cv::Mat_<cv::Vec3f> flow3d(u.height(), u.width());
			for(int q = 0; q < u.height(); q++)
				for(int r = 0; r < u.width(); r++)
					flow3d(q, r) = cv::Vec3f(u(q, r), v(q, r), w(q, r));
			writeSceneFlow(flow3d, outdirPlusFilebase.get().string() + (boost::format("-flow%1%.flo3") % k).str());
		}
#endif
	}
	t.stop("run flow pyramid loop");
}
