/*
 * rgbdFlowTest: test driver for flow code release
 *
 * Evan Herbst
 * 12 / 16 / 13
 */

#include <iostream>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "rgbd_util/assert.h"
#include "rgbd_util/timer.h"
#include "rgbd_util/mathUtils.h"
#include "rgbd_util/primesensorUtils.h"
#include "optical_flow_utils/middleburyFlowIO.h"
#include "optical_flow_utils/sceneFlowIO.h"
#include "rgbd_flow/rgbdFrameUtils.h"
#include "rgbd_flow/runFlowTwoFrames.h"
using std::vector;
using std::string;
using std::cout;
using std::endl;
using std::ofstream;
namespace fs = boost::filesystem;
namespace po = boost::program_options;

/*
 * read 8- or 16-bit png
 */
void readDepthMapValuesImg(const cv::Mat& img, cv::Mat_<float>& depth)
{
	ASSERT_ALWAYS(img.type() == CV_8UC1 || img.type() == CV_16UC1);
	const bool read16bit = (img.type() == CV_16UC1);
	depth.create(img.rows, img.cols);
	for(uint32_t i = 0, k = 0; i < img.rows; i++)
		for(uint32_t j = 0; j < img.cols; j++, k++)
		{
			if(read16bit) depth(i, j) = img.at<uint16_t>(i, j) * .001; //mm -> m
			else depth(i, j) = img.at<uint8_t>(i, j) * .001; //mm -> m
		}
}
void readDepthMapValuesImg(const fs::path& filepath, cv::Mat_<float>& depth)
{
	const cv::Mat depthImg = cv::imread(filepath.string(), -1);
	readDepthMapValuesImg(depthImg, depth);
}

int main(int argc, char* argv[])
{
	po::options_description desc("Options");
	desc.add_options()
		//or these
		("image1,i", po::value<fs::path>(), "color image 1")
		("depth1,d", po::value<fs::path>(), "depth image 1 (16-bit png, depth in mm)")
		("image2,j", po::value<fs::path>(), "color image 2")
		("depth2,e", po::value<fs::path>(), "depth image 2 (16-bit png, depth in mm)")

		("flow-type,t", po::value<uint32_t>(), "1 = modified Liu optical flow; 2 = Liu optical flow on the depth channel; 6 = RGB-D Flow")
		("smoothness-weight,s", po::value<double>()->default_value(.02), "")
		("depth-vs-color-data-weight,v", po::value<double>()->default_value(1), "")
		("regularization-type,z", po::value<uint32_t>()->default_value(2), "0 = isotropic; 2 = anisotropic from icra13 paper")
		("run-fwd", po::value<bool>()->default_value(true), "whether to run forward flow")
		("run-bkwd", po::value<bool>()->default_value(false), "whether to run backward flow")
		("outdir,o", po::value<fs::path>(), "")
		;
	po::variables_map vars;
	po::store(po::command_line_parser(argc, argv).options(desc).run(), vars);
	po::notify(vars);

	const uint32_t flowType = vars["flow-type"].as<uint32_t>();
	const double smoothnessWeight = vars["smoothness-weight"].as<double>();
	const double depthVsColorDataWeight = vars["depth-vs-color-data-weight"].as<double>();
	const OpticalFlow::sceneFlowRegularizationType regularizationType = OpticalFlow::sceneFlowRegularizationType(vars["regularization-type"].as<uint32_t>());
	const bool runFwdFlow = vars["run-fwd"].as<bool>(), runBkwdFlow = vars["run-bkwd"].as<bool>();
	const fs::path outdir = vars["outdir"].as<fs::path>();
	fs::create_directories(outdir);

	const rgbd::CameraParams camParams = primesensor::getColorCamParams(rgbd::KINECT_640_DEFAULT);

	const bool isSceneFlow = (flowType == 6 || flowType == 7);
	std::function<cv::Mat_<cv::Vec3f> (const cv::Mat& prevImg, const cv::Mat_<float>& prevDepth, const cv::Mat& curImg, const cv::Mat_<float>& curDepth,
					const double smoothnessWeight, const double depthVsColorDataWeight, const OpticalFlow::sceneFlowRegularizationType regularizationType, const boost::optional<fs::path>& outdirPlusFilebase)> runFlowFunc;
	switch(flowType)
	{
		case 1:
			runFlowFunc = [](const cv::Mat& prevImg, const cv::Mat_<float>& prevDepth, const cv::Mat& curImg, const cv::Mat_<float>& curDepth,
				const double smoothnessWeight, const double depthVsColorDataWeight, const OpticalFlow::sceneFlowRegularizationType regularizationType, const boost::optional<fs::path>& outdirPlusFilebase)
				{return runFlowTwoFramesCeLiu(prevImg, prevDepth, curImg, curDepth, smoothnessWeight, outdirPlusFilebase);};
			break;
		case 2:
			runFlowFunc = [](const cv::Mat& prevImg, const cv::Mat_<float>& prevDepth, const cv::Mat& curImg, const cv::Mat_<float>& curDepth,
				const double smoothnessWeight, const double depthVsColorDataWeight, const OpticalFlow::sceneFlowRegularizationType regularizationType, const boost::optional<fs::path>& outdirPlusFilebase)
				{return runFlowTwoFramesCeLiuOnDepth(prevImg, prevDepth, curImg, curDepth, smoothnessWeight, outdirPlusFilebase);};
			break;
		case 6:
			runFlowFunc = [&camParams](const cv::Mat& prevImg, const cv::Mat_<float>& prevDepth, const cv::Mat& curImg, const cv::Mat_<float>& curDepth,
				const double smoothnessWeight, const double depthVsColorDataWeight, const OpticalFlow::sceneFlowRegularizationType regularizationType, const boost::optional<fs::path>& outdirPlusFilebase)
				{return runFlowTwoFramesSceneFlowExperiments(camParams, prevImg, prevDepth, curImg, curDepth, smoothnessWeight, depthVsColorDataWeight, regularizationType, 10000/* iters */, boost::none, outdirPlusFilebase);};
			break;
		default: ASSERT_ALWAYS(false);
	}

	const fs::path img1path = vars["image1"].as<fs::path>(),
		depth1path = vars["depth1"].as<fs::path>(),
		img2path = vars["image2"].as<fs::path>(),
		depth2path = vars["depth2"].as<fs::path>();

	cv::Mat prevImg = cv::imread(img1path.string()), curImg = cv::imread(img2path.string());
	cv::Mat_<float> prevDepth, curDepth;
	readDepthMapValuesImg(depth1path, prevDepth);
	readDepthMapValuesImg(depth2path, curDepth);

#if 1 //for speed
	downsizeFrame(prevImg, prevDepth);
	downsizeFrame(curImg, curDepth);
#endif

	if(runFwdFlow)
	{
		rgbd::timer t;
		cv::Mat flow = runFlowFunc(prevImg, prevDepth, curImg, curDepth, smoothnessWeight, depthVsColorDataWeight, regularizationType, outdir / "0");
		t.stop("call runFlowTwoFrames");
		if(!isSceneFlow) writeFlow(flow, outdir / "flow0.flo");
		else writeSceneFlow(flow, outdir / "flow0.flo3");

		/*
		 * debug imgs
		 */
		cv::imwrite((outdir / "fwd-img1.png").string(), prevImg);
		cv::Mat_<cv::Vec3b> img2samples(prevImg.size());
		for(int32_t i = 0; i < prevImg.rows; i++)
			for(int32_t j = 0; j < prevImg.cols; j++)
			{
				const cv::Vec3f f = flow.at<cv::Vec3f>(i, j);
				cv::Mat_<cv::Vec3b> pix(1, 1);
				cv::getRectSubPix(curImg, cv::Size(1, 1), cv::Point2f(j + f[0], i + f[1]), pix);
				img2samples.at<cv::Vec3b>(i, j) = pix(0, 0);
			}
		cv::imwrite((outdir / "fwd-img2sampled.png").string(), img2samples);
	}
	if(runBkwdFlow)
	{
		rgbd::timer t;
		cv::Mat flow = runFlowFunc(curImg, curDepth, prevImg, prevDepth, smoothnessWeight, depthVsColorDataWeight, regularizationType, outdir / "0B");
		t.stop("call runFlowTwoFrames");
		if(!isSceneFlow) writeFlow(flow, outdir / "flow0bkwd.flo");
		else writeSceneFlow(flow, outdir / "flow0bkwd.flo3");
	}

	return 0;
}
