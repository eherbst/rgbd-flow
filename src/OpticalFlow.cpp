#include <cassert>
#include <vector>
#include <iostream>
#include <fstream>
#include <utility>
#include <boost/multi_array.hpp>
#include <boost/format.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/lexical_cast.hpp>
#include <Eigen/Core>
#include "celiu_flow/recursiveMedianFiltering.h"
#include "celiu_flow/ImageProcessing.h"
#include "celiu_flow/GaussianPyramid.h"
#include "celiu_flow/opticalFlowUtils.h"
#include "celiu_flow/OpticalFlow.h"
using namespace std;
namespace fs = boost::filesystem;
using namespace scene_flow;

static double sqr(double x) {return x * x;}

bool OpticalFlow::IsDisplay=true;
OpticalFlow::InterpolationMethod OpticalFlow::interpolation = OpticalFlow::Bilinear;
OpticalFlow::NoiseModel OpticalFlow::noiseModel = OpticalFlow::Lap;
GaussianMixture OpticalFlow::GMPara;
Vector<double> OpticalFlow::LapPara;


OpticalFlow::OpticalFlow(void)
{
}

OpticalFlow::~OpticalFlow(void)
{
}

//--------------------------------------------------------------------------------------------------------
// SmoothFlowSOR: function to compute optical flow field using two fixed point iterations
// Input arguments:
//     Im1, Im2:						frame 1 and frame 2
//	warpIm2:						the warped frame 2 according to the current flow field u and v
//	u,v:									the current flow field, NOTICE that they are also output arguments
//	
//--------------------------------------------------------------------------------------------------------

//various versions of the per-pyramid-level main flow function
#include "SmoothFlowSOR.cpp" //the original
#include "SmoothFlowSORNonrobust.cpp" //original with quadratic penalties

/*
 * compute 2-d flow over the pyramid
 */
#include "opticalFlowMultiscale.cpp"

/*
 * compute 3-d flow over the pyramid
 */
#include "sceneFlowMultiscale.cpp"
/*
 * compute 3-d flow at just the largest scale given an initialization
 */
#include "sceneFlowWithInit.cpp"

/*************************************************************************************************************
 * per-pixel weights for the scene flow objective function
 */

/*
 * depthDataTermWeights: at each pixel, a weight for the depth data constraint (basically a depth uncertainty)
 */
void OpticalFlow::calculateDepthDataTermWeights(const sceneFlowParams& params, DImage& depthDataTermWeights)
{
	depthDataTermWeights.allocate(params.camParams.xRes, params.camParams.yRes, 1);
	double* d = depthDataTermWeights.data();
	for(uint32_t i = 0; i < params.camParams.yRes; i++)
		for(uint32_t j = 0; j < params.camParams.xRes; j++)
		{
			ASSERT_ALWAYS(params.depthSigma1(i, j) > 0);
			*d++ = primesensor::stereoError(1) / params.depthSigma1(i, j) * params.depthVsColorDataWeight;
		}
}

/*
 * depthSmoothnessTermWeights: at each pixel, a weight for the depth smoothness term (basically a depth uncertainty)
 *
 * weight depth vs x-y smoothness
 *
 * weight maybe depends on how little color variation you have--if none but there's depth noise, this being high will cause large xy-flows
 */
void OpticalFlow::calculateDepthSmoothnessTermWeights(const sceneFlowParams& params, DImage& depthSmoothnessTermWeights)
{
	depthSmoothnessTermWeights.allocate(params.camParams.xRes, params.camParams.yRes, 1);
	double* d = depthSmoothnessTermWeights.data();
	for(uint32_t i = 0; i < params.camParams.yRes; i++)
		for(uint32_t j = 0; j < params.camParams.xRes; j++)
			*d++ = sqr(params.camParams.focalLength);
}

/*
 * depth-component-specific weights for flow magnitude penalty
 *
 * TODO should change with pyramid scale since pixels/meter changes?
 */
void OpticalFlow::calculateDepthMagnitudeTermWeights(const DImage& depth, const sceneFlowParams& params, DImage& depthMagnitudeTermWeights)
{
	depthMagnitudeTermWeights.allocate(params.camParams.xRes, params.camParams.yRes); //to make xy- and z-flow weighted equally in depth-magnitude penalty term
	for(uint32_t i = 0, l = 0; i < params.camParams.yRes; i++)
		for(uint32_t j = 0; j < params.camParams.xRes; j++, l++)
			if(depth(i, j) > 0) depthMagnitudeTermWeights(i, j) = sqr(params.camParams.focalLength / depth(i, j));
			else depthMagnitudeTermWeights(i, j) = sqr(params.camParams.focalLength / 2/* TODO ? */);
}

void OpticalFlow::calculateRegularizationWeights(const DImage& img, const DImage& depth, const sceneFlowParams& params, const boost::multi_array<float, 3>& hpb, DImage& regularizationWeights)
{
	regularizationWeights.allocate(img.width(), img.height(), 2); //channels are for right and down nbr pixels
	std::function<double (int32_t i, int32_t j, int32_t l, int32_t i2, int32_t j2, int32_t l2, const float hpbVal)> WEIGHT;
	switch(params.regularizationType)
	{
		case OpticalFlow::sceneFlowRegularizationType::ISOTROPIC:
			WEIGHT = [](int32_t i, int32_t j, int32_t l, int32_t i2, int32_t j2, int32_t l2, const float hpbVal){return 1;};
			break;
		case OpticalFlow::sceneFlowRegularizationType::LETOUZEY:
			WEIGHT = [&depth](int32_t i, int32_t j, int32_t l, int32_t i2, int32_t j2, int32_t l2, const float hpbVal){return exp(-sqr((depth.data()[l] - depth.data()[l2]) / .02/* TODO ? */));};
			break;
		case OpticalFlow::sceneFlowRegularizationType::HPB:
			WEIGHT = [](int32_t i, int32_t j, int32_t l, int32_t i2, int32_t j2, int32_t l2, const float hpbVal)
								{
									return (1 - hpbVal * .99/* TODO ? */);
								};
			break;
		default: ASSERT_ALWAYS(false);
	}
	for(int32_t i = 0, l = 0; i < img.height(); i++)
		for(int32_t j = 0; j < img.width(); j++, l++)
		{
			if(j < img.width() - 1) regularizationWeights(i, j, 0) = WEIGHT(i, j, l, i, (j + 1), (l + 1), *(hpb.data() + (i * img.width() + j) * 2 + 0));
			//else won't be used
			if(i < img.height() - 1) regularizationWeights(i, j, 1) = WEIGHT(i, j, l, (i + 1), j, (l + img.width()), *(hpb.data() + (i * img.width() + j) * 2 + 1));
			//else won't be used
		}
}

/***************************************************************************************************************************************************************************************/

/*
 * function to convert image to features (things we'll enforce constancy for, eg brightness or intensity gradient)
 */
void OpticalFlow::im2feature(DImage &imfeature, const DImage &im, const constancyType c)
{
	int width=im.width();
	int height=im.height();
	int nchannels=im.nchannels();
	if(nchannels==1)
	{
		switch(c)
		{
			case constancyType::GRAY:
			{
				imfeature.allocate(im.width(),im.height(),1);
				double* data=imfeature.data();
				for(int i=0;i<height;i++)
					for(int j=0;j<width;j++)
					{
						int offset=i*width+j;
						data[offset]=im.data()[offset];
					}
				break;
			}
			case constancyType::GRAY_WITH_GRADIENT:
			{
				imfeature.allocate(im.width(),im.height(),3);
				DImage imdx(im.width(),im.height()),imdy(im.width(),im.height());
				im.dx(imdx,true);
				im.dy(imdy,true);
				double* data=imfeature.data();
				for(int i=0;i<height;i++)
					for(int j=0;j<width;j++)
					{
						int offset=i*width+j;
						data[offset*3]=im.data()[offset];
						data[offset*3+1]=imdx.data()[offset];
						data[offset*3+2]=imdy.data()[offset];
					}
				break;
			}
			default: ASSERT_ALWAYS(false);
		}
	}
	else if(nchannels==3)
	{
		DImage grayImage;
		im.desaturate(grayImage);

		switch(c)
		{
			case constancyType::RGB_WITH_GRADIENT:
			{
				imfeature.allocate(im.width(),im.height(),5);
				DImage imdx(im.width(),im.height()),imdy(im.width(),im.height());
				grayImage.dx(imdx,true);
				grayImage.dy(imdy,true);
				double* data=imfeature.data();
				for(int i=0;i<height;i++)
					for(int j=0;j<width;j++)
					{
						int offset=i*width+j;
						data[offset*5]=grayImage.data()[offset];
						data[offset*5+1]=imdx.data()[offset];
						data[offset*5+2]=imdy.data()[offset];
						data[offset*5+3]=im.data()[offset*3+1]-im.data()[offset*3];
						data[offset*5+4]=im.data()[offset*3+1]-im.data()[offset*3+2];
					}
				break;
			}
			default: ASSERT_ALWAYS(false);
		}
	}
	else
		imfeature.copyData(im);
}

void OpticalFlow::im2featureWithDepth(DImage& imfeature, const DImage& im, const DImage& depth)
{
	int width=im.width();
	int height=im.height();
	int nchannels=im.nchannels();
	if(nchannels==3)
	{
		DImage grayImage;
		im.desaturate(grayImage);

		const uint32_t NF = 6;
		imfeature.allocate(im.width(),im.height(),NF);
		DImage imdx(im.width(),im.height()),imdy(im.width(),im.height());
		grayImage.dx(imdx,true);
		grayImage.dy(imdy,true);
		double* data=imfeature.data();
		for(int i=0;i<height;i++)
			for(int j=0;j<width;j++)
			{
				int offset=i*width+j;
				data[offset*NF]=grayImage.data()[offset];
				data[offset*NF+1]=imdx.data()[offset];
				data[offset*NF+2]=imdy.data()[offset];
				data[offset*NF+3]=im.data()[offset*3+1]-im.data()[offset*3];
				data[offset*NF+4]=im.data()[offset*3+1]-im.data()[offset*3+2];
				data[offset*NF+5] = depth.data()[offset];
			}
	}
	else ASSERT_ALWAYS(false);
}
