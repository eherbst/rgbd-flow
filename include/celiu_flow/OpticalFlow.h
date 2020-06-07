#pragma once

#include <vector>
#include <boost/multi_array.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/optional.hpp>
#include <opencv2/core/core.hpp>
#include "rgbd_util/CameraParams.h"
#include "celiu_flow/Image.h"
#include "celiu_flow/NoiseModel.h"
#include "celiu_flow/Vector.h"
namespace fs = boost::filesystem;

/*
 * current experiments in scene flow, removed from the class below to make use of CUDA within it easier
 */
void sceneFlowSingleScale(const rgbd::CameraParams& camParams, const DImage &Im1, const DImage& dm1, const DImage &Im2, const DImage& dm2, DImage &warpIm2, DImage& warpDm2, DImage &u, DImage &v, DImage& w,
	const double colorDataTermWeight, const DImage& depthDataTermWeights,
	const double alpha, const DImage& depthSmoothnessTermWeights,
	const double gamma, const DImage& depthMagTermWeights,
	const DImage& regularizationWeights,
	int nOuterFPIterations, int nInnerFPIterations, int nSORIterations, const bool allowEarlyTermination);

class OpticalFlow
{
public:

	static bool IsDisplay;
	enum InterpolationMethod {Bilinear,Bicubic};
	static InterpolationMethod interpolation;
	enum NoiseModel {GMixture,Lap};
	OpticalFlow(void);
	~OpticalFlow(void);

	//noise model info
	static GaussianMixture GMPara;
	static Vector<double> LapPara; //laplacian
	static NoiseModel noiseModel;

	/*****************************************************************************************************************
	 * optical flow algorithms
	 */

	enum class opticalFlowVariant
	{
		CELIU, //run Liu's code
		CELIU_NONROBUST, //Liu's code + quadratic penalizers on data & smoothness
		NAGEL_ENKELMANN, //use their matrix D for anisotropic regularization
		SKIP_INVALID_DEPTHS //skip pts with invalid depth in regularization (often improves things; see results on my wiki)
	};
	struct opticalFlowParams
	{
		//20120914 xf best params for 320x240 synthetic w/o noise: alpha = .05; #iters = 10,1,100
		opticalFlowParams() : variant(opticalFlowVariant::SKIP_INVALID_DEPTHS), alpha(-1), pyramidRatio(.75), minWidth(20/* Liu used 20 */), nOuterFPIterations(7), nInnerFPIterations(1), nSORIterations(30)
		{}

		opticalFlowVariant variant;

		double alpha; //used if not negative
		cv::Mat_<float> dzdx, dzdy; //for regularization in SORAnisotropic
		cv::Mat_<float> depthMap; //for regularization in SORValidPtsOnly; not rescaled
		cv::Mat_<float> depthMap1, depthMap2; //if given, we add depth as an extra channel in the constancy constraint (so these should be scaled to [0, 1] like the color channels)
		boost::multi_array<float, 2> depthConstancyWeights; //needed if depthMapN are given; at each pixel, a weight for the constancy constraint (basically a depth uncertainty)

		//pyramid
		double pyramidRatio;
		int minWidth;

		//fixed-point stuff
		int nOuterFPIterations, nInnerFPIterations, nSORIterations;
	};
	/*
	 * top-level function
	 */
	static void Coarse2FineFlow(DImage& vx,DImage& vy,DImage &warpI2,const DImage& Im1,const DImage& Im2, const opticalFlowParams& params);

	/*
	 * Liu's original code
	 */
	static void SmoothFlowSOR(const DImage& Im1,const DImage& Im2, DImage& warpIm2, DImage& vx, DImage& vy,
														 double alpha,int nOuterFPIterations,int nInnerFPIterations,int nSORIterations);

	/*
	 * added by EVH: nonrobust regularization on data and smoothness terms
	 */
	static void SmoothFlowSORNonrobust(const DImage &Im1, const DImage &Im2, DImage &warpIm2, DImage &u, DImage &v,
	   double alpha, int nOuterFPIterations, int nInnerFPIterations, int nSORIterations);

	/*****************************************************************************************************************
	 * scene flow algorithms
	 */

	enum class sceneFlowVariant
	{
		NONROBUST,
		LINEARIZED,
		FULL
	};
	enum class sceneFlowRegularizationType
	{
		ISOTROPIC,
		LETOUZEY, //depth-dependent gaussian
		HPB //"heuristic probability of boundary", a non-learned Pb-like measure
	};
	struct sceneFlowParams
	{
		sceneFlowParams()
		: variant(sceneFlowVariant::FULL), regularizationType(sceneFlowRegularizationType::HPB), colorDataWeight(1), depthVsColorDataWeight(1), alpha(0), gamma(0), pyramidRatio(.75), minWidth(20/* Liu used 20 */), nOuterFPIterations(7), nInnerFPIterations(1), nSORIterations(30)
		{}

		sceneFlowVariant variant;
		sceneFlowRegularizationType regularizationType;

		rgbd::CameraParams camParams;

		double colorDataWeight; //should be 1 when not debugging
		double depthVsColorDataWeight;
		double alpha; //weight smoothness (vs data term)
		double gamma; //weight zero-flow prior
		cv::Mat_<float> depthMap; //for regularization; not rescaled; float type
		cv::Mat_<float> depthSigma1, depthSigma2; //depth stdev at each pixel; float type

		//pyramid
		double pyramidRatio;
		int minWidth;

		//fixed-point stuff
		int nOuterFPIterations, nInnerFPIterations, nSORIterations;
	};
	/*
	 * top-level function
	 */
	static void sceneFlowMultiscale(const DImage& Im1, const DImage& dm1, const DImage& Im2, const DImage& dm2, const sceneFlowParams& params, DImage& u, DImage& v, DImage& w, const boost::optional<fs::path>& outdirPlusFilebase);

	static void sceneFlowSingleScaleNonrobust(const DImage &Im1, const DImage& dm1, const DImage &Im2, const DImage& dm2, DImage &warpIm2, DImage& warpDm2, DImage &u, DImage &v, DImage& w,
		const DImage& depthDataTermWeights,
		const double alpha, const DImage& depthSmoothnessTermWeights,
		const double gamma, const DImage& depthMagTermWeights,
		const DImage& regularizationWeights,
		int nOuterFPIterations, int nInnerFPIterations, int nSORIterations);

	/*
	 * compute 3-d flow at just the largest scale given an initialization
	 */
	static void sceneFlowWithInit(const DImage& Im1, const DImage& dm1, const DImage& Im2, const DImage& dm2, const sceneFlowParams& params, const DImage& estFlow, DImage& u, DImage& v, DImage& w, const boost::optional<fs::path>& outdirPlusFilebase);

	/*************************************************************************************************************
	 * per-pixel weights for the scene flow objective function
	 */

	/*
	 * depthDataTermWeights: at each pixel, a weight for the depth data constraint (basically a depth uncertainty)
	 */
	static void calculateDepthDataTermWeights(const sceneFlowParams& params, DImage& depthDataTermWeights);

	/*
	 * depthSmoothnessTermWeights: at each pixel, a weight for the depth smoothness term (basically a depth uncertainty)
	 *
	 * weight depth vs x-y smoothness
	 *
	 * weight maybe depends on how little color variation you have--if none but there's depth noise, this being high will cause large xy-flows
	 */
	static void calculateDepthSmoothnessTermWeights(const sceneFlowParams& params, DImage& depthSmoothnessTermWeights);

	/*
	 * depth-component-specific weights for flow magnitude penalty
	 */
	static void calculateDepthMagnitudeTermWeights(const DImage& depth, const sceneFlowParams& params, DImage& depthMagnitudeTermWeights);

	static void calculateRegularizationWeights(const DImage& img, const DImage& depth, const sceneFlowParams& params, const boost::multi_array<float, 3>& hpb, DImage& regularizationWeights);

	/*************************************************************************************************************
	 * measures of flow uncertainty
	 */

	/*
	 * include all the energy data terms
	 */
	static cv::Mat_<float> calcFlowTotalEnergyPerPixel(const DImage& img1, const DImage& depth1, const DImage& img2, const DImage& depth2, const OpticalFlow::sceneFlowParams& params, const cv::Mat_<cv::Vec3f>& sceneFlow);

protected:

	enum class constancyType
	{
		GRAY,
		GRAY_WITH_GRADIENT,
		RGB_WITH_GRADIENT,
		HUE_WITH_GRADIENT,
		HSL_WITH_GRADIENT
	};
	/*
	 * function to convert image to features (things we'll enforce constancy for, eg brightness or intensity gradient)
	 */
	static void im2feature(DImage& imfeature,const DImage& im, const constancyType c);
	static void im2featureWithDepth(DImage& imfeature, const DImage& im, const DImage& depth);
};
