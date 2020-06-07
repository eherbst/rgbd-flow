/*
 * ceLiuOpticalFlow: wrapper around Liu's code to provide opencv data structures
 *
 * Evan Herbst
 * 7 / 24 / 12
 */

#include <fstream>
#include "rgbd_util/timer.h"
#include "celiu_flow/Image.h"
#include "celiu_flow/ImageIO.h"
#include "celiu_flow/recursiveMedianFiltering.h"
#include "celiu_flow/ceLiuOpticalFlow.h"
using std::ofstream;

/*
 * bgr -> rgb and put result into [0, 1)
 */
DImage cv2liuBGR(const cv::Mat& img)
{
	ASSERT_ALWAYS(img.type() == cv::DataType<cv::Vec3b>::type);
	DImage im(img.cols, img.rows, 3);
	double* d = im.data();
	for(int i = 0; i < img.rows; i++)
		for(int j = 0; j < img.cols; j++)
		{
			const cv::Vec3b pix = img.at<cv::Vec3b>(i, j);
			for(int k = 0; k < 3; k++)
			{
				*d++ = pix[/*2 - */k] / 255.0;
			}
		}
	return im;
}

/*
 * fillIn: whether to run recursive median filtering
 */
DImage cv2liuDepth(const cv::Mat_<float>& dm, const bool fillIn)
{
	DImage dim(dm.cols, dm.rows, 1);
	double* d = dim.data();
	for(int i = 0; i < dm.rows; i++)
		for(int j = 0; j < dm.cols; j++)
			*d++ = dm(i, j);
	if(fillIn) recursiveMedianFilter(dim);
	return dim;
}

cv::Mat_<cv::Vec2f> liu2cvFlow2d(const DImage& vx, const DImage& vy)
{
	cv::Mat_<cv::Vec2f> flowMat(vx.height(), vx.width());
	const double* dx = vx.data(), *dy = vy.data();
	for(int i = 0; i < vx.height(); i++)
		for(int j = 0; j < vx.width(); j++)
		{
			cv::Vec2f& pix = flowMat(i, j);
			pix[0] = *dx++;
			pix[1] = *dy++;
		}
	return flowMat;
}

DImage cv2liuFlow3d(const cv::Mat_<cv::Vec3f>& flow)
{
	DImage f(flow.cols, flow.rows, 3);
	double* d = f.data();
	for(int i = 0, l = 0; i < flow.rows; i++)
		for(int j = 0; j < flow.cols; j++)
		{
			const cv::Vec3f u = flow(i, j);
			for(int k = 0; k < 3; k++)
				*d++ = u[k];
		}
	return f;
}

cv::Mat_<cv::Vec3f> liu2cvFlow3d(const DImage& vx, const DImage& vy, const DImage& vz)
{
	cv::Mat_<cv::Vec3f> flowMat(vx.height(), vx.width());
	const double* dx = vx.data(), *dy = vy.data(), *dz = vz.data();
	for(int i = 0; i < vx.height(); i++)
		for(int j = 0; j < vx.width(); j++)
		{
			cv::Vec3f& pix = flowMat(i, j);
			pix[0] = *dx++;
			pix[1] = *dy++;
			pix[2] = *dz++;
		}
	return flowMat;
}

/**********************************************************************************************************************************/

cv::Mat_<cv::Vec2f> ceLiuOpticalFlow(const DImage& Im1, const DImage& Im2, const OpticalFlow::opticalFlowParams& params)
{
	DImage vx,vy,warpI2;
	rgbd::timer t;
	OpticalFlow::Coarse2FineFlow(vx,vy,warpI2,Im1,Im2,params);
	t.stop("run coarse2fineFlow");

	return liu2cvFlow2d(vx, vy);
}

cv::Mat_<cv::Vec2f> ceLiuOpticalFlow(const cv::Mat& img1, const cv::Mat& img2, const OpticalFlow::opticalFlowParams& params)
{
	ASSERT_ALWAYS(img1.type() == cv::DataType<cv::Vec3b>::type);
	ASSERT_ALWAYS(img2.type() == cv::DataType<cv::Vec3b>::type);
	/*
	 * put imgs into the range [0, 1]
	 */
	DImage Im1 = cv2liuBGR(img1), Im2 = cv2liuBGR(img2);

	return ceLiuOpticalFlow(Im1, Im2, params);
}

/*
 * run on depth imgs (we'll do the normalization)
 */
cv::Mat_<cv::Vec2f> ceLiuOpticalFlowDepth(const cv::Mat_<float>& img1, const cv::Mat_<float>& img2, const OpticalFlow::opticalFlowParams& params)
{
	/*
	 * put imgs into the range [0, 1]
	 */
	float minVal = FLT_MAX, maxVal = -FLT_MAX;
	for(int i = 0; i < img1.rows; i++)
		for(int j = 0; j < img1.cols; j++)
		{
			const float pix1 = img1(i, j), pix2 = img2(i, j);
			if(pix1 < minVal) minVal = pix1;
			if(pix2 < minVal) minVal = pix2;
			if(pix1 > maxVal) maxVal = pix1;
			if(pix2 > maxVal) maxVal = pix2;
		}
	DImage Im1(img1.cols, img1.rows, 1), Im2(img1.cols, img1.rows, 1);
	double* d1 = Im1.data(), *d2 = Im2.data();
	for(int i = 0; i < img1.rows; i++)
		for(int j = 0; j < img1.cols; j++)
		{
			const float pix1 = img1(i, j), pix2 = img2(i, j);
			*d1++ = (pix1 - minVal) / (maxVal - minVal);
			*d2++ = (pix2 - minVal) / (maxVal - minVal);
		}

	return ceLiuOpticalFlow(Im1, Im2, params);
}

/*
 * compute scene flow by computing optical flow, then following it and sampling the second depth map
 */
cv::Mat_<cv::Vec3f> ceLiuSceneFlowFrom2DFlow(const cv::Mat& img1, const cv::Mat_<float>& depth1, const cv::Mat& img2, const cv::Mat_<float>& depth2, const OpticalFlow::opticalFlowParams& params)
{
	/*
	 * put imgs into the range [0, 1]
	 */
	DImage Im1 = cv2liuBGR(img1), Im2 = cv2liuBGR(img2);
	DImage dm1 = cv2liuDepth(depth1, true/* fill in */), dm2 = cv2liuDepth(depth2, true/* fill in */);

	const int rows = Im1.height(), cols = Im1.width();

	DImage vx,vy,warpI2;
	rgbd::timer t;
	OpticalFlow::Coarse2FineFlow(vx,vy,warpI2,Im1,Im2,params);
	t.stop("run coarse2fineFlow");
	ASSERT_ALWAYS(vx.nchannels() == 1);
	ASSERT_ALWAYS(vy.nchannels() == 1);

	cv::Mat_<cv::Vec3f> flowMat(rows, cols);
	double* dx = vx.data(), *dy = vy.data();
	for(int i = 0; i < rows; i++)
		for(int j = 0; j < cols; j++)
		{
			cv::Vec3f& pix = flowMat(i, j);
			pix[0] = *dx++;
			pix[1] = *dy++;

			/*
			 * follow the optical flow to sample the second depth map to get 3-d flow
			 */
			const float x2 = j + pix[0], y2 = i + pix[1];
			const float z0 = ImageProcessing::BilinearInterpolate(dm1.data(), img1.cols, img1.rows, j, i), z1 = ImageProcessing::BilinearInterpolate(dm2.data(), img1.cols, img1.rows, x2, y2);
			pix[2] = z1 - z0;
		}
	return flowMat;
}

/*
 * estFlow: initial estimate; if given, used to initialize
 */
cv::Mat_<cv::Vec3f> ceLiuSceneFlow(const cv::Mat& img1, const cv::Mat_<float>& depth1, const cv::Mat& img2, const cv::Mat_<float>& depth2, const boost::optional<cv::Mat_<cv::Vec3f>>& estFlow, const OpticalFlow::sceneFlowParams& params,
	const boost::optional<fs::path>& outdirPlusFilebase)
{
	/*
	 * put imgs into the range [0, 1]
	 */
	DImage Im1 = cv2liuBGR(img1), Im2 = cv2liuBGR(img2);
	DImage dm1 = cv2liuDepth(depth1, false/* fill in */), dm2 = cv2liuDepth(depth2, false/* fill in */);

	DImage vx,vy,vz;
	if(estFlow)
	{
		DImage estFlowD = cv2liuFlow3d(estFlow.get());
		rgbd::timer t;
		OpticalFlow::sceneFlowWithInit(Im1, dm1, Im2, dm2, params, estFlowD, vx, vy, vz, outdirPlusFilebase);
		t.stop("run sceneFlowWithInit");
	}
	else
	{
		rgbd::timer t;
		OpticalFlow::sceneFlowMultiscale(Im1, dm1, Im2, dm2, params, vx, vy, vz, outdirPlusFilebase);
		t.stop("run sceneFlowMultiscale");
	}

	return liu2cvFlow3d(vx, vy, vz);
}
