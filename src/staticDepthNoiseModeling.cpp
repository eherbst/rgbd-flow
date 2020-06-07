/*
 * staticDepthNoiseModeling: modeling depth uncertainty from a single frame
 *
 * Evan Herbst
 * 10 / 31 / 12
 */

#include <cmath>
#include <vector>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "rgbd_util/assert.h"
#include "rgbd_util/timer.h"
#include "rgbd_util/parallelism.h"
#include "rgbd_util/threadPool.h"
#include "rgbd_util/primesensorUtils.h"
#include "rgbd_frame_common/staticDepthNoiseModeling.h"
using std::vector;

static double sqr(double x) {return x * x;}

/*
 * auxiliary
 *
 * increaseNearMaxDist: we'll increase stdevs  of pts w/in this many pixels of invalid pts (0 -> don't)
 */
void increaseStdevsNearInvalid(const cv::Mat_<float>& depth, const uint32_t increaseNearMaxDist, const float invalidSigma, boost::multi_array<float, 2>& stdevs, const bool multithread)
{
	cv::Mat_<uint8_t> invalidMask(depth.rows, depth.cols); //mark pixels with invalid depth with 0
	const uint32_t numThreads = getSuggestedThreadCount();
	const vector<unsigned int> indices = partitionEvenly(depth.rows, numThreads);
	const auto threadFunc = [&](const size_t startrow, const size_t endrow)
		{
			for(int32_t i = startrow, l = startrow * depth.cols; i < endrow; i++)
				for(int32_t j = 0; j < depth.cols; j++, l++)
				{
					if(depth(i, j) <= 0) invalidMask(i, j) = 0;
					else invalidMask(i, j) = 255;
				}
		};
	if(multithread)
	{
		rgbd::threadGroup tg(numThreads);
		for(size_t m = 0; m < numThreads; m++)
			tg.addTask([&,m](){threadFunc(indices[m], indices[m + 1]);});
		tg.wait();
	}
	else
	{
		threadFunc(0, depth.rows);
	}

	cv::Mat distsToInvalid;
	cv::distanceTransform(invalidMask, distsToInvalid, CV_DIST_L1, CV_DIST_MASK_PRECISE);

	const auto threadFunc2 = [&](const size_t startrow, const size_t endrow)
		{
			for(int32_t i = startrow, l = startrow * depth.cols; i < endrow; i++)
				for(int32_t j = 0; j < depth.cols; j++, l++)
					//decreasing series of large sigmas for points anywhere near invalid pts (TODO better way?)
					if(distsToInvalid.at<float>(i, j) <= increaseNearMaxDist)
					{
						const float alpha = distsToInvalid.at<float>(i, j) / increaseNearMaxDist;
						stdevs[i][j] = exp((1 - alpha) * log(10/* TODO ? */ * stdevs[i][j]) + alpha * log(stdevs[i][j]));
					}
		};
	if(multithread)
	{
		rgbd::threadGroup tg(numThreads);
		for(size_t m = 0; m < numThreads; m++)
			tg.addTask([&,m](){threadFunc2(indices[m], indices[m + 1]);});
		tg.wait();
	}
	else
	{
		threadFunc2(0, depth.rows);
	}
}

/*
 * auxiliary
 *
 * increaseNearDepthBoundariesMaxDist: we'll increase stdevs of pts w/in this many pixels of depth boundaries (0 -> don't)
 */
void increaseStdevsNearDepthBoundaries(const cv::Mat_<float>& depth, const uint32_t increaseNearDepthBoundariesMaxDist, const float invalidSigma, boost::multi_array<float, 2>& stdevs, const bool multithread)
{
	cv::Mat_<uint8_t> boundaryMask(depth.rows, depth.cols); //mark pixels at depth boundaries with 0
	const uint32_t numThreads = getSuggestedThreadCount();
	const vector<unsigned int> indices = partitionEvenly(depth.rows, numThreads);
	const auto threadFunc = [&](const size_t startrow, const size_t endrow)
		{
			for(int32_t i = startrow, l = startrow * depth.cols; i < endrow; i++)
				for(int32_t j = 0; j < depth.cols; j++, l++)
				{
					if(i < depth.rows - 1 && depth(i + 1, j) > 0 && fabs(depth(i, j) - depth(i + 1, j)) > .02/* TODO ? */ * primesensor::stereoErrorRatio(std::min(depth(i, j), depth(i + 1, j)))) boundaryMask(i, j) = 0;
					else if(j < depth.cols - 1 && depth(i, j + 1) > 0 && fabs(depth(i, j) - depth(i, j + 1)) > .02/* TODO ? */ * primesensor::stereoErrorRatio(std::min(depth(i, j), depth(i, j + 1)))) boundaryMask(i, j) = 0;
					else boundaryMask(i, j) = 255;
				}
		};
	if(multithread)
	{
		rgbd::threadGroup tg(numThreads);
		for(size_t m = 0; m < numThreads; m++)
			tg.addTask([&,m](){threadFunc(indices[m], indices[m + 1]);});
		tg.wait();
	}
	else
	{
		threadFunc(0, depth.rows);
	}

	cv::Mat distsToBoundary;
	cv::distanceTransform(boundaryMask, distsToBoundary, CV_DIST_L1, CV_DIST_MASK_PRECISE);

	const auto threadFunc2 = [&](const size_t startrow, const size_t endrow)
		{
			for(int32_t i = startrow, l = startrow * depth.cols; i < endrow; i++)
				for(int32_t j = 0; j < depth.cols; j++, l++)
					//decreasing series of large and depth-dependent sigmas for points near depth boundaries (TODO better way?)
					if(distsToBoundary.at<float>(i, j) <= increaseNearDepthBoundariesMaxDist)
					{
						const float alpha = distsToBoundary.at<float>(i, j) / increaseNearDepthBoundariesMaxDist;
						stdevs[i][j] = exp((1 - alpha) * log(10/* TODO ? */ * stdevs[i][j]) + alpha * log(stdevs[i][j]));
					}
		};
	if(multithread)
	{
		rgbd::threadGroup tg(numThreads);
		for(size_t m = 0; m < numThreads; m++)
			tg.addTask([&,m](){threadFunc2(indices[m], indices[m + 1]);});
		tg.wait();
	}
	else
	{
		threadFunc2(0, depth.rows);
	}
}

/*
 * compute a local std dev of depth for each pixel with valid depth, using a window of local depths
 * (useful, eg, for putting into a diffingSingleFrameInfo)
 *
 * careful when using this function: the local window can cross depth discontinuities and cause larger stdev values than you'd like
 *
 * stdevs will be resized if empty
 *
 * stdevs: indexed by (y, x)
 *
 * increaseNearDepthBoundariesMaxDist: we'll increase stdevs of pts w/in this many pixels of depth boundaries (0 -> don't)
 */
void computeDepthMapLocalStdevs(const cv::Mat_<float>& depth, const uint32_t nbrhoodHalfwidth, boost::multi_array<float, 2>& stdevs, const uint32_t increaseNearDepthBoundariesMaxDist, const rgbd::CameraParams& camParams, const bool multithread)
{
	if(stdevs.num_elements() == 0) stdevs.resize(boost::extents[depth.rows][depth.cols]);

	const float stdevInvalid = 1/* TODO ? */; //stdev if all pts are invalid (if you get this value for too many pts, try using a bigger nbrhood)
	const int32_t N = (int32_t)nbrhoodHalfwidth;
	const auto threadFunc = [N,stdevInvalid,&depth,&stdevs,&camParams](const int32_t v0, const int32_t v1)
		{
			auto s = stdevs.data() + v0 * depth.cols;
			for(int32_t v = v0; v <= v1; v++)
				for(int32_t u = 0; u < depth.cols; u++, s++)
					if(depth(v, u) <= 0) //depth invalid
					{
						*s = stdevInvalid;
					}
					else
					{
						double sum = 0, sqrSum = 0;
						uint32_t countValid = 0, countInvalid = 0;
						for(int32_t vv = std::max((int32_t)0, v - N); vv <= std::min((int32_t)depth.rows - 1, v + N); vv++)
							for(int32_t uu = std::max((int32_t)0, u - N); uu <= std::min((int32_t)depth.cols - 1, u + N); uu++)
							{
								const float z = depth(vv, uu);
								if(z <= 0) //if depth invalid
									countInvalid++;
								else
								{
									float maxDZ = depth(v, u) / camParams.focalLength / .5774/* tan(pi/2 - max surface angle expected)*/ * (fabs(vv - v) + fabs(uu - u))/*sqrt(sqr((float)vv - v) + sqr((float)uu - u))*/;
									maxDZ = .001 * ceil(maxDZ / .001) + .0001; //account for sensor discretization; TODO this is kinect-specific
									if(fabs(z - depth(v, u)) < maxDZ) //if likely same surface as nbrhood center
									{
										sum += z;
										sqrSum += z * z;
										countValid++;
									}
									else countInvalid++;
								}
							}

						//TODO do something if countValid is really small
						const float pctInvalid = (float)countInvalid / ((float)countInvalid + (float)countValid);
						const float stdevValid = sqrt(sqrSum / countValid - sqr(sum / countValid)); //the stdev of just the valid pts
						const float alpha = sqr(pctInvalid); //interpolation parameter; TODO ?
						*s = std::max(3e-4/* TODO ?; just to avoid zeros */, exp(alpha * log(stdevInvalid) + (1 - alpha) * log(stdevValid)));
					}
		};

	if(multithread)
	{
		const uint32_t numThreads = getSuggestedThreadCount();
		const vector<unsigned int> indices = partitionEvenly(depth.rows, numThreads);
		rgbd::threadGroup tg(numThreads);
		for(size_t i = 0; i < numThreads; i++)
			tg.addTask([i,&indices,&threadFunc](){threadFunc(indices[i], indices[i + 1] - 1);});
		tg.wait();
	}
	else
	{
		threadFunc(0, depth.rows - 1);
	}

	if(increaseNearDepthBoundariesMaxDist > 0) increaseStdevsNearInvalid(depth, increaseNearDepthBoundariesMaxDist, stdevInvalid, stdevs, multithread);
}

/*
 * compute a local std dev of depth for each pixel with valid depth, using only each depth point (as opposed to a local nbrhood) to get its stdev
 * (useful, eg, for putting into a diffingSingleFrameInfo)
 *
 * stdevs will be resized if empty
 *
 * stdevs: indexed by (y, x)
 *
 * increaseNearDepthBoundariesMaxDist: we'll increase stdevs of pts w/in this many pixels of depth boundaries (0 -> don't)
 */
void computeDepthMapStdevsPointwise(const cv::Mat_<float>& depth, boost::multi_array<float, 2>& stdevs, const uint32_t increaseNearDepthBoundariesMaxDist, const bool multithread)
{
	computeDepthMapStdevsPointwise(depth, stdevs, primesensor::stereoError(1), increaseNearDepthBoundariesMaxDist, multithread);
}
/*
 * use a non-real value of depth uncertainty at 1 m, to adjust diffing sensitivity
 */
void computeDepthMapStdevsPointwise(const cv::Mat_<float>& depth, boost::multi_array<float, 2>& stdevs, const float sigmaAt1m, const uint32_t increaseNearDepthBoundariesMaxDist, const bool multithread)
{
	ASSERT_ALWAYS(depth.rows > 0 && depth.cols > 0); //else cv::distanceTransform() will yell

	if(stdevs.num_elements() == 0) stdevs.resize(boost::extents[depth.rows][depth.cols]);

	const float invalidSigma = 1; //TODO ?

	for(int32_t i = 0, l = 0; i < depth.rows; i++)
		for(int32_t j = 0; j < depth.cols; j++, l++)
			if(depth(i, j) > 0)
			{
				stdevs[i][j] = std::max(1e-4/* smaller than any actual sensor noise */, sigmaAt1m * primesensor::stereoErrorRatio(depth(i, j)));
			}
			else //depth invalid
			{
				stdevs[i][j] = invalidSigma;
			}

	if(increaseNearDepthBoundariesMaxDist > 0)
	{
		increaseStdevsNearDepthBoundaries(depth, increaseNearDepthBoundariesMaxDist, invalidSigma, stdevs, multithread);

		//don't need it because it happens in the depth-boundaries call anyway because invalid is encoded as 0
		//increaseStdevsNearInvalid(depth, increaseNearDepthBoundariesMaxDist, invalidSigma, stdevs, multithread);
	}
}
