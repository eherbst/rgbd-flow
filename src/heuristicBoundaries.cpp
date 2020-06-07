/*
 * heuristicBoundaries: use single-frame features to estimate boundary points in 2-d
 *
 * Evan Herbst
 * 6 / 26 / 12
 */

#include <cstdint>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <opencv2/core/core.hpp>
#include "rgbd_util/timer.h"
#include "rgbd_util/parallelism.h"
#include "rgbd_util/threadPool.h"
#include "pcl_rgbd/depth_to_cloud_lib.h"
#include "pcl_rgbd/pointTypes.h"
#include "cloud_features/principalCurvatures.h"
#include "image_features/halfDiskFeatures.h"
#include "boundary_probs_heuristic/heuristicBoundaries.h"
using std::vector;
using std::cout;
using std::endl;
using Eigen::Vector3f;
using Eigen::VectorXf;
using Eigen::VectorXd;

/*
 * compute a boundary prob between each pixel and each of its nbr pixels
 *
 * not Malik's Pb, but a very heuristic rough boundary-prob map to identify all possible boundary pts in an rgbd frame
 *
 * img: rgb
 *
 * depth should not have been renormalized (eg to [0, 1])
 *
 * return: p(boundary) for each pixel pair in row-major order
 * (first two output dimensions are y, x; third is [right, down] nbrs)
 */
boost::multi_array<float, 3> computeHeuristicPb(const rgbd::CameraParams& camParams, const cv::Mat& img, const cv::Mat_<float>& depth)
{
	rgbd::timer u;
	rgbd::timer t;

	const uint32_t numThreads = getSuggestedThreadCount();

	t.restart();
	pcl::PointCloud<rgbd::pt>::Ptr organizedCloudPtr(new pcl::PointCloud<rgbd::pt>);
	depth_to_cloud(depth, true, true, *organizedCloudPtr, camParams);
	boost::multi_array<bool, 2> validityGrid;
	rgbd::getValidityGrid(*organizedCloudPtr, validityGrid);
	rgbd::setOrganizedNormals(*organizedCloudPtr, validityGrid, 4, 1, .05);
	t.stop("get cloud ptrs");

	t.restart();
	const uint32_t nbrhoodRadiusInPixels = 3; //TODO ?
	const std::tuple<cloudFeatDescs, std::vector<char>> curvatures = std::move(computePrincipalCurvaturesOrganized<rgbd::pt>(organizedCloudPtr, nbrhoodRadiusInPixels));
	const cloudFeatDescs& curvDescs = std::get<0>(curvatures);
	const vector<char>& curvValidity = std::get<1>(curvatures);
	boost::multi_array<float, 2> curvOrganized(boost::extents[img.rows][img.cols]);
	std::fill(curvOrganized.data(), curvOrganized.data() + curvOrganized.num_elements(), 0);
	for(uint32_t i = 0, l = 0; i < img.rows; i++)
		for(uint32_t j = 0; j < img.cols; j++, l++)
			if(curvValidity[l])
				curvOrganized[i][j] = curvDescs(l, 0);
	t.stop("get curv feats");

	t.restart();
	boost::multi_array<float, 3> colsImg(boost::extents[camParams.yRes][camParams.xRes][3]); //img w/ rgb as 3 channels
	for(uint32_t i = 0; i < camParams.yRes; i++)
		for(uint32_t j = 0; j < camParams.xRes; j++)
		{
			for(uint32_t k = 0; k < 3; k++) colsImg[i][j][k] = img.at<cv::Vec3b>(i, j)[k] / 255.0;
		}
	t.stop("copy rgb img");

	t.restart();
	boost::multi_array<float, 3> normalsImg(boost::extents[camParams.yRes][camParams.xRes][3]);
	boost::multi_array<uint32_t, 3> normalValidityImg(boost::extents[camParams.yRes][camParams.xRes][1]);
	std::fill(normalsImg.data(), normalsImg.data() + normalsImg.num_elements(), 0);
	std::fill(normalValidityImg.data(), normalValidityImg.data() + normalValidityImg.num_elements(), 0);
	for(uint32_t i = 0, l = 0; i < (uint32_t)img.rows; i++)
		for(uint32_t j = 0; j < (uint32_t)img.cols; j++, l++)
		{
			const rgbd::pt& pt = organizedCloudPtr->points[l];
			const Vector3f& normal = rgbd::ptNormal2eigen<Vector3f>(pt);
			for(uint32_t k = 0; k < 3; k++) normalsImg[i][j][k] = normal[k];
			if(std::isnan(normal[0]) || std::isinf(normal[0]) || std::isnan(normal[1]) || std::isinf(normal[1]) || std::isnan(normal[2]) || std::isinf(normal[2])
				|| fabs(normal.squaredNorm() - 1) < 1e-4)
				normalValidityImg[i][j][0] = 1;
		}
	t.stop("fill normals info");

//#define USE_CONSTANT_RADIUS
#ifdef USE_CONSTANT_RADIUS
	const int32_t radius = 2; //TODO ?
#else //use a different half-disk radius at each pixel
	t.restart();
	boost::multi_array<int32_t, 2> radii(boost::extents[camParams.yRes][camParams.xRes]);
{
	const int32_t minRadius = 1, maxRadius = 5; //TODO ?
	const float metricRadius = .02; //pick a pixel radius to get this radius in meters
	for(uint32_t i = 0, l = 0; i < (uint32_t)img.rows; i++)
		for(uint32_t j = 0; j < (uint32_t)img.cols; j++, l++)
		{
			if(depth(i, j) > 0) radii[i][j] = std::max(minRadius, std::min(maxRadius, (int32_t)rint(camParams.focalLength * metricRadius / depth(i, j))));
			else radii[i][j] = maxRadius;
		}
}
	t.stop("compute radii");
#endif

	boost::multi_array<float, 2> colorHorizHalfDisks(boost::extents[camParams.yRes][camParams.xRes]), colorVertHalfDisks(boost::extents[camParams.yRes][camParams.xRes]); //wrt pixels above (vert) and to left (horiz)
	boost::multi_array<float, 2> normalHorizHalfDisks(boost::extents[camParams.yRes][camParams.xRes]), normalVertHalfDisks(boost::extents[camParams.yRes][camParams.xRes]);
{
	std::fill(colorHorizHalfDisks.data(), colorHorizHalfDisks.data() + colorHorizHalfDisks.num_elements(), 0);
	std::fill(normalHorizHalfDisks.data(), normalHorizHalfDisks.data() + normalHorizHalfDisks.num_elements(), 0);
	std::fill(colorVertHalfDisks.data(), colorVertHalfDisks.data() + colorVertHalfDisks.num_elements(), 0);
	std::fill(normalVertHalfDisks.data(), normalVertHalfDisks.data() + normalVertHalfDisks.num_elements(), 0);

	/*
	 * compute integral imgs
	 */
	t.restart();
	boost::multi_array<float, 3> colorIntegralImg, normalIntegralImg;
	boost::multi_array<uint32_t, 3> normalValidityIntegralImg;
	computeIntegralImage(colsImg, colorIntegralImg);
	computeIntegralImage(normalsImg, normalIntegralImg);
	computeIntegralImage(normalValidityImg, normalValidityIntegralImg);
	t.stop("compute integral imgs");

	/*
	 * compute feature integrals over half-squares
	 */
	t.restart();
#define USE_CONVEXITY
	rgbd::threadGroup tg(numThreads);
	const vector<unsigned int> indices = partitionEvenly(camParams.xRes, numThreads);
	for(uint32_t m = 0; m < numThreads; m++)
		tg.addTask([&,m]()
			{
	for(int32_t i = 0; i < (int32_t)camParams.yRes; i++)
		for(int32_t j = (int32_t)indices[m]; j < (int32_t)indices[m + 1]; j++)
		{
#ifdef USE_CONSTANT_RADIUS
			const int32_t halfwidth = radius;
#else
			const int32_t halfwidth = radii[i][j];
#endif
			const int32_t xmin = std::max(0, j - halfwidth), xmax = std::min((int32_t)camParams.xRes - 1, j + halfwidth - 1), ymin = std::max(0, i - halfwidth), ymax = std::min((int32_t)camParams.yRes - 1, i + halfwidth - 1); //of the integral-img nbrhood
			const float size = (xmax - xmin + 1) * (ymax - ymin + 1);
			const float hsize0 = (j - xmin) * (ymax - ymin + 1), hsize1 = (xmax - j + 1) * (ymax - ymin + 1), vsize0 = (xmax - xmin + 1) * (i - ymin), vsize1 = (xmax - xmin + 1) * (ymax - i + 1); //half-disk sizes
#define HORIZ_HALFDISK1(ii, k) (ii[ymax][xmax][k] - ii[ymax][j - 1][k] - ii[ymin - 1][xmax][k] + ii[ymin - 1][j - 1][k])
#define HORIZ_HALFDISK0(ii, k) (ii[ymax][j - 1][k] - ii[ymin - 1][j - 1][k] - ii[ymax][xmin - 1][k] + ii[ymin - 1][xmin - 1][k])
#define HORIZ_HALFDISK(ii, k) (HORIZ_HALFDISK0(ii, k) - HORIZ_HALFDISK1(ii, k))
#define VERT_HALFDISK1(ii, k) (ii[ymax][xmax][k] - ii[i - 1][xmax][k] - ii[ymax][xmin - 1][k] + ii[i - 1][xmin - 1][k])
#define VERT_HALFDISK0(ii, k) (ii[i - 1][xmax][k] - ii[i - 1][xmin - 1][k] - ii[ymin - 1][xmax][k] + ii[ymin - 1][xmin - 1][k])
#define VERT_HALFDISK(ii, k) (VERT_HALFDISK0(ii, k) - VERT_HALFDISK1(ii, k))
			if(j > 0) //so that neither half-disk is empty
			{
				colorHorizHalfDisks[i][j] = sqrt(sqr(HORIZ_HALFDISK1(colorIntegralImg, 0) / hsize1 - HORIZ_HALFDISK0(colorIntegralImg, 0) / hsize0)
															+ sqr(HORIZ_HALFDISK1(colorIntegralImg, 1) / hsize1 - HORIZ_HALFDISK0(colorIntegralImg, 1) / hsize0)
															+ sqr(HORIZ_HALFDISK1(colorIntegralImg, 2) / hsize1 - HORIZ_HALFDISK0(colorIntegralImg, 2) / hsize0));

				Vector3f normfeats0, normfeats1;
				for(int k = 0; k < 3; k++)
				{
					normfeats0[k] = HORIZ_HALFDISK0(normalIntegralImg, k);
					normfeats1[k] = HORIZ_HALFDISK1(normalIntegralImg, k);
				}
				normfeats0.normalize();
				normfeats1.normalize();
				const uint32_t normvalid0 = HORIZ_HALFDISK0(normalValidityIntegralImg, 0),
					normvalid1 = HORIZ_HALFDISK1(normalValidityIntegralImg, 0);
				normalHorizHalfDisks[i][j] = (std::min(normvalid0, normvalid1) < 3) ? 0 //if no evidence, never guess there's an edge based on normals
					: .5 * (1 - normfeats0.dot(normfeats1));
#ifdef USE_CONVEXITY
				if(std::min(normvalid0, normvalid1) > 3)
				{
					const Vector3f dx = (rgbd::ptX2eigen<Vector3f>(organizedCloudPtr->points[i * camParams.xRes + (j - 1)]) - rgbd::ptX2eigen<Vector3f>(organizedCloudPtr->points[i * camParams.xRes + j])).normalized();
					const double convexity = (normfeats0.dot(dx) + normfeats1.dot(-dx)) / 4 + .5; //in [0, 1)
					normalHorizHalfDisks[i][j] *= (1 - convexity);
				}
#endif
			}
			if(i > 0)
			{
				colorVertHalfDisks[i][j] = sqrt(sqr(VERT_HALFDISK1(colorIntegralImg, 0) / vsize1 - VERT_HALFDISK0(colorIntegralImg, 0) / vsize0)
															+ sqr(VERT_HALFDISK1(colorIntegralImg, 1) / vsize1 - VERT_HALFDISK0(colorIntegralImg, 1) / vsize0)
															+ sqr(VERT_HALFDISK1(colorIntegralImg, 2) / vsize1 - VERT_HALFDISK0(colorIntegralImg, 2) / vsize0));

				Vector3f normfeats0, normfeats1;
				for(int k = 0; k < 3; k++)
				{
					normfeats0[k] = VERT_HALFDISK0(normalIntegralImg, k);
					normfeats1[k] = VERT_HALFDISK1(normalIntegralImg, k);
				}
				normfeats0.normalize();
				normfeats1.normalize();
				const uint32_t normvalid0 = VERT_HALFDISK0(normalValidityIntegralImg, 0),
					normvalid1 = VERT_HALFDISK1(normalValidityIntegralImg, 0);
				normalVertHalfDisks[i][j] = (std::min(normvalid0, normvalid1) < 3) ? 0 //if no evidence, never guess there's an edge based on normals
					: .5 * (1 - normfeats0.dot(normfeats1));
#ifdef USE_CONVEXITY
				if(std::min(normvalid0, normvalid1) > 3)
				{
					const Vector3f dx = (rgbd::ptX2eigen<Vector3f>(organizedCloudPtr->points[(i - 1) * camParams.xRes + j]) - rgbd::ptX2eigen<Vector3f>(organizedCloudPtr->points[i * camParams.xRes + j])).normalized();
					const double convexity = (normfeats0.dot(dx) + normfeats1.dot(-dx)) / 4 + .5; //in [0, 1)
					normalVertHalfDisks[i][j] *= (1 - convexity);
				}
#endif
			}
#undef HORIZ_HALFDISK
#undef VERT_HALFDISK
		}
			});
	tg.wait();
	t.stop("compute feature integrals");
#undef USE_CONVEXITY
}

	/*
	 * gaussian weights are apparently somewhat motivated here; see Letouzey bmvc11 for a pointer to info on that
	 */
	t.restart();
	const float curvSigma = 12, //TODO magic numbers
		depthSigma = .02,
		colSigma = .7,
		normSigma = .4;
	boost::multi_array<float, 3> boundaryProbs(boost::extents[camParams.yRes][camParams.xRes][2]);
	std::fill(boundaryProbs.data(), boundaryProbs.data() + boundaryProbs.num_elements(), 0);
	const auto CALC = [&](const uint32_t i, const uint32_t j, const uint32_t k, const float z1, const float z2, const float dcol, const float dnorm, const float curv)
		{
			float colProb;
			float depthProb;
			if(std::max(z1, z2) <= 0) //neither depth is valid
			{
				depthProb = 0; //if depth isn't defined, don't guess there's a boundary based on depth info
				colProb = 1 - exp(-sqr(dcol / colSigma));
			}
			else if(std::min(z1, z2) <= 0) //one depth is valid, one isn't
			{
				//measure the size of this blob of invalid pixels in the given direction
				const int32_t di[4] = {0, -1, 0, 1}, dj[4] = {-1, 0, 1, 0}; //pixel increments by direction code
				const uint32_t dir = (z1 <= 0) ? (k + 2) % 4 : k; //direction to move in (depends on which is the invalid reading)
				int32_t i0 = (z1 <= 0) ? i : i + di[k], j0 = (z1 <= 0) ? j : j + dj[k]; //start at whichever of z1, z2 is invalid
				bool oob = false;
				uint32_t size = 0; //# invalid readings in a line
				while(j0 >= 0 && j0 < depth.cols && i0 >= 0 && i0 < depth.rows && depth(i0, j0) <= 0)
				{
					size++;
					i0 += di[dir];
					j0 += dj[dir];
				}
				if(j0 < 0 || j0 >= depth.cols || i0 < 0 || i0 >= depth.rows) oob = true;

				if(oob) depthProb = 0; //if the invalid region extends to edge of img, don't assume there's a boundary
				else depthProb = linterp(.2, 1, std::min(1.0f, (float)size / 30)); //TODO ?: have prob depend on invalid-region size

				colProb = 1 - exp(-sqr(dcol / colSigma));
			}
			else
			{
				const float dz = fabs(z1 - z2) / primesensor::stereoErrorRatio(std::min(z1, z2));
				depthProb = 1 - exp(-sqr(dz / depthSigma));

				colProb = 0; //don't use color info if we have good depth info
			}
			const float /*curvProb = 1 - exp(-sqr(curv / curvSigma)),*/
				normProb = 1 - exp(-sqr(dnorm / normSigma));
			boundaryProbs[i][j][k] = std::max(depthProb, std::max(colProb, std::max(0.0f/*curvProb*/, normProb)));

			//TODO TODO hack to avoid single pixels at img edges blowing up in optimizations due to high hpb w/ all their neighbors; how to avoid this hack?
			if(((i == 0 || i == camParams.yRes - 2) && k == 1) || ((j == 0 || j == camParams.xRes - 2) && k == 0)) boundaryProbs[i][j][k] = 0;
		};
{
	rgbd::threadGroup tg(numThreads);
	const vector<unsigned int> indices = partitionEvenly(camParams.xRes, numThreads);
	for(uint32_t m = 0; m < numThreads; m++)
		tg.addTask([&,m]()
			{
				for(int32_t i = 0; i < (int32_t)camParams.yRes; i++)
					for(int32_t j = indices[m]; j < indices[m + 1]; j++)
					{
						if(j < (int32_t)camParams.xRes - 1)
						{
							const float curv = curvOrganized[i][j];
							const float
								z1 = depth(i, j), z2 = depth(i, j + 1),
								dcol = colorHorizHalfDisks[i][j + 1],
								dnorm = normalHorizHalfDisks[i][j + 1];
							CALC(i, j, 0, z1, z2, dcol, dnorm, curv);
						}
						if(i < (int32_t)camParams.yRes - 1)
						{
							const float curv = curvOrganized[i][j];
							const float
								z1 = depth(i, j), z2 = depth(i + 1, j),
								dcol = colorVertHalfDisks[i + 1][j],
								dnorm = normalVertHalfDisks[i + 1][j];
							CALC(i, j, 1, z1, z2, dcol, dnorm, curv);
						}
					}
			});
	tg.wait();
}
	t.stop("compute final hpb feats");

#if 1
	/*
	 * non-max suppression
	 *
	 * TODO TODO TODO 20120912: for some synthetic data (blackSphereRight), at least, the values prior to this are the same for several pixels in a row, so after NMS we end up making a strong boundary guess at a point that's way off the real boundary -- ??
	 */
	t.restart();
{
	auto boundaryProbs2 = boundaryProbs;
	rgbd::threadGroup tg(numThreads);
	const vector<unsigned int> indices = partitionEvenly(camParams.xRes, numThreads);
	for(uint32_t m = 0; m < numThreads; m++)
		tg.addTask([&,m]()
			{
				for(int32_t i = 0, l = 0; i < (int32_t)camParams.yRes; i++)
					for(int32_t j = (int32_t)indices[m]; j < (int32_t)indices[m + 1]; j++, l++)
					{
#ifdef USE_CONSTANT_RADIUS
						const int32_t halfwidth = radius;
#else
						const int32_t halfwidth = radii[i][j];
#endif

					{
						int32_t maxIndex = -1;
						for(int32_t k = std::max(0, i - halfwidth); k <= std::min((int32_t)camParams.yRes - 1, i + halfwidth); k++)
							if(maxIndex < 0 || boundaryProbs[k][j][1] > boundaryProbs[maxIndex][j][1])
								maxIndex = k;
						if(maxIndex != i) boundaryProbs2[i][j][1] = 0;
					}

					{
						int32_t maxIndex = -1;
						for(int32_t k = std::max(0, j - halfwidth); k <= std::min((int32_t)camParams.xRes - 1, j + halfwidth); k++)
							if(maxIndex < 0 || boundaryProbs[i][k][0] > boundaryProbs[i][maxIndex][0])
								maxIndex = k;
						if(maxIndex != j) boundaryProbs2[i][j][0] = 0;
					}

					}
			});
	tg.wait();
	boundaryProbs = boundaryProbs2;
}
	t.stop("do non-max suppression");
#endif

	u.stop("run hpb2d");
	return boundaryProbs;
}
