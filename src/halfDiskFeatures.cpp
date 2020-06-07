/*
 * halfDiskFeatures: over 2-d images
 *
 * Evan Herbst
 * 6 / 18 / 12
 */

#include <cmath>
#include <algorithm>
#include <iostream>
#define BOOST_DISABLE_ASSERTS //disable multiarray range checking -- gives a speedup of ~60% (20s vs 50s per frame) when using the integral-image method
#include <boost/multi_array.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "rgbd_util/eigen/Core"
#include "rgbd_util/eigen/LU" //inverse()
#include "rgbd_util/timer.h"
#include "rgbd_util/mathUtils.h"
#include "rgbd_util/parallelism.h"
#include "rgbd_util/threadPool.h"
#include "image_features/halfDiskFeatures.h"
using std::vector;
using std::cout;
using std::endl;
using Eigen::Vector2f;
using Eigen::Matrix2f;
using Eigen::MatrixXf;

#define USE_INTEGRAL_IMAGES //approximate disks with squares in integral imgs; gives huge speedup
#ifndef USE_INTEGRAL_IMAGES

cv::Mat computeHalfDiskImgFeatures(const std::vector<int32_t>& radii, const uint32_t numAngles, const cv::Mat& pixelFeats)
{
	rgbd::timer t;
	const int32_t rows = pixelFeats.size[0], cols = pixelFeats.size[1];
	const int matSize[3] = {rows, cols, (int)(pixelFeats.size[2] * radii.size())};
	cv::Mat feats(3, matSize, cv::DataType<float>::type);
	std::vector<float> angles(numAngles); //in [0, pi)
	for(uint32_t m = 0; m < numAngles; m++) angles[m] = m * M_PI / numAngles;
	Eigen::VectorXf sums[2]; //preallocate
	sums[0].resize(pixelFeats.size[2]);
	sums[1].resize(pixelFeats.size[2]);
	Eigen::VectorXf maxSums[2];
	for(uint32_t k = 0; k < radii.size(); k++)
	{
		const int32_t radius = radii[k];

		boost::multi_array<bool, 3> pixelSidesByAngle(boost::extents[numAngles][2 * radius + 1][2 * radius + 1]); //for each pixel offset and angle, which disk side is it on
		pixelSidesByAngle.reindex(boost::array<int32_t, 3>{{0, -radius, -radius}});
		for(int32_t i2 = -radius; i2 <= radius; i2++) //use all pixels in a square rather than in a circle, for ease
			for(int32_t j2 = -radius; j2 <= radius; j2++)
				for(uint32_t m = 0; m < numAngles; m++)
				{
					const float angle = atan2(i2, j2);
					pixelSidesByAngle[m][i2][j2] = (angle < angles[m] && angles[m] - angle <= M_PI); //which half-disk
				}

		for(int32_t i = 0; i < rows; i++)
			for(int32_t j = 0; j < cols; j++)
			{
				float maxResponse = 0;
				for(uint32_t m = 0; m < numAngles; m++)
				{
					sums[0].setZero();
					sums[1].setZero();
					for(int32_t i2 = std::max(0, i - radius); i2 <= std::min(rows - 1, i + radius); i2++) //use all pixels in a square rather than in a circle, for ease
						for(int32_t j2 = std::max(0, j - radius); j2 <= std::min(cols - 1, j + radius); j2++)
						{
							const bool side = pixelSidesByAngle[m][i2 - i][j2 - j];
							const float* ptr = pixelFeats.ptr<float>(i2, j2);
							for(uint32_t l = 0; l < pixelFeats.size[2]; l++) sums[side][l] += *ptr++;//pixelFeats.at<float>(i2, j2, l);
						}
					const float response = (sums[1] - sums[0]).array().abs().sum();
					if(response >= maxResponse)
					{
						maxResponse = response;
						maxSums[0] = sums[0];
						maxSums[1] = sums[1];
					}
				}
				for(uint32_t l = 0; l < pixelFeats.size[2]; l++)
					feats.at<float>(i, j, k * pixelFeats.size[2] + l) = fabs(maxSums[1][l] - maxSums[0][l]);
			}
	}
	t.stop("compute half-disk feats");
	return feats;
}

#else //use integral imgs

cv::Mat computeHalfDiskImgFeatures(const std::vector<int32_t>& radii, const uint32_t numAngles, const cv::Mat& pixelFeats, const halfDisk2dFeatFunc featFunc)
{
	rgbd::timer t;
	const uint32_t numThreads = getSuggestedThreadCount();
	rgbd::threadGroup tg(numThreads);
	const int32_t rows = pixelFeats.size[0], cols = pixelFeats.size[1];
	const int matSize[3] = {rows, cols, (int)((featFunc ? 1 : pixelFeats.size[2]) * radii.size())};
	cv::Mat result(3, matSize, cv::DataType<float>::type, cv::Scalar(0)); //aggregated over angles
	cv::Mat maxAbsSum(rows, cols, cv::DataType<float>::type, cv::Scalar(0)); //over angles
	std::vector<float> angles(numAngles); //in [0, pi)
	for(uint32_t m = 0; m < numAngles; m++) angles[m] = m * M_PI / numAngles;
	for(uint32_t m = 0; m < numAngles; m++)
	{
		rgbd::timer u;
		/*
		 * rotate the per-pixel features
		 */
		boost::multi_array<float, 3> rotPixelFeats;
		boost::multi_array<bool, 2> rotFeatValidity;
		rotateValuesInImgSpace(pixelFeats, angles[m], rotPixelFeats, rotFeatValidity);
		Matrix2f rotMtx;
		rotMtx(0, 0) = cos(angles[m]);
		rotMtx(0, 1) = -sin(angles[m]);
		rotMtx(1, 0) = sin(angles[m]);
		rotMtx(1, 1) = cos(angles[m]);
		const int32_t rotXmin = rotPixelFeats.index_bases()[1], rotXmax = rotXmin + rotPixelFeats.shape()[1] - 1,
			rotYmin = rotPixelFeats.index_bases()[0], rotYmax = rotYmin + rotPixelFeats.shape()[0] - 1;
		u.stop("rotate feats");
#if 0 //debugging
		rotPixelFeats.reindex(0);
		for(uint32_t k = 0; k < pixelFeats.size[2]; k++)
		{
			cv::Mat img(pixelFeats.size[0], pixelFeats.size[1], CV_8UC1, cv::Scalar(0));
			for(int i = 0; i < pixelFeats.size[0]; i++)
				for(int j = 0; j < pixelFeats.size[1]; j++)
					img.at<uint8_t>(i, j) = std::max(0, std::min(255, 128 + (int)rint(127 * pixelFeats.at<float>(i, j, k))));
			cv::imwrite((boost::format("img%1%.png") % k).str(), img);

			cv::Mat rimg(rotPixelFeats.shape()[0], rotPixelFeats.shape()[1], CV_8UC1, cv::Scalar(0));
			for(int i = 0; i < rotPixelFeats.shape()[0]; i++)
				for(int j = 0; j < rotPixelFeats.shape()[1]; j++)
					rimg.at<uint8_t>(i, j) = std::max(0, std::min(255, 128 + (int)rint(127 * rotPixelFeats[i][j][k])));
			cv::imwrite((boost::format("rimg%1%.png") % k).str(), rimg);
		}
		cout << "wrote imgs" << endl;
		{int q; std::cin >> q;}
#endif
		u.restart();

		/*
		 * compute integral imgs
		 */
		boost::multi_array<float, 3> rotPixelFeatsIntegralImg;
		boost::multi_array<uint32_t, 2> rotFeatValidityIntegralImg;
		computeIntegralImage(rotPixelFeats, rotPixelFeatsIntegralImg);
		computeIntegralImage(rotFeatValidity, rotFeatValidityIntegralImg);
		u.stop("make integral imgs");
		u.restart();

		/*
		 * compute feature integrals over half-squares
		 */
		const vector<unsigned int> indices2 = partitionEvenly(rows, numThreads);
		for(uint32_t m = 0; m < numThreads; m++)
			tg.addTask([&,m]()
				{
					for(int32_t i = indices2[m]; i < indices2[m + 1]; i++)
						for(int32_t j = 0; j < cols; j++)
						{
							const Vector2f rotPt = rotMtx * Vector2f(j, i); //project an orig-img pixel into the rotated img
							const int32_t rotX = rint(rotPt.x()), rotY = rint(rotPt.y());

							/*
							 * choose which angle to use the difference vector for
							 * (right now, use sum of abs diffs to choose; TODO a better way? include them all in the result like xf's nips12 submission?)
							 */
							vector<float> diffs(result.size[2], 0);
							for(uint32_t k = 0; k < radii.size(); k++)
							{
								const int32_t radius = radii[k];
								const int32_t xmin = std::max(rotXmin, rotX - radius), xmax = std::min(rotXmax, rotX + radius), ymin = std::max(rotYmin, rotY - radius), ymax = std::min(rotYmax, rotY + radius); //of the integral-img nbrhood
								const int32_t side1validitySum = rotFeatValidityIntegralImg[ymax][rotX] - rotFeatValidityIntegralImg[ymin][rotX] - rotFeatValidityIntegralImg[ymax][xmin] + rotFeatValidityIntegralImg[ymin][xmin],
									side2validitySum = rotFeatValidityIntegralImg[ymax][xmax] - rotFeatValidityIntegralImg[ymin][xmax] - rotFeatValidityIntegralImg[ymax][rotX] + rotFeatValidityIntegralImg[ymin][rotX];
								if(side1validitySum + side2validitySum >= 4/* TODO ? */)
								{
									if(featFunc)
									{
										//copy each side's sum to a vector
										Eigen::VectorXf side1feats(pixelFeats.size[2]), side2feats(pixelFeats.size[2]);
										float* out = side1feats.data(),
											*ia = rotPixelFeatsIntegralImg[ymin][xmin].origin(), *ib = rotPixelFeatsIntegralImg[ymin][rotX].origin(),
											*ic = rotPixelFeatsIntegralImg[ymax][xmin].origin(), *id = rotPixelFeatsIntegralImg[ymax][rotX].origin();
										for(uint32_t l = 0; l < pixelFeats.size[2]; l++, out++, ia++, ib++, ic++, id++) *out = (*id - *ic - *ib + *ia) / side1validitySum;
										out = side2feats.data();
										ia = rotPixelFeatsIntegralImg[ymin][rotX].origin();
										ib = rotPixelFeatsIntegralImg[ymin][xmax].origin();
										ic = rotPixelFeatsIntegralImg[ymax][rotX].origin();
										id = rotPixelFeatsIntegralImg[ymax][xmax].origin();
										for(uint32_t l = 0; l < pixelFeats.size[2]; l++, out++, ia++, ib++, ic++, id++) *out = (*id - *ic - *ib + *ia) / side2validitySum;
										diffs[k] = featFunc(side1feats, side2feats);
									}
									else
									{
										float* out = diffs.data() + k * pixelFeats.size[2],
											*ia = rotPixelFeatsIntegralImg[ymin][xmin].origin(), *ib = rotPixelFeatsIntegralImg[ymin][rotX].origin(), *ic = rotPixelFeatsIntegralImg[ymin][xmax].origin(),
											*id = rotPixelFeatsIntegralImg[ymax][xmin].origin(), *ie = rotPixelFeatsIntegralImg[ymax][rotX].origin(), *ig = rotPixelFeatsIntegralImg[ymax][xmax].origin();
										for(uint32_t l = 0; l < pixelFeats.size[2]; l++, out++, ia++, ib++, ic++, id++, ie++, ig++)
										{
											*out = (*ig - *ie - *ic + *ib) / side2validitySum - (*ie - *id - *ib + *ia) / side1validitySum;
										}
									}
								}
							}

							float absSum = 0;
							for(float d : diffs) absSum += fabs(d);
							if(absSum > maxAbsSum.at<float>(i, j))
							{
								for(uint32_t l = 0; l < diffs.size(); l++) result.at<float>(i, j, l) = diffs[l];
								maxAbsSum.at<float>(i, j) = absSum;
							}
						}
				});
		tg.wait();
		u.stop("compute feats");
	}
	t.stop("compute half-disk feats");
	return result;
}
#endif
