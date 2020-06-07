/*
 * halfDiskFeatures: over 2-d images
 *
 * Evan Herbst
 * 6 / 20 / 12
 */

#include <cstdint>
#include <cmath>
#include <vector>
#include <algorithm>
#include "rgbd_util/assert.h"
#include <Eigen/LU> //inverse()
#include "rgbd_util/mathUtils.h"
#include "rgbd_util/parallelism.h"
#include "rgbd_util/threadPool.h"

/*
 * resample values by rotating in xy by angle; the result arrays will be large enough to contain samples for all values, and won't be indexed starting at 0
 *
 * result arrays will be allocated
 *
 * pre: values has type ValueT
 */
template <typename ValueT>
void rotateValuesInImgSpace(const cv::Mat& values, const float angle, boost::multi_array<ValueT, 3>& rotatedValues, boost::multi_array<bool, 2>& resultValidity)
{
	ASSERT_ALWAYS(values.size[0] >= 0 && values.size[1] >= 0); //must be a 3-d image
	const int32_t rows = values.size[0], cols = values.size[1];
	Eigen::Matrix2f rotMtx;
	rotMtx(0, 0) = cos(angle);
	rotMtx(0, 1) = -sin(angle);
	rotMtx(1, 0) = sin(angle);
	rotMtx(1, 1) = cos(angle);
	Eigen::MatrixXf corners(2, 4);
	corners.col(0) = Eigen::Vector2f(0, 0);
	corners.col(1) = Eigen::Vector2f(0, rows);
	corners.col(2) = Eigen::Vector2f(cols, 0);
	corners.col(3) = Eigen::Vector2f(cols, rows);
	const Eigen::MatrixXf rotatedCorners = rotMtx * corners;
	const int32_t rotXmin = floor(rotatedCorners.row(0).minCoeff()), rotXmax = ceil(rotatedCorners.row(0).maxCoeff()),
		rotYmin = floor(rotatedCorners.row(1).minCoeff()), rotYmax = ceil(rotatedCorners.row(1).maxCoeff());
	rotatedValues.resize(boost::extents[rotYmax - rotYmin + 1][rotXmax - rotXmin + 1][values.size[2]]);
	resultValidity.resize(boost::extents[rotYmax - rotYmin + 1][rotXmax - rotXmin + 1]);
	rotatedValues.reindex(boost::array<int32_t, 3>{{rotYmin, rotXmin, 0}});
	resultValidity.reindex(boost::array<int32_t, 3>{{rotYmin, rotXmin, 0}});
	std::fill(rotatedValues.data(), rotatedValues.data() + rotatedValues.num_elements(), 0);
	std::fill(resultValidity.data(), resultValidity.data() + resultValidity.num_elements(), false);
	const Eigen::Matrix2f rotInv = rotMtx.inverse();
#ifdef MULTITHREAD_HALF_DISK_IMG_FEATS
	const uint32_t numThreads = 1;//getSuggestedThreadCount();
	const std::vector<unsigned int> indices = partitionEvenly(rotYmax - rotYmin + 1, numThreads);
	rgbd::threadGroup tg(numThreads);
	for(uint32_t m = 0; m < numThreads; m++)
		tg.addTask([&,m]()
			{
				const int32_t rows = values.size[0], cols = values.size[1]; //TODO why do I have to do this to make rows and cols not suddenly be 0 when I get into the lambda?
				for(int32_t i = rotYmin + (int64_t)indices[m]; i < rotYmin + (int64_t)indices[m + 1]; i++)
#else
	for(int32_t i = rotYmin; i <= rotYmax; i++)
#endif
	{
		for(int32_t j = rotXmin; j <= rotXmax; j++)
		{
			const Eigen::Vector2f rotPt = rotInv * Eigen::Vector2f(j, i);// + Eigen::Vector2f(1e-4, 1e-4)/* TODO ? this is to get it in bounds at row or col 0 */;
			if(rotPt.x() >= 0 && rotPt.x() <= cols - 1 && rotPt.y() >= 0 && rotPt.y() <= rows - 1)
			{
				const int32_t x0 = floor(rotPt.x()), y0 = floor(rotPt.y()), x1 = std::min(cols - 1, x0 + 1), y1 = std::min(rows - 1, y0 + 1);
				const float alphaX = rotPt.x() - x0, alphaY = rotPt.y() - y0;
#if 1 //for speed (~3s/frame)
				ValueT* out = rotatedValues[i][j].origin();
				const ValueT* in00 = values.ptr<ValueT>(y0, x0), *in01 = values.ptr<ValueT>(y1, x0), *in10 = values.ptr<ValueT>(y0, x1), *in11 = values.ptr<ValueT>(y1, x1);
				for(uint32_t l = 0; l < values.size[2]; l++, out++, in00++, in01++, in10++, in11++)
				{
					*out = static_cast<ValueT>(linterp(linterp(*in00, *in10, alphaX), linterp(*in01, *in11, alphaX), alphaY));
				}
#else
				for(uint32_t l = 0; l < values.size[2]; l++)
					rotatedValues[i][j][l] = linterp(
														linterp(values.at<ValueT>(y0, x0, l), values.at<ValueT>(y0, x1, l), alphaX),
														linterp(values.at<ValueT>(y1, x0, l), values.at<ValueT>(y1, x1, l), alphaX),
														alphaY);
#endif
				resultValidity[i][j] = true;
			}
		}
	}
#ifdef MULTITHREAD_HALF_DISK_IMG_FEATS
			});
	tg.wait();
#endif
}

/*
 * the result array will be allocated
 *
 * pre: D is 2 or 3
 *
 * post: valid indices for integralImg in the x and y dimensions will start one before valid indices for img (and all the values in the first row and column will be 0)
 */
template <typename T, typename T2, const size_t D>
void computeIntegralImage(const boost::multi_array<T, D>& img, boost::multi_array<T2, D>& integralImg)
{
	static_assert(D == 2 || D == 3, "bad dimensionality");
	boost::array<size_t, D> dims;
	std::copy(img.shape(), img.shape() + D, dims.begin());
	for(size_t i = 0; i < 2; i++) dims[i]++;
	integralImg.resize(dims);
	boost::array<int32_t, D> bases;
	std::copy(img.index_bases(), img.index_bases() + D, bases.begin());
	for(size_t i = 0; i < 2; i++) bases[i]--;
	integralImg.reindex(bases);
	const size_t size0 = img.shape()[0], size1 = img.shape()[1], size2 = (D == 3) ? img.shape()[2] : 1;

#if 1 //use pointers for speed (~1.4s/frame)
	const T* pfin = img.data();
	T2* pfout = integralImg.data();

	/*
	 * first row (all zeros)
	 */
	for(size_t l = 0; l < (size1 + 1) * size2; l++) *pfout++ = 0;

	/*
	 * second row, first col
	 */
	for(size_t l = 0; l < size2; l++) *pfout++ = 0;

	/*
	 * rest of second row
	 */
	for(size_t j = 0; j < size1; j++)
	{
		for(uint32_t l = 0; l < size2; l++, pfout++)
			*pfout = *(pfout - size2) + *pfin++;
	}

	/*
	 * rest of rows
	 */
	for(size_t i = 1; i < size0; i++)
	{
		for(size_t l = 0; l < size2; l++) *pfout++ = 0;
		std::vector<T2> rowSum(size2, 0); //sum of current row so far
		for(size_t j = 0; j < size1; j++)
		{
			for(uint32_t l = 0; l < size2; l++, pfout++)
			{
				*pfout = *(pfout - (size1 + 1) * size2) + rowSum[l] + *pfin;
				rowSum[l] += *pfin++;
			}
		}
	}
#else
	TODO fix to have first row and col be zero
	for(uint32_t l = 0; l < pixelFeats.size[2]; l++)
		rotPixelFeatsIntegralImg[rotYmin][rotXmin][l] = rotPixelFeats[rotYmin][rotXmin][l];
	rotFeatValidityIntegralImg[rotYmin][rotXmin] = rotFeatValidity[rotYmin][rotXmin];

	for(int32_t j = rotXmin + 1; j <= rotXmax; j++)
	{
		for(uint32_t l = 0; l < pixelFeats.size[2]; l++)
			rotPixelFeatsIntegralImg[rotYmin][j][l] = rotPixelFeatsIntegralImg[rotYmin][j - 1][l] + rotPixelFeats[rotYmin][j][l];
		rotFeatValidityIntegralImg[rotYmin][j] = rotFeatValidityIntegralImg[rotYmin][j - 1] + rotFeatValidity[rotYmin][j];
	}

	for(int32_t i = rotYmin + 1; i <= rotYmax; i++)
	{
		vector<float> rowSum(pixelFeats.size[2], 0);
		uint32_t rowVSum = 0;
		for(int32_t j = rotXmin; j <= rotXmax; j++)
		{
			for(uint32_t l = 0; l < pixelFeats.size[2]; l++)
			{
				rotPixelFeatsIntegralImg[i][j][l] = rotPixelFeatsIntegralImg[i - 1][j][l] + rowSum[l] + rotPixelFeats[i][j][l];
				rowSum[l] += rotPixelFeats[i][j][l];
			}
			rotFeatValidityIntegralImg[i][j] = rotFeatValidityIntegralImg[i - 1][j] + rowVSum + rotFeatValidity[i][j];
			rowVSum += rotFeatValidity[i][j];
		}
	}
#endif
}
