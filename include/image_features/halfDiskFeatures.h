/*
 * halfDiskFeatures: over 2-d images
 *
 * Evan Herbst
 * 6 / 18 / 12
 */

#ifndef EX_HALF_DISK_IMAGE_FEATURES_H
#define EX_HALF_DISK_IMAGE_FEATURES_H

#include <cstdint>
#include <vector>
#include <boost/multi_array.hpp>
#include <opencv2/core/core.hpp>
#include <Eigen/Core>

/*
 * resample values by rotating in xy by angle; the result arrays will be large enough to contain samples for all values, and won't be indexed starting at 0
 *
 * result arrays will be allocated
 *
 * pre: values has type ValueT
 */
template <typename ValueT>
void rotateValuesInImgSpace(const cv::Mat& values, const float angle, boost::multi_array<ValueT, 3>& rotatedValues, boost::multi_array<bool, 2>& resultValidity);

/*
 * the result array will be allocated
 *
 * pre: D is 2 or 3
 *
 * post: valid indices for integralImg in the x and y dimensions will start one before valid indices for img (and all the values in the first row and column will be 0)
 */
template <typename T, typename T2, const size_t D>
void computeIntegralImage(const boost::multi_array<T, D>& img, boost::multi_array<T2, D>& integralImg);

typedef std::function<float (const Eigen::VectorXf& side1avg, const Eigen::VectorXf& side2avg)> halfDisk2dFeatFunc;
/*
 * pixelFeats: 3-d float img with features at each pixel (all will be treated as valid)
 *
 * return:
 * - if featFunc given, 3-d img with max of featFunc calls over angles at each pixel and scale (third dimension has size 1)
 * - if featFunc not given, 3-d img with max of each feat over angles at each pixel and scale (third dimension has all features at each scale)
 */
cv::Mat computeHalfDiskImgFeatures(const std::vector<int32_t>& radii, const uint32_t numAngles, const cv::Mat& pixelFeats, const halfDisk2dFeatFunc featFunc = halfDisk2dFeatFunc());

#define MULTITHREAD_HALF_DISK_IMG_FEATS
#include "halfDiskFeatures.ipp"

#endif //header
