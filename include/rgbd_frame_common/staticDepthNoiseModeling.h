/*
 * staticDepthNoiseModeling: modeling depth uncertainty from a single frame
 *
 * Evan Herbst
 * 10 / 31 / 12
 */

#ifndef EX_RGBD_STATIC_DEPTH_NOISE_MODELING_H
#define EX_RGBD_STATIC_DEPTH_NOISE_MODELING_H

#include <cstdint>
#include <boost/multi_array.hpp>
#include <opencv2/core/core.hpp>
#include "rgbd_util/CameraParams.h"

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
void computeDepthMapLocalStdevs(const cv::Mat_<float>& depth, const uint32_t nbrhoodHalfwidth, boost::multi_array<float, 2>& stdevs, const uint32_t increaseNearDepthBoundariesMaxDist, const rgbd::CameraParams& camParams, const bool multithread);

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
void computeDepthMapStdevsPointwise(const cv::Mat_<float>& depth, boost::multi_array<float, 2>& stdevs, const uint32_t increaseNearDepthBoundariesMaxDist, const bool multithread);
/*
 * use a non-real value of depth uncertainty at 1 m, to adjust diffing sensitivity
 */
void computeDepthMapStdevsPointwise(const cv::Mat_<float>& depth, boost::multi_array<float, 2>& stdevs, const float sigmaAt1m, const uint32_t increaseNearDepthBoundariesMaxDist, const bool multithread);

#endif //header
