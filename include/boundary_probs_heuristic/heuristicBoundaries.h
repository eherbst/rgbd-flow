/*
 * heuristicBoundaries: use single-frame features to estimate boundary points in 2-d
 *
 * Evan Herbst
 * 6 / 26 / 12
 */

#ifndef EX_HEURISTIC_RGBD_FRAME_BOUNDARIES_H
#define EX_HEURISTIC_RGBD_FRAME_BOUNDARIES_H

#include <boost/multi_array.hpp>
#include <Eigen/Core>
#include <opencv2/core/core.hpp>
#include "rgbd_util/CameraParams.h"

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
boost::multi_array<float, 3> computeHeuristicPb(const rgbd::CameraParams& camParams, const cv::Mat& img, const cv::Mat_<float>& depth);

#endif //header
