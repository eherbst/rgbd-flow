/*
 * rgbdFrameUtils
 *
 * Evan Herbst
 * 10 / 12 / 12
 */

#ifndef EX_RGBD_FRAME_UTILS_H
#define EX_RGBD_FRAME_UTILS_H

#include <opencv2/core/core.hpp>

/*
 * halve the width and height
 */
void downsizeFrame(cv::Mat& img, cv::Mat_<float>& depth);

#endif //header
