/*
 * middleburyFlowIO.h: read/write Middlebury flow evaluation .flo format
 *
 * Evan Herbst
 * 4 / 16 / 08
 */

#ifndef EX_MIDDLEBURY_FLOW_IO_H
#define EX_MIDDLEBURY_FLOW_IO_H

#include <string>
#include <boost/filesystem/path.hpp>
#include <opencv2/core/core.hpp>
namespace fs = boost::filesystem;

/*
 * the .flo format is all little-endian; for now I just assume the OS is too
 */
cv::Mat_<cv::Vec2f> readFlow(const fs::path& filepath);

/*
 * the .flo format is all little-endian; for now I just assume the OS is too
 */
void writeFlow(const cv::Mat_<cv::Vec2f>& flowImg, const fs::path& filepath);

#endif //EX_MIDDLEBURY_FLOW_IO_H
