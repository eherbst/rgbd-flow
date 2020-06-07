/*
 * sceneFlowIO: read/write Evan's .flo3 format (similar to Middlebury .flo format)
 *
 * Evan Herbst
 * 8 / 31 / 12
 */

#ifndef EX_SCENE_FLOW_IO_H
#define EX_SCENE_FLOW_IO_H

#include <boost/filesystem/path.hpp>
#include <opencv2/core/core.hpp>
namespace fs = boost::filesystem;

/*
 * read .flo3 format
 */
cv::Mat_<cv::Vec3f> readSceneFlow(const fs::path& filepath);
/*
 * write .flo3 format
 */
void writeSceneFlow(const cv::Mat_<cv::Vec3f>& flow, const fs::path& filepath);

#endif //header
