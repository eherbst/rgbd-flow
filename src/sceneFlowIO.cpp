/*
 * sceneFlowIO: read/write Evan's .flo3 format (similar to Middlebury .flo format)
 *
 * Evan Herbst
 * 8 / 31 / 12
 */

#include <cassert>
#include <array>
#include <fstream>
#include <stdexcept>
#include "rgbd_util/assert.h"
#include "optical_flow_utils/sceneFlowIO.h"
using std::ifstream;
using std::ofstream;

//some random bytes for making sure our format isn't easily confused
const std::array<uint8_t, 8> MAGIC_NUMBER = {{237, 159, 43, 26, 202, 94, 218, 68}};

/*
 * read .flo3 format
 */
cv::Mat_<cv::Vec3f> readSceneFlow(const fs::path& filepath)
{
	ifstream infile(filepath.string().c_str(), ifstream::binary);
	ASSERT_ALWAYS(infile);
	std::array<uint8_t, 8> magicNum;
	int32_t w, h;
	if(!infile.read((char*)magicNum.data(), magicNum.size()) || !infile.read((char*)&w, 4) || !infile.read((char*)&h, 4)) throw std::runtime_error("can't read metadata for flow image");
	if(magicNum != MAGIC_NUMBER) throw std::invalid_argument("magic number doesn't match .flo3 format");
	cv::Mat_<cv::Vec3f> img(h, w);
	float val[3];
	for(unsigned int i = 0; i < (unsigned int)h; i++)
		for(unsigned int j = 0; j < (unsigned int)w; j++)
		{
			if(!infile.read((char*)&val[0], 4) || !infile.read((char*)&val[1], 4) || !infile.read((char*)&val[2], 4)) throw std::runtime_error("can't read pixel values for flow image");
			cv::Vec3f& pix = img(i, j);
			pix[0] = val[0];
			pix[1] = val[1];
			pix[2] = val[2];
		}
	return img;
}

/*
 * write .flo3 format
 */
void writeSceneFlow(const cv::Mat_<cv::Vec3f>& flow, const fs::path& filepath)
{
	ofstream outfile(filepath.string().c_str(), ofstream::binary);
	ASSERT_ALWAYS(outfile);
	const int32_t w = flow.cols, h = flow.rows;
	if(!outfile.write((char*)MAGIC_NUMBER.data(), MAGIC_NUMBER.size()) || !outfile.write((char*)&w, 4) || !outfile.write((char*)&h, 4))
		throw std::runtime_error("can't write metadata for flow image");
	for(unsigned int i = 0; i < (unsigned int)flow.rows; i++)
		for(unsigned int j = 0; j < (unsigned int)flow.cols; j++)
		{
			const cv::Vec3f pix = flow(i, j);
			if(!outfile.write((char*)&pix[0], 4) || !outfile.write((char*)&pix[1], 4) || !outfile.write((char*)&pix[2], 4)) throw std::runtime_error("can't write pixel values for flow image");
		}
}
