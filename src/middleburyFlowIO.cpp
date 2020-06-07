/*
 * middleburyFlowIO.h: read/write Middlebury flow evaluation .flo format
 *
 * Evan Herbst
 * 4 / 16 / 08
 */

#include <fstream>
#include <stdexcept>
#include "rgbd_util/assert.h"
#include "optical_flow_utils/middleburyFlowIO.h"
using namespace std;

// first four bytes, should be the same in little endian
#define TAG_FLOAT 202021.25  // check for this when READING the file
#define TAG_STRING "PIEH"    // use this when WRITING the file

/*
 * the .flo format is all little-endian; for now I just assume the OS is too
 */
cv::Mat_<cv::Vec2f> readFlow(const fs::path& filepath)
{
	ifstream infile(filepath.string().c_str(), ifstream::binary);
	ASSERT_ALWAYS(infile);
	float tag; //file format ID
	int32_t w, h;
	/*
	 * don't use >>; it might (don't know when (not)) skip whitespace even in binary mode
	 */
	if(!infile.read((char*)&tag, 4) || !infile.read((char*)&w, 4) || !infile.read((char*)&h, 4)) throw std::runtime_error("can't read metadata for flow image");
	if(tag != TAG_FLOAT) throw std::invalid_argument("magic number doesn't match .flo format");
	cv::Mat_<cv::Vec2f> img(h, w);
	float val[2];
	for(unsigned int i = 0; i < (unsigned int)h; i++)
		for(unsigned int j = 0; j < (unsigned int)w; j++)
		{
			if(!infile.read((char*)&val[0], 4) || !infile.read((char*)&val[1], 4)) throw std::runtime_error("can't read pixel values for flow image");
			cv::Vec2f& pix = img(i, j);
			pix[0] = val[0];
			pix[1] = val[1];
		}
	return img;
}

/*
 * the .flo format is all little-endian; for now I just assume the OS is too
 */
void writeFlow(const cv::Mat_<cv::Vec2f>& flowImg, const fs::path& filepath)
{
	ofstream outfile(filepath.string().c_str(), ofstream::binary);
	ASSERT_ALWAYS(outfile);
	const int32_t w = flowImg.cols, h = flowImg.rows;
	if(!outfile.write((char*)TAG_STRING, 4) || !outfile.write((char*)&w, 4) || !outfile.write((char*)&h, 4)) //to be later interpreted as float, int, int
		throw std::runtime_error("can't write metadata for flow image");
	for(unsigned int i = 0; i < (unsigned int)flowImg.rows; i++)
		for(unsigned int j = 0; j < (unsigned int)flowImg.cols; j++)
		{
			const cv::Vec2f pix = flowImg(i, j);
			if(!outfile.write((char*)&pix[0], 4) || !outfile.write((char*)&pix[1], 4)) throw std::runtime_error("can't write pixel values for flow image");
		}
}
