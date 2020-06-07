/*
 * depthMapPyramid: downsample depth maps without uncertainty information but accounting for invalid measurements
 *
 * Evan Herbst
 * 12 / 19 / 12
 */

#ifndef EX_FLOW_DEPTH_MAP_PYRAMID_H
#define EX_FLOW_DEPTH_MAP_PYRAMID_H

#include <vector>
#include "Image.h"

/*
 * basically, downsample while avoiding mixing valid and invalid values in a single downsampled pixel
 */
class DepthMapPyramid
{
private:
	std::vector<DImage> ImPyramid;
public:
	DepthMapPyramid() {}
	~DepthMapPyramid() {}

	/*
	 * values <= 0 will be considered invalid
	 */
	void ConstructPyramid(const DImage& image,double ratio,int minWidth);
	int numLevels() const {return ImPyramid.size();};
	DImage& image(int index) {return ImPyramid[index];};
};

#endif //header
