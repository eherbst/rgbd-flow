#ifndef _GaussianPyramid_h
#define _GaussianPyramid_h

#include "Image.h"

class GaussianPyramid
{
private:
	DImage* ImPyramid;
	int nLevels;
public:
	GaussianPyramid(void);
	~GaussianPyramid(void);
	void ConstructPyramid(const DImage& image,double ratio=0.8,int minWidth=30);
	int numLevels() const {return nLevels;};
	DImage& image(int index) {return ImPyramid[index];};
};

#endif
