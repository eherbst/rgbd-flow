#include <cmath>
#include "celiu_flow/ImageProcessing.h"
#include "celiu_flow/GaussianPyramid.h"

GaussianPyramid::GaussianPyramid(void)
{
	ImPyramid=NULL;
}

GaussianPyramid::~GaussianPyramid(void)
{
	if(ImPyramid!=NULL)
		delete []ImPyramid;
}

//---------------------------------------------------------------------------------------
// function to construct the pyramid
// this is the fast way
//---------------------------------------------------------------------------------------
void GaussianPyramid::ConstructPyramid(const DImage &image, double ratio, int minWidth)
{
	// the ratio cannot be arbitrary numbers
	if(ratio>0.98 || ratio<0.4)
		ratio=0.75;
	// first decide how many levels
	double w = image.width() * ratio;
	nLevels = 1;
	while(w >= minWidth)
	{
		nLevels++;
		w *= ratio;
	}

	if(ImPyramid!=NULL)
		delete []ImPyramid;
	ImPyramid=new DImage[nLevels];
#if 1
	ImPyramid[0].copyData(image);
#else
	//xf suggests smoothing both color and depth a little, at least for synthetic data
	const double sigma = .5;
	image.GaussianSmoothing(ImPyramid[0], sigma, sigma * 3);
#endif
	double baseSigma=(1/ratio-1);
	int n=log(0.25)/log(ratio);
	double nSigma=baseSigma*n;
	for(int i=1;i<nLevels;i++)
	{
		DImage foo;
		if(i<=n)
		{
			double sigma=baseSigma*i;
			image.GaussianSmoothing(foo,sigma,sigma*3);
			foo.imresize(ImPyramid[i],pow(ratio,i));
		}
		else
		{
			ImPyramid[i-n].GaussianSmoothing(foo,nSigma,nSigma*3);
			double rate=(double)pow(ratio,i)*image.width()/foo.width();
			foo.imresize(ImPyramid[i],rate);
		}
	}
}
