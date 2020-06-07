/*
 * opticalFlowUtils: non-main functions, separated out so CUDA can see these but not main functions using c++0x features
 *
 * Evan Herbst
 * 10 / 29 / 12
 */

#include "rgbd_util/timer.h"
#include "celiu_flow/OpticalFlow.h"
#include "celiu_flow/opticalFlowUtils.h"

namespace scene_flow
{

//--------------------------------------------------------------------------------------------------------
//  function to compute dx, dy and dt for motion estimation
//--------------------------------------------------------------------------------------------------------
void getDxs(DImage &imdx, DImage &imdy, DImage &imdt, const DImage &im1, const DImage &im2)
{
	static const double gfilter[5]={0.02,0.11,0.74,0.11,0.02};
	// Im1 and Im2 are the smoothed version of im1 and im2
	rgbd::timer t;
#if 1 //testing 20130304; verdict is this works fine at least on imgs of people moving
#define Im1 im1
#define Im2 im2
#else
	DImage Im1,Im2;
	im1.imfilter_hv(Im1,gfilter,2,gfilter,2);
	im2.imfilter_hv(Im2,gfilter,2,gfilter,2);
	t.stop("filter");
#endif
	const bool useAdvancedFilters = true; //was true; false causes numerical problems
	if(true) //EVH: not using this causes numerical problems
	{
		DImage Im;
		Im.copyData(Im1);
		Im.Multiplywith(0.4);
		Im.Add(Im2,0.6);
		Im.dx(imdx,useAdvancedFilters);
		Im.dy(imdy,useAdvancedFilters);
	}
	else
	{
		Im2.dx(imdx,useAdvancedFilters);
		Im2.dy(imdy,useAdvancedFilters);
	}
	imdt.Subtract(Im2,Im1);

	imdx.setDerivative();
	imdy.setDerivative();
	imdt.setDerivative();

#undef Im1
#undef Im2
}

/***************************************************************************************************************************************************************************************/

//--------------------------------------------------------------------------------------------------------
// function to warp image based on the flow field
//--------------------------------------------------------------------------------------------------------
void warpFL(DImage &warpIm2, const DImage &Im1, const DImage &Im2, const DImage &vx, const DImage &vy)
{
	if(warpIm2.matchDimension(Im2)==false)
		warpIm2.allocate(Im2.width(),Im2.height(),Im2.nchannels());
	ImageProcessing::warpImage(warpIm2.data(),Im1.data(),Im2.data(),vx.data(),vy.data(),Im2.width(),Im2.height(),Im2.nchannels());
}

void warpFL(DImage &warpIm2, const DImage &Im1, const DImage &Im2, const DImage &Flow)
{
	if(warpIm2.matchDimension(Im2)==false)
		warpIm2.allocate(Im2.width(),Im2.height(),Im2.nchannels());
	ImageProcessing::warpImageFlow(warpIm2.data(),Im1.data(),Im2.data(),Flow.data(),Im2.width(),Im2.height(),Im2.nchannels());
}

void warpBy3DFlow(const DImage& Im1, const DImage& dm1, const DImage& Im2, const DImage& dm2, DImage& warpIm2, DImage& warpDm2, const DImage& vx, const DImage& vy, const DImage& vz)
{
	if(!warpIm2.matchDimension(Im2)) warpIm2.allocate(Im2.width(),Im2.height(),Im2.nchannels());
	if(!warpDm2.matchDimension(dm2)) warpDm2.allocate(dm2.width(), dm2.height(), dm2.nchannels());

	if(OpticalFlow::interpolation == OpticalFlow::Bilinear)
	{
		ImageProcessing::warpImage(warpIm2.data(),Im1.data(),Im2.data(),vx.data(),vy.data(),Im2.width(),Im2.height(),Im2.nchannels());

		ImageProcessing::warpImage(warpDm2.data(), dm1.data(), dm2.data(), vx.data(), vy.data(), dm2.width(), dm2.height(), dm2.nchannels());
		for(uint32_t i = 0, l = 0; i < dm1.height(); i++)
			for(uint32_t j = 0; j < dm1.width(); j++, l++)
				warpDm2.data()[l] -= vz.data()[l];
	}
	else
	{
		Im2.warpImageBicubicRef(Im1,warpIm2,vx,vy);

		dm2.warpImageBicubicRef(dm1, warpDm2, vx, vy);
		for(uint32_t i = 0, l = 0; i < dm1.height(); i++)
			for(uint32_t j = 0; j < dm1.width(); j++, l++)
				warpDm2.data()[l] -= vz.data()[l];
	}
}

/***************************************************************************************************************************************************************************************/

void estGaussianMixture(const DImage& Im1,const DImage& Im2,GaussianMixture& para,double prior)
{
	int nIterations = 3, nChannels = Im1.nchannels();
	DImage weight1(Im1),weight2(Im1);
	std::vector<double> total1(nChannels), total2(nChannels);
	for(int count = 0; count<nIterations; count++)
	{
		double temp;
		std::fill(total1.begin(), total1.end(), 0);
		std::fill(total2.begin(), total2.end(), 0);

		// E step
		for(int i = 0;i<weight1.npixels();i++)
			for(int k=0;k<nChannels;k++)
			{
				int offset = i*weight1.nchannels()+k;
				temp = Im1[offset]-Im2[offset];
				temp *= temp;
				weight1[offset] = para.Gaussian(temp,0,k)*para.alpha[k];
				weight2[offset] = para.Gaussian(temp,1,k)*(1-para.alpha[k]);
				temp = weight1[offset]+weight2[offset];
				weight1[offset]/=temp;
				weight2[offset]/=temp;
				total1[k] += weight1[offset];
				total2[k] += weight2[offset];
			}

		// M step
		para.reset();


		for(int i = 0;i<weight1.npixels();i++)
			for(int k =0;k<nChannels;k++)
			{
				int offset = i*weight1.nchannels()+k;
				temp = Im1[offset]-Im2[offset];
				temp *= temp;
				para.sigma[k]+= weight1[offset]*temp;
				para.beta[k] += weight2[offset]*temp;
			}

		for(int k =0;k<nChannels;k++)
		{
			para.alpha[k] = total1[k]/(total1[k]+total2[k])*(1-prior)+0.95*prior; // regularize alpha
			para.sigma[k] = sqrt(para.sigma[k]/total1[k]);
			para.beta[k]   = sqrt(para.beta[k]/total2[k])*(1-prior)+0.3*prior; // regularize beta
		}
		para.square();
		count = count;
	}
}

void estLaplacianNoise(const DImage& Im1,const DImage& Im2,Vector<double>& para)
{
	int nChannels = Im1.nchannels();
	if(para.dim()!=nChannels)
		para.allocate(nChannels);
	else
		para.reset();
	double temp;
	Vector<double> total(nChannels);
	for(int k = 0;k<nChannels;k++)
		total[k] = 0;

	for(int i =0;i<Im1.npixels();i++)
		for(int k = 0;k<nChannels;k++)
		{
			int offset = i*nChannels+k;
			temp= fabs(Im1.data()[offset]-Im2.data()[offset]);
			if(temp>0 && temp<1000000)
			{
				para[k] += temp;
				total[k]++;
			}
		}
	for(int k = 0;k<nChannels;k++)
	{
		if(total[k]==0)
		{
			//cout<<"All the pixels are invalid in estimation Laplacian noise!!!"<<endl;
			//cout<<"Something severely wrong happened!!!"<<endl;
			para[k] = 0.001;
		}
		else
			para[k]/=total[k];
	}
}

/*
 * output <- negative Laplacian of input, multiplied by weight at each pixel
 */
void weightedLaplacian(DImage &output, const DImage &input, const DImage& weight)
{
	if(output.matchDimension(input)==false)
		output.allocate(input);
	output.reset();

	if(input.matchDimension(weight)==false)
	{
		cout<<"Error in image dimension matching OpticalFlow::Laplacian()!"<<endl;
		return;
	}

	const double *inputData=input.data(),*weightData=weight.data();
	int width=input.width(),height=input.height();
	DImage foo(width,height);
	double *fooData=foo.data(),*outputData=output.data();


	// horizontal filtering
	for(int i=0;i<height;i++)
		for(int j=0;j<width-1;j++)
		{
			int offset=i*width+j;
			fooData[offset]=(inputData[offset+1]-inputData[offset])*weightData[offset];
		}
	for(int i=0;i<height;i++)
		for(int j=0;j<width;j++)
		{
			int offset=i*width+j;
			if(j<width-1)
				outputData[offset]-=fooData[offset];
			if(j>0)
				outputData[offset]+=fooData[offset-1];
		}
	foo.reset();
	// vertical filtering
	for(int i=0;i<height-1;i++)
		for(int j=0;j<width;j++)
		{
			int offset=i*width+j;
			fooData[offset]=(inputData[offset+width]-inputData[offset])*weightData[offset];
		}
	for(int i=0;i<height;i++)
		for(int j=0;j<width;j++)
		{
			int offset=i*width+j;
			if(i<height-1)
				outputData[offset]-=fooData[offset];
			if(i>0)
				outputData[offset]+=fooData[offset-width];
		}
}

/*
 * normWeight gets multiplied into numerator and denominator
 *
 * normWeights: 2-channel, for right and down pixel nbrs
 */
void Laplacian(DImage &output, const DImage &input, const DImage& normWeights)
{
	if(output.matchDimension(input)==false)
		output.allocate(input);
	output.reset();

	const double *inputData=input.data();
	int width=input.width(),height=input.height();
	DImage foo(width,height);
	double *fooData=foo.data(),*outputData=output.data();


	// horizontal filtering
	for(int i=0;i<height;i++)
		for(int j=0;j<width-1;j++)
		{
			int offset=i*width+j;
			fooData[offset]=(inputData[offset+1]-inputData[offset]) * normWeights(i, j, 0);
		}
	for(int i=0;i<height;i++)
		for(int j=0;j<width;j++)
		{
			int offset=i*width+j;
			if(j<width-1) outputData[offset]-=fooData[offset];
			if(j>0) outputData[offset]+=fooData[offset-1];
		}
	foo.reset();
	// vertical filtering
	for(int i=0;i<height-1;i++)
		for(int j=0;j<width;j++)
		{
			int offset=i*width+j;
			fooData[offset]=(inputData[offset+width]-inputData[offset]) * normWeights(i, j, 1);
		}
	for(int i=0;i<height;i++)
		for(int j=0;j<width;j++)
		{
			int offset=i*width+j;
			if(i<height-1) outputData[offset]-=fooData[offset];
			if(i>0) outputData[offset]+=fooData[offset-width];
		}

	/*
	 * normalize such that if normWeights are all 1, we always divide by 1 here
	 */
	for(int i=0;i<height;i++)
		for(int j=0;j<width;j++)
		{
			int offset=i*width+j;
			double norm = 0;
			int count = 0;
			if(j<width-1) {norm += normWeights(i, j, 0); count++;}
			if(j > 0) {norm += normWeights(i, j - 1, 0); count++;}
			if(i < height - 1) {norm += normWeights(i, j, 1); count++;}
			if(i > 0) {norm += normWeights(i - 1, j, 1); count++;}
			if(fabs(norm) < 1e-5/* TODO ? */) outputData[offset] = 0;
			else outputData[offset] /= (norm / count);
		}
}

/*
 * nonnormWeight gets multiplied in; normWeight gets multiplied into numerator and denominator
 */
void Laplacian(DImage &output, const DImage &input, const DImage& nonnormWeights, const DImage& normWeights)
{
	if(output.matchDimension(input)==false)
		output.allocate(input);
	output.reset();

	if(input.matchDimension(nonnormWeights)==false)
	{
		cout<<"Error in image dimension matching OpticalFlow::Laplacian()!"<<endl;
		return;
	}

	const double *inputData=input.data(),*nonnormWeightData=nonnormWeights.data();
	int width=input.width(),height=input.height();
	DImage foo(width,height);
	double *fooData=foo.data(),*outputData=output.data();


	// horizontal filtering
	for(int i=0;i<height;i++)
		for(int j=0;j<width-1;j++)
		{
			int offset=i*width+j;
			fooData[offset]=(inputData[offset+1]-inputData[offset]) * nonnormWeightData[offset] * normWeights(i, j, 0);
		}
	for(int i=0;i<height;i++)
		for(int j=0;j<width;j++)
		{
			int offset=i*width+j;
			if(j<width-1) outputData[offset]-=fooData[offset];
			if(j>0) outputData[offset]+=fooData[offset-1];
		}
	foo.reset();
	// vertical filtering
	for(int i=0;i<height-1;i++)
		for(int j=0;j<width;j++)
		{
			int offset=i*width+j;
			fooData[offset]=(inputData[offset+width]-inputData[offset]) * nonnormWeightData[offset] * normWeights(i, j, 1);
		}
	for(int i=0;i<height;i++)
		for(int j=0;j<width;j++)
		{
			int offset=i*width+j;
			if(i<height-1) outputData[offset]-=fooData[offset];
			if(i>0) outputData[offset]+=fooData[offset-width];
		}

	/*
	 * normalize such that if normWeights are all 1, we always divide by 1 here
	 */
	for(int i=0;i<height;i++)
		for(int j=0;j<width;j++)
		{
			int offset=i*width+j;
			double norm = 0;
			int count = 0;
			if(j<width-1) {norm += normWeights(i, j, 0); count++;}
			if(j > 0) {norm += normWeights(i, j - 1, 0); count++;}
			if(i < height - 1) {norm += normWeights(i, j, 1); count++;}
			if(i > 0) {norm += normWeights(i - 1, j, 1); count++;}
			if(fabs(norm) < 1e-5/* TODO ? */) outputData[offset] = 0;
			else outputData[offset] /= (norm / count);
		}
}

/*
 * second partials (not negated)
 */
void hessian(cv::Mat& output, const DImage &input, const DImage& weight)
{
	output = cv::Mat(input.height(), input.width(), cv::DataType<cv::Vec4f>::type);
	output = cv::Scalar(0, 0, 0, 0); //fill the mtx
	const double *inputData=input.data(),*weightData=weight.data();
	int width=input.width(),height=input.height();

	cv::Mat dx(input.height(), input.width(), cv::DataType<float>::type), dy(input.height(), input.width(), cv::DataType<float>::type);
	for(int i=0;i<height;i++)
		for(int j=0;j<width-1;j++)
		{
			int offset=i*width+j;
			dx.at<float>(i, j)=(inputData[offset+1]-inputData[offset])*weightData[offset];
		}
	for(int i=0;i<height-1;i++)
		for(int j=0;j<width;j++)
		{
			int offset=i*width+j;
			dy.at<float>(i, j)=(inputData[offset+width]-inputData[offset])*weightData[offset];
		}

	for(int i=0;i<height;i++)
		for(int j=0;j<width;j++)
		{
			cv::Vec4f& p = output.at<cv::Vec4f>(i, j);

			if(j < width - 1) p[0] += dx.at<float>(i, j);
			if(j > 0) p[0] -= dx.at<float>(i, j - 1);

			if(i < height - 1) p[3] += dy.at<float>(i, j);
			if(i > 0) p[3] -= dy.at<float>(i - 1, j);

			if(i < height - 1 && j > 0) p[1] = dy.at<float>(i, j) - dy.at<float>(i, j - 1);

			if(j < width - 1 && i > 0) p[2] = dx.at<float>(i, j) - dx.at<float>(i - 1, j);
		}
}

} //namespace
