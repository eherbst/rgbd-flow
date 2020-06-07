/*
 * compute 2-d flow over the pyramid
 */
void OpticalFlow::Coarse2FineFlow(DImage &vx, DImage &vy, DImage &warpI2,const DImage &Im1, const DImage &Im2, const opticalFlowParams& params)
{
	double ratio = params.pyramidRatio;

	GaussianPyramid GPyramid1, GPyramid2;
	GaussianPyramid gPyramidDZDX, gPyramidDZDY;
	GaussianPyramid gPyramidDepth;
	GaussianPyramid gPyramidDepth1, gPyramidDepth2;
	GaussianPyramid gPyramidDepthConstancyWts;
	if(IsDisplay)
		cout<<"Constructing pyramid...";
	GPyramid1.ConstructPyramid(Im1,ratio,params.minWidth);
	GPyramid2.ConstructPyramid(Im2,ratio,params.minWidth);
	if(params.dzdx.rows > 0)
	{
		DImage dzdx(Im1.width(), Im1.height(), 1), dzdy(Im1.width(), Im1.height(), 1);
		double* dx = dzdx.data(), *dy = dzdy.data();
		for(int32_t i = 0; i < Im1.height(); i++)
			for(int32_t j = 0; j < Im1.width(); j++)
			{
				*dx++ = params.dzdx.at<float>(i, j);
				*dy++ = params.dzdy.at<float>(i, j);
			}
		gPyramidDZDX.ConstructPyramid(dzdx,ratio,params.minWidth);
		gPyramidDZDY.ConstructPyramid(dzdy,ratio,params.minWidth);
	}
	if(params.depthMap.rows > 0)
	{
		DImage depth(Im1.width(), Im1.height(), 1);
		double* d = depth.data();
		for(int32_t i = 0; i < Im1.height(); i++)
			for(int32_t j = 0; j < Im1.width(); j++)
			{
				*d++ = params.depthMap.at<float>(i, j);
			}

		/*
		 * fill in invalid depths
		 */
		recursiveMedianFilter(depth);

		gPyramidDepth.ConstructPyramid(depth,ratio,params.minWidth);
	}
	if(params.depthMap1.rows > 0)
	{
		DImage depth1(Im1.width(), Im1.height(), 1), depth2(Im1.width(), Im1.height(), 1);
		double* d1 = depth1.data(), *d2 = depth2.data();
		for(int32_t i = 0; i < Im1.height(); i++)
			for(int32_t j = 0; j < Im1.width(); j++)
			{
				*d1++ = params.depthMap1.at<float>(i, j);
				*d2++ = params.depthMap2.at<float>(i, j);
			}

#if 1
		//fill in invalid depths
		recursiveMedianFilter(depth1);
		recursiveMedianFilter(depth2);
#endif
		gPyramidDepth1.ConstructPyramid(depth1,ratio,params.minWidth);
		gPyramidDepth2.ConstructPyramid(depth2,ratio,params.minWidth);
	}
	if(params.depthConstancyWeights.num_elements() > 0)
	{
		DImage wts(Im1.width(), Im1.height(), 1);
		double* d = wts.data();
		for(int32_t i = 0; i < Im1.height(); i++)
			for(int32_t j = 0; j < Im1.width(); j++)
				*d++ = params.depthConstancyWeights[i][j];
		gPyramidDepthConstancyWts.ConstructPyramid(wts, ratio, params.minWidth);
	}
	if(IsDisplay)
		cout<<"done!"<<endl;

	// now iterate from the top level to the bottom
	DImage Image1,Image2,WarpImage2;

	// initialize noise
	switch(noiseModel){
	case GMixture:
		GMPara.reset(Im1.nchannels()+2);
		break;
	case Lap:
		LapPara.allocate(Im1.nchannels()+2);
		for(int i = 0;i<LapPara.dim();i++)
			LapPara[i] = 0.02;
		break;
	}

	for(int k=GPyramid1.numLevels()-1;k>=0;k--)
	{
		if(IsDisplay)
			cout<<"Pyramid level "<<k;
		int width=GPyramid1.image(k).width();
		int height=GPyramid1.image(k).height();
		if(params.depthMap1.rows > 0)
		{
			im2featureWithDepth(Image1,GPyramid1.image(k), gPyramidDepth1.image(k));
			im2featureWithDepth(Image2,GPyramid2.image(k), gPyramidDepth2.image(k));
		}
		else
		{
			im2feature(Image1,GPyramid1.image(k), constancyType::RGB_WITH_GRADIENT);
			im2feature(Image2,GPyramid2.image(k), constancyType::RGB_WITH_GRADIENT);
		}

		if(k==GPyramid1.numLevels()-1) // if at the smallest scale
		{
			vx.allocate(width,height);
			vy.allocate(width,height);
			WarpImage2.copyData(Image2);
		}
		else
		{

			vx.imresize(width,height);
			vy.imresize(width,height);
			vx.Multiplywith(1/ratio);
			vy.Multiplywith(1/ratio);
			if(interpolation == Bilinear)
				warpFL(WarpImage2,Image1,Image2,vx,vy);
			else
				Image2.warpImageBicubicRef(Image1,WarpImage2,vx,vy);
		}

		switch(params.variant)
		{
			case opticalFlowVariant::CELIU_NONROBUST:
			{
				ASSERT_ALWAYS(params.alpha >= 0);
				SmoothFlowSORNonrobust(Image1,Image2,WarpImage2,vx,vy, params.alpha,params.nOuterFPIterations+k,params.nInnerFPIterations,params.nSORIterations+k*3);
				break;
			}
			case opticalFlowVariant::CELIU:
			{
				ASSERT_ALWAYS(params.alpha >= 0);
				SmoothFlowSOR(Image1,Image2,WarpImage2,vx,vy, params.alpha,params.nOuterFPIterations+k,params.nInnerFPIterations,params.nSORIterations+k*3);
				break;
			}
			default: ASSERT_ALWAYS(false && "unhandled variant");
		}

		if(IsDisplay)
			cout<<endl;
	}
	Im2.warpImageBicubicRef(Im1,warpI2,vx,vy);
	warpI2.clampToNormalRange();
}
