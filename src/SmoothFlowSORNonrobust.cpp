/*
 * added by EVH: nonrobust regularization on data and smoothness terms
 *
 * 20120727: alpha=.003 works pretty well for twoObjsMove1
 */
void OpticalFlow::SmoothFlowSORNonrobust(const DImage &Im1, const DImage &Im2, DImage &warpIm2, DImage &u, DImage &v,
   double alpha, int nOuterFPIterations, int nInnerFPIterations, int nSORIterations)
{
	DImage imdx,imdy,imdt;
	int imWidth,imHeight,nChannels,nPixels;
	imWidth=Im1.width();
	imHeight=Im1.height();
	nChannels=Im1.nchannels();
	nPixels=imWidth*imHeight;

	DImage du(imWidth,imHeight),dv(imWidth,imHeight); //delta calculated at each iter
	DImage uu(imWidth,imHeight),vv(imWidth,imHeight); //uu = u + du
	DImage ux(imWidth,imHeight),uy(imWidth,imHeight); //ux = d/dx(uu)
	DImage vx(imWidth,imHeight),vy(imWidth,imHeight); //vx = d/dx(vv)

	DImage imdxy,imdx2,imdy2,imdtdx,imdtdy;
	DImage ImDxy,ImDx2,ImDy2,ImDtDx,ImDtDy;
	DImage foo1,foo2; //Laplacians of the flow field

	//--------------------------------------------------------------------------
	// the outer fixed point iteration
	//--------------------------------------------------------------------------
	for(int count=0;count<nOuterFPIterations;count++)
	{
		// compute the gradient
		getDxs(imdx,imdy,imdt,Im1,warpIm2);

		// set the derivative of the flow field to be zero
		du.reset();
		dv.reset();

		//--------------------------------------------------------------------------
		// the inner fixed point iteration
		//--------------------------------------------------------------------------
		for(int hh=0;hh<nInnerFPIterations;hh++)
		{
			// compute the derivatives of the current flow field
			if(hh==0)
			{
				uu.copyData(u);
				vv.copyData(v);
			}
			else
			{
				uu.Add(u,du); //uu = u + du
				vv.Add(v,dv);
			}
			uu.dx(ux); //ux = d/dx(uu)
			uu.dy(uy);
			vv.dx(vx);
			vv.dy(vy);

			const double *uxData,*uyData,*vxData,*vyData;
			uxData=ux.data();
			uyData=uy.data();
			vxData=vx.data();
			vyData=vy.data();
			const double *imdxData,*imdyData,*imdtData;
			const double *duData,*dvData;
			imdxData=imdx.data();
			imdyData=imdy.data();
			imdtData=imdt.data();
			duData=du.data();
			dvData=dv.data();

			// prepare the components of the large linear system
			ImDxy.Multiply(imdx,imdy); //z = w .* x .* y
			ImDx2.Multiply(imdx,imdx);
			ImDy2.Multiply(imdy,imdy);
			ImDtDx.Multiply(imdx,imdt);
			ImDtDy.Multiply(imdy,imdt);

			if(nChannels>1)
			{
				ImDxy.collapse(imdxy); //result is single-channel holding the avg of all source channels
				ImDx2.collapse(imdx2);
				ImDy2.collapse(imdy2);
				ImDtDx.collapse(imdtdx);
				ImDtDy.collapse(imdtdy);
			}
			else
			{
				imdxy.copyData(ImDxy);
				imdx2.copyData(ImDx2);
				imdy2.copyData(ImDy2);
				imdtdx.copyData(ImDtDx);
				imdtdy.copyData(ImDtDy);
			}

			// here we start SOR

			//SOR parameter
			double omega = 1.8;

			du.reset();
			dv.reset();

			for(int k = 0; k<nSORIterations; k++)
				for(int i = 0; i<imHeight; i++)
					for(int j = 0; j<imWidth; j++)
					{
						int offset = i * imWidth+j;

						double sumU = 0, sumDUNbrs = 0, sumV = 0, sumDVNbrs = 0, nbrCount = 0;

						if(j>0)
						{
							sumU += uu.data()[offset - 1] - uu.data()[offset];
							sumV += vv.data()[offset - 1] - vv.data()[offset];
							sumDUNbrs  += du.data()[offset-1];
							sumDVNbrs  += dv.data()[offset-1];
									 nbrCount   += 1;
						}
						if(j<imWidth-1)
						{
							sumU += uu.data()[offset + 1] - uu.data()[offset];
							sumV += vv.data()[offset + 1] - vv.data()[offset];
							sumDUNbrs += du.data()[offset+1];
							sumDVNbrs += dv.data()[offset+1];
									 nbrCount   += 1;
						}
						if(i>0)
						{
							sumU += uu.data()[offset - imWidth] - uu.data()[offset];
							sumV += vv.data()[offset - imWidth] - vv.data()[offset];
							sumDUNbrs += du.data()[offset-imWidth];
							sumDVNbrs += dv.data()[offset-imWidth];
									 nbrCount   += 1;
						}
						if(i<imHeight-1)
						{
							sumU += uu.data()[offset + imWidth] - uu.data()[offset];
							sumV += vv.data()[offset + imWidth] - vv.data()[offset];
							sumDUNbrs  += du.data()[offset+imWidth];
							sumDVNbrs  += dv.data()[offset+imWidth];
									 nbrCount   += 1;
						}

						du.data()[offset] = (1-omega)*du.data()[offset] + omega * (-imdtdx.data()[offset] - imdxy.data()[offset]*dv.data()[offset] + alpha * sumU + alpha * sumDUNbrs) / (imdx2.data()[offset] + alpha * nbrCount);
						dv.data()[offset] = (1-omega)*dv.data()[offset] + omega * (-imdtdy.data()[offset] - imdxy.data()[offset]*du.data()[offset] + alpha * sumV + alpha * sumDVNbrs) / (imdy2.data()[offset] + alpha * nbrCount);
					}
		}
		u.Add(du);
		v.Add(dv);
		if(interpolation == Bilinear)
			warpFL(warpIm2,Im1,Im2,u,v);
		else
		{
			Im2.warpImageBicubicRef(Im1,warpIm2,u,v);
			warpIm2.clampToNormalRange();
		}

		//Im2.warpImageBicubicRef(Im1,warpIm2,BicubicCoeff,u,v);

		// estimate noise level
		switch(noiseModel)
		{
		case GMixture:
			estGaussianMixture(Im1,warpIm2,GMPara);
			break;
		case Lap:
			estLaplacianNoise(Im1,warpIm2,LapPara);
		}
	}
}
