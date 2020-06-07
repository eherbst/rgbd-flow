#include <iostream>
using std::cout;
using std::endl;

/*
 * color constancy, gradient constancy and L1 smoothness
 */
void OpticalFlow::SmoothFlowSOR(const DImage &Im1, const DImage &Im2, DImage &warpIm2, DImage &u, DImage &v,
																    double alpha, int nOuterFPIterations, int nInnerFPIterations, int nSORIterations)
{
	int imWidth,imHeight,nChannels,nPixels;
	imWidth=Im1.width();
	imHeight=Im1.height();
	nChannels=Im1.nchannels();
	nPixels=imWidth*imHeight;

	DImage imdx(imWidth, imHeight, nChannels), imdy(imWidth, imHeight, nChannels), imdt(imWidth, imHeight, nChannels);

	DImage du(imWidth,imHeight),dv(imWidth,imHeight); //delta calculated at each iter
	DImage uu(imWidth,imHeight),vv(imWidth,imHeight); //uu = u + du
	DImage ux(imWidth,imHeight),uy(imWidth,imHeight); //ux = d/dx(uu)
	DImage vx(imWidth,imHeight),vy(imWidth,imHeight); //vx = d/dx(vv)
	DImage Phi_1st(imWidth,imHeight); //1st deriv of smoothness energy
	DImage Psi_1st(imWidth,imHeight,nChannels); //1st deriv of data energy

	DImage imdxy,imdx2,imdy2,imdtdx,imdtdy;
	DImage ImDxy,ImDx2,ImDy2,ImDtDx,ImDtDy;
	DImage foo1,foo2; //Laplacians of the flow field

	const double varepsilon_phi=pow(0.001,2);
	const double varepsilon_psi=pow(0.001,2);

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

			// compute the weight of phi
			Phi_1st.reset(); //1st deriv of smoothness energy
			double* phiData=Phi_1st.data();
			double temp;
			const double *uxData,*uyData,*vxData,*vyData;
			uxData=ux.data();
			uyData=uy.data();
			vxData=vx.data();
			vyData=vy.data();
			for(int i=0;i<nPixels;i++)
			{
				temp=uxData[i]*uxData[i]+uyData[i]*uyData[i]+vxData[i]*vxData[i]+vyData[i]*vyData[i];
				phiData[i] = 0.5/sqrt(temp+varepsilon_phi);
			}

			// compute the nonlinear term of psi
			Psi_1st.reset(); //1st deriv of data energy
			double* psiData=Psi_1st.data();
			const double *imdxData,*imdyData,*imdtData;
			const double *duData,*dvData;
			imdxData=imdx.data();
			imdyData=imdy.data();
			imdtData=imdt.data();
			duData=du.data();
			dvData=dv.data();

			for(int i=0;i<nPixels;i++)
				for(int k=0;k<nChannels;k++)
				{
					int offset=i*nChannels+k;
					temp=imdtData[offset]+imdxData[offset]*duData[i]+imdyData[offset]*dvData[i];
					temp *= temp;
					switch(noiseModel)
					{
					case GMixture:
					{
						double prob1,prob2,prob11,prob22;
						prob1 = GMPara.Gaussian(temp,0,k)*GMPara.alpha[k];
						prob2 = GMPara.Gaussian(temp,1,k)*(1-GMPara.alpha[k]);
						prob11 = prob1/(2*GMPara.sigma_square[k]);
						prob22 = prob2/(2*GMPara.beta_square[k]);
						psiData[offset] = (prob11+prob22)/(prob1+prob2);
						break;
					}
					case Lap:
						if(LapPara[k]<1E-20)
						{
							ASSERT_ALWAYS(false && "bad lap para");
							continue;
						}
						psiData[offset]=1/(2*sqrt(temp+varepsilon_psi));
						break;
					}
				}
			// prepare the components of the large linear system
			ImDxy.Multiply(Psi_1st,imdx,imdy); //z = w .* x .* y
			ImDx2.Multiply(Psi_1st,imdx,imdx);
			ImDy2.Multiply(Psi_1st,imdy,imdy);
			ImDtDx.Multiply(Psi_1st,imdx,imdt);
			ImDtDy.Multiply(Psi_1st,imdy,imdt);

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
			// laplacian filtering of the current flow field, weighted by Phi_1st
		    Laplacian(foo1,u,Phi_1st);
			Laplacian(foo2,v,Phi_1st);

			for(int i=0;i<nPixels;i++)
			{
				imdtdx.data()[i] = -imdtdx.data()[i]-alpha*foo1.data()[i];
				imdtdy.data()[i] = -imdtdy.data()[i]-alpha*foo2.data()[i];
			}

			// here we start SOR

			//SOR parameter
			double omega = 1.8;

			du.reset();
			dv.reset();

			bool converged = false;
			//for(int k = 0; !converged; k++)
			for(int k = 0; k<nSORIterations && !converged; k++)
			{
				double maxDiff = 0;
				for(int i = 0; i<imHeight; i++)
					for(int j = 0; j<imWidth; j++)
					{
						int offset = i * imWidth+j;
						double sigmaU = 0, sigmaV = 0, //terms with other variables (imdtdx is the constant terms) in the linear eqn
							weightSum = 0;
                  double _weight;

						if(j>0)
						{
                            _weight = phiData[offset-1];
                            sigmaU  += _weight*du.data()[offset-1];
                            sigmaV  += _weight*dv.data()[offset-1];
							weightSum   += _weight;
						}
						if(j<imWidth-1)
						{
                            _weight = phiData[offset];
                            sigmaU += _weight*du.data()[offset+1];
                            sigmaV += _weight*dv.data()[offset+1];
							weightSum   += _weight;
						}
						if(i>0)
						{
                            _weight = phiData[offset-imWidth];
                            sigmaU += _weight*du.data()[offset-imWidth];
                            sigmaV += _weight*dv.data()[offset-imWidth];
							weightSum   += _weight;
						}
						if(i<imHeight-1)
						{
                            _weight = phiData[offset];
                            sigmaU  += _weight*du.data()[offset+imWidth];
                            sigmaV  += _weight*dv.data()[offset+imWidth];
							weightSum   += _weight;
						}

						sigmaU *= -alpha;
						sigmaV *= -alpha;
						weightSum *= alpha;

						/*
						 * EVH 20120730: the .05*alpha can be removed without affecting much (I think it's just there to prevent nans); no other term seems to be removable
						 */
						const double prevDU = du.data()[offset], prevDV = dv.data()[offset];

						 // compute du
						sigmaU += imdxy.data()[offset]*dv.data()[offset];
						du.data()[offset] = (1-omega)*du.data()[offset] + omega/(imdx2.data()[offset]/* + alpha*0.05*/ + weightSum)*(imdtdx.data()[offset] - sigmaU);
						// compute dv
						sigmaV += imdxy.data()[offset]*du.data()[offset];
						dv.data()[offset] = (1-omega)*dv.data()[offset] + omega/(imdy2.data()[offset]/* + alpha*0.05*/ + weightSum)*(imdtdy.data()[offset] - sigmaV);

						maxDiff = std::max(maxDiff, std::max(fabs(du.data()[offset] - prevDU), fabs(dv.data()[offset] - prevDV)));
					}

				converged = (maxDiff < 1e-5/* TODO ? */);
				const int x0 = du.width() / 2, y0 = du.height() / 2;
				if(k % 10 == 0) cout << " sor " << k << ' ' << maxDiff << " : flow@" <<x0 << "," << y0 << " = " << (u.data()[y0 * du.width() + x0] + du.data()[y0 * du.width() + x0]) << ' ' << (v.data()[y0 * du.width() + x0] + dv.data()[y0 * du.width() + x0]) << endl;
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
