#include <stdint.h>
#include <float.h>
#include <cassert>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <boost/format.hpp>
#include <thrust/device_vector.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>
#include "rgbd_util/timer.h"
#define RGBD_MATHUTILS_DISABLE_CXX0X
#include "rgbd_util/mathUtils.h"
#include "rgbd_util/CameraParams.h"
#include "cuda_util/cudaUtils.h"
#include "celiu_flow/Image.h"
#include "celiu_flow/opticalFlowUtils.h"
using std::cout;
using std::endl;
using namespace scene_flow;

#define MODEL_OCCLUSION //include an occlusion weight in the color data term to reduce that penalty if the point is occluded

#define EX_CUDA_POSINF __int_as_float(0x7f800000)
#define EX_CUDA_NEGINF __int_as_float(0xff800000)

void CHECK(const DImage& img)
{
	for(int i = 0; i < img.height() * img.width() * img.nchannels(); i++)
	{
		assert(!isnan(img.data()[i]));
		assert(!isinf(img.data()[i]));
	}
}

//20130304: disabling these asserts saves .2s/frame on 320x240
#define CHECKF(f) /*\
	assert(f != EX_CUDA_POSINF);\
	assert(f != EX_CUDA_NEGINF);\
	assert(f == f); //check not nan
	*/

__host__ __device__ float sqrCU(float x)
{
	return x * x;
}

struct thrustAbsDiffFunctor
{
	template <typename Tuple>
	__host__ __device__ float operator () (const Tuple t) const
	{
		return ::fabs(thrust::get<0>(t) - thrust::get<1>(t));
	}
};

template <typename T>
struct thrustSqrFunctor
{
	__host__ __device__ T operator () (const T& x) const
	{
		return x * x;
	}
};

template <typename T>
struct thrustSqrMultFunctor
{
	template <typename Tuple>
	__host__ __device__ T operator () (const Tuple t) const
	{
		return sqrCU(thrust::get<0>(t)) * thrust::get<1>(t);
	}
};

const uint32_t BlockSize1D = 16; //for kernels; 8 is almost exactly same speed as 16, and 32 is too many threads when you add in 3 channels as the third dimension of the block size

/****************************************************************************************************************************************************/

/*
 * compute derivatives of img2 wrt x, y and t
 */
__global__ void imgDerivativesKernel(const uint32_t imgWidth, const uint32_t imgHeight, const uint32_t numChannels, const double* img1, const double* img2, double* dx, double* dy, double* dt)
{
	const double gradFilter[5] = {1,-8,0,8,-1};
	const int32_t i = blockIdx.y * blockDim.y + threadIdx.y, j = blockIdx.x * blockDim.x + threadIdx.x, k = blockIdx.z * blockDim.z + threadIdx.z;

	//TODO use shared memory

	if(i < imgHeight && j < imgWidth)
	{
		const int32_t idx = (i * imgWidth + j) * numChannels + k;
		double dxSum = 0, dxWtSum = 0, dySum = 0, dyWtSum = 0;
		for(int32_t c = j - 2, m = 0; c <= j + 2; c++, m++)
		{
			const int32_t cc = ::max(0, ::min((int32_t)imgWidth - 1, c));
			dxSum += gradFilter[m] * (.4 * img1[(i * imgWidth + cc) * numChannels + k] + .6 * img2[(i * imgWidth + cc) * numChannels + k]);
		//	dxWtSum += ::fabs(gradFilter[m]);
		}
		dx[idx] = dxSum / 12;//dxWtSum;
		for(int32_t r = i - 2, m = 0; r <= i + 2; r++, m++)
		{
			const int32_t rr = ::max(0, ::min((int32_t)imgHeight - 1, r));
			dySum += gradFilter[m] * (.4 * img1[(rr * imgWidth + j) * numChannels + k] + .6 * img2[(rr * imgWidth + j) * numChannels + k]);
		//	dyWtSum += ::fabs(gradFilter[m]);
		}
		dy[idx] = dySum / 12;//dyWtSum;

		dt[idx] = img2[idx] - img1[idx];
	}
}

/*
 * compute derivatives of img2 wrt x, y and t
 *
 * clamp each resulting value to [-maxVal, maxVal]
 */
__global__ void imgDerivativesKernelWithClamp(const uint32_t imgWidth, const uint32_t imgHeight, const uint32_t numChannels, const double* img1, const double* img2, double* dx, double* dy, double* dt, const double maxVal)
{
	const double gradFilter[5] = {1,-8,0,8,-1};
	const int32_t i = blockIdx.y * blockDim.y + threadIdx.y, j = blockIdx.x * blockDim.x + threadIdx.x, k = blockIdx.z * blockDim.z + threadIdx.z;

	//TODO use shared memory

	if(i < imgHeight && j < imgWidth)
	{
		const int32_t idx = (i * imgWidth + j) * numChannels + k;
		double dxSum = 0, dxWtSum = 0, dySum = 0, dyWtSum = 0;
		for(int32_t c = j - 2, m = 0; c <= j + 2; c++, m++)
		{
			const int32_t cc = ::max(0, ::min((int32_t)imgWidth - 1, c));
			dxSum += gradFilter[m] * (.4 * img1[(i * imgWidth + cc) * numChannels + k] + .6 * img2[(i * imgWidth + cc) * numChannels + k]);
		//	dxWtSum += ::fabs(gradFilter[m]);
		}
		dx[idx] = ::min(::max(dxSum / 12/*dxWtSum*/, -maxVal), maxVal);
		for(int32_t r = i - 2, m = 0; r <= i + 2; r++, m++)
		{
			const int32_t rr = ::max(0, ::min((int32_t)imgHeight - 1, r));
			dySum += gradFilter[m] * (.4 * img1[(rr * imgWidth + j) * numChannels + k] + .6 * img2[(rr * imgWidth + j) * numChannels + k]);
		//	dyWtSum += ::fabs(gradFilter[m]);
		}
		dy[idx] = ::min(::max(dySum / 12/*dyWtSum*/, -maxVal), maxVal);

		dt[idx] = ::min(::max(img2[idx] - img1[idx], -maxVal), maxVal);
	}
}

/****************************************************************************************************************************************************/

/*
 * u, du: three-channel; the sum of them is the current flow estimate
 *
 * write ux and uy, the first derivatives of u+du
 */
__global__ void flowDerivativesKernel(const uint32_t imgWidth, const uint32_t imgHeight, const double* u, const float* du, const uint32_t channel, double* ux, double* uy)
{
	const int32_t i = blockIdx.y * blockDim.y + threadIdx.y, j = blockIdx.x * blockDim.x + threadIdx.x;

	//TODO use shared memory

	if(i < imgHeight && j < imgWidth)
	{
		const uint32_t pix = i * imgWidth + j, pix3 = (i * imgWidth + j) * 3 + channel;
		if(i == imgHeight - 1)
		{
			uy[pix] = 0;
		}
		else
		{
			uy[pix] = u[pix3 + imgWidth * 3] + du[pix3 + imgWidth * 3] - u[pix3] - du[pix3];
		}
		if(j == imgWidth - 1)
		{
			ux[pix] = 0;
		}
		else
		{
			ux[pix] = u[pix3 + 3] + du[pix3 + 3] - u[pix3] - du[pix3];
		}
	}
}

/****************************************************************************************************************************************************/

/*
 * o <- f1 * f2 * f3
 */
__global__ void multiply3Kernel(const uint32_t imgWidth, const uint32_t imgHeight, const double* f1, const double* f2, const double* f3, double* o)
{
	const int32_t i = blockIdx.y * blockDim.y + threadIdx.y, j = blockIdx.x * blockDim.x + threadIdx.x;
	if(i < imgHeight && j < imgWidth)
	{
		const uint32_t pix = i * imgWidth + j;
		o[pix] = f1[pix] * f2[pix] * f3[pix];
	}
}

/*
 * o <- collapse(f1 * f2 * f3), where collapse() takes the mean of all channels
 */
__global__ void multiply3CollapseKernel(const uint32_t imgWidth, const uint32_t imgHeight, const uint32_t nChannels, const double* f1, const double* f2, const double* f3, double* o)
{
	const int32_t i = blockIdx.y * blockDim.y + threadIdx.y, j = blockIdx.x * blockDim.x + threadIdx.x;
	if(i < imgHeight && j < imgWidth)
	{
		const uint32_t pix = i * imgWidth + j;
		const double* f1i = f1 + pix * nChannels, *f2i = f2 + pix * nChannels, *f3i = f3 + pix * nChannels;
		double sum = 0;
		for(uint32_t k = 0; k < nChannels; k++, f1i++, f2i++, f3i++) sum += *f1i * *f2i * *f3i;
		o[pix] = sum / nChannels;
	}
}

#ifdef MODEL_OCCLUSION
/*
 * o <- collapse(f1 * f2 * f3) * f4 * f5, where collapse() takes the mean of all channels
 */
__global__ void multiply3Collapse2Kernel(const uint32_t imgWidth, const uint32_t imgHeight, const uint32_t nChannels, const double* f1, const double* f2, const double* f3, const float* f4, const float* f5, double* o)
{
	const int32_t i = blockIdx.y * blockDim.y + threadIdx.y, j = blockIdx.x * blockDim.x + threadIdx.x;
	if(i < imgHeight && j < imgWidth)
	{
		const uint32_t pix = i * imgWidth + j;
		const double* f1i = f1 + pix * nChannels, *f2i = f2 + pix * nChannels, *f3i = f3 + pix * nChannels;
		double sum = 0;
		for(uint32_t k = 0; k < nChannels; k++, f1i++, f2i++, f3i++) sum += *f1i * *f2i * *f3i;
		o[pix] = sum * f4[pix] * f5[pix] / nChannels;
	}
}
#endif

/****************************************************************************************************************************************************/

//TODO do all 3 laplacians in a single kernel so can get some common subexpressions

/*
 * u: 3-channel; we'll operate on a single channel
 */
__global__ void uvflowLaplacianKernel(const uint32_t imgWidth, const uint32_t imgHeight, const double* u, const uint32_t uChannel, const double* nonnormWeights, const float* normWeights, double* uLaplacian)
{
	const int32_t i = blockIdx.y * blockDim.y + threadIdx.y, j = blockIdx.x * blockDim.x + threadIdx.x;

	//TODO shared memory

	if(i < imgHeight && j < imgWidth)
	{
		const uint32_t pix = i * imgWidth + j;
		double lapval = 0;

		// horizontal filtering
		if(j < imgWidth - 1)
		{
			lapval -= (u[(pix + 1) * 3 + uChannel] - u[pix * 3 + uChannel]) * nonnormWeights[pix] * normWeights[pix * 2 + 0];
		}
		if(j > 0)
		{
			lapval += (u[pix * 3 + uChannel] - u[(pix - 1) * 3 + uChannel]) * nonnormWeights[pix - 1] * normWeights[(pix - 1) * 2 + 0];
		}

		// vertical filtering
		if(i < imgHeight - 1)
		{
			lapval -= (u[(pix + imgWidth) * 3 + uChannel] - u[pix * 3 + uChannel]) * nonnormWeights[pix] * normWeights[pix * 2 + 1];
		}
		if(i > 0)
		{
			lapval += (u[pix * 3 + uChannel] - u[(pix - imgWidth) * 3 + uChannel]) * nonnormWeights[pix - imgWidth] * normWeights[(pix - imgWidth) * 2 + 1];
		}

		/*
		 * normalize such that if normWeights are all 1, we always divide by 1 here
		 */
		double norm = 0;
		int count = 0;
		if(j < imgWidth - 1) {norm += normWeights[pix * 2 + 0]; count++;}
		if(j > 0) {norm += normWeights[(pix - 1) * 2 + 0]; count++;}
		if(i < imgHeight - 1) {norm += normWeights[pix * 2 + 1]; count++;}
		if(i > 0) {norm += normWeights[(pix - imgWidth) * 2 + 1]; count++;}
		if(fabs(norm) < 1e-5/* TODO ? */) lapval = 0;
		else lapval /= (norm / count);
		uLaplacian[pix] = lapval;
	}
}

/*
 * u: 3-channel; we'll operate on a single channel
 */
__global__ void wflowLaplacianKernel(const uint32_t imgWidth, const uint32_t imgHeight, const double* u, const uint32_t uChannel, const double* nonnormWeights, const float* normWeights, const float* postWeights, double* uLaplacian)
{
	const int32_t i = blockIdx.y * blockDim.y + threadIdx.y, j = blockIdx.x * blockDim.x + threadIdx.x;

	//TODO shared memory

	if(i < imgHeight && j < imgWidth)
	{
		const uint32_t pix = i * imgWidth + j;
		double lapval = 0;

		// horizontal filtering
		if(j < imgWidth - 1)
		{
			lapval -= (u[(pix + 1) * 3 + uChannel] - u[pix * 3 + uChannel]) * nonnormWeights[pix] * normWeights[pix * 2 + 0];
		}
		if(j > 0)
		{
			lapval += (u[pix * 3 + uChannel] - u[(pix - 1) * 3 + uChannel]) * nonnormWeights[pix - 1] * normWeights[(pix - 1) * 2 + 0];
		}

		// vertical filtering
		if(i < imgHeight - 1)
		{
			lapval -= (u[(pix + imgWidth) * 3 + uChannel] - u[pix * 3 + uChannel]) * nonnormWeights[pix] * normWeights[pix * 2 + 1];
		}
		if(i > 0)
		{
			lapval += (u[pix * 3 + uChannel] - u[(pix - imgWidth) * 3 + uChannel]) * nonnormWeights[pix - imgWidth] * normWeights[(pix - imgWidth) * 2 + 1];
		}

		/*
		 * normalize such that if normWeights are all 1, we always divide by 1 here
		 */
		double norm = 0;
		int count = 0;
		if(j < imgWidth - 1) {norm += normWeights[pix * 2 + 0]; count++;}
		if(j > 0) {norm += normWeights[(pix - 1) * 2 + 0]; count++;}
		if(i < imgHeight - 1) {norm += normWeights[pix * 2 + 1]; count++;}
		if(i > 0) {norm += normWeights[(pix - imgWidth) * 2 + 1]; count++;}
		if(fabs(norm) < 1e-5/* TODO ? */) lapval = 0;
		else lapval /= (norm / count);
		uLaplacian[pix] = postWeights[pix] * lapval;
	}
}

/****************************************************************************************************************************************************/

/*
 * fill derivative terms for color data, depth data and smoothness energies
 */
__global__ void penaltyDerivativesKernel(const uint32_t imgWidth, const uint32_t imgHeight, const uint32_t nChannels, const double epsD, const double epsZ, const double epsS,
	const double* imdx, const double* imdy, const double* imdt, const double* dmdx, const double* dmdy, const double* dmdt, const float* du,
	const double* ux, const double* uy, const double* vx, const double* vy, const double* wx, const double* wy, const double colorDataTermWeight, const float* depthDataTermWeights, const float* depthSmoothnessTermWeights,
#ifdef MODEL_OCCLUSION
	const float* occPenalty,
#endif
	double* piData, double* psiData, double* phiData)
{
	const int32_t i = blockIdx.y * blockDim.y + threadIdx.y, j = blockIdx.x * blockDim.x + threadIdx.x;

	if(i < imgHeight && j < imgWidth)
	{
		const uint32_t pix = i * imgWidth + j;

		const double temp = sqrCU(ux[pix]) + sqrCU(uy[pix]) + sqrCU(vx[pix]) + sqrCU(vy[pix]) + depthSmoothnessTermWeights[pix] * (sqrCU(wx[pix]) + sqrCU(wy[pix]));
		phiData[pix] = 0.5/sqrt(temp+epsS);

		for(int k=0;k<nChannels;k++)
		{
			const int offset = pix * nChannels+k;
			const double temp = (imdt[offset]+imdx[offset]*du[pix * 3 + 0] + imdy[offset] * du[pix * 3 + 1])
#ifdef MODEL_OCCLUSION
			                      * occPenalty[pix]
#endif
			                      ;
			psiData[offset] = .5 / sqrt(sqrCU(temp) + epsD) * colorDataTermWeight;
		}

		const double temp2 = dmdt[pix] + dmdx[pix] * du[pix * 3 + 0] + dmdy[pix] * du[pix * 3 + 1] - du[pix * 3 + 2];
		piData[pix] = .5 / sqrt(sqrCU(temp2) + epsZ) * depthDataTermWeights[pix];
	}
}

/****************************************************************************************************************************************************/

const float maxFlowStep = .3; //max change in any one dimension at any pixel, to avoid diverging; TODO ?

/*
 * A: for each variable, 7 entries in A plus one for rhs; these are arranged indexed first by the [0..7] for each variable's row of A, in order to get coalesced memory reads (this does help a bunch 20130306)
 *
 * TODO having all 8 entries per row in A is slower (275s/frame vs 255s) than having 7 in A and one in a separate array rhs -- ??
 */
__global__ void sceneFlowSingleScaleJacobiKernel(const uint32_t imWidth, const uint32_t imHeight, const uint32_t maxNZ, const float* A, float* x, const bool redblack)
{
	const int32_t i = blockIdx.y * blockDim.y + threadIdx.y, j = blockIdx.x * blockDim.x + threadIdx.x, k = threadIdx.z;
	assert(maxNZ == 8);
	assert(blockDim.z == 3);

	//20130227: the shared memory gives about a 13% speedup
	//this will break if block z-size is 1; we must load all x values at the current pixel before we update them, so those three threads have to be in the same block
	__shared__ float xblock[BlockSize1D + 2][BlockSize1D + 2][3];

	//fill xblock in linear order -- each thread should load 1 or 2 floats
	uint32_t lindex = threadIdx.y * blockDim.x + threadIdx.x;
	int32_t xbj = lindex % (BlockSize1D + 2), xbi = lindex / (BlockSize1D + 2);
	int32_t xj = blockIdx.x * blockDim.x - 1 + xbj, xi = blockIdx.y * blockDim.y - 1 + xbi;
	if(xi >= 0 && xj >= 0 && xi < imHeight && xj < imWidth) xblock[xbi][xbj][k] = x[(xi * imWidth + xj) * 3 + k];
	lindex += BlockSize1D * BlockSize1D;
	xbi = lindex / (BlockSize1D + 2); xbj = lindex % (BlockSize1D + 2);
	xj = blockIdx.x * blockDim.x - 1 + xbj; xi = blockIdx.y * blockDim.y - 1 + xbi;
	if(lindex < (BlockSize1D + 2) * (BlockSize1D + 2) && xi >= 0 && xj >= 0 && xi < imHeight && xj < imWidth) xblock[xbi][xbj][k] = x[(xi * imWidth + xj) * 3 + k];

	__syncthreads(); //per-block

	if(i < imHeight && j < imWidth)
	{
		if((i + j) % 2 == redblack)
		{
			const uint32_t pixOffset = (i * imWidth + j) * 3 + k, nzSize = imWidth * imHeight * 3;
			float newval = 0;
			if(j > 0) //left
			{
				newval -= A[0 * nzSize + pixOffset] * xblock[threadIdx.y + 1][threadIdx.x][k];
			}
			if(i > 0) //up
			{
				newval -= A[1 * nzSize + pixOffset] * xblock[threadIdx.y][threadIdx.x + 1][k];
			}
			if(j < imWidth - 1) //right
			{
				newval -= A[2 * nzSize + pixOffset] * xblock[threadIdx.y + 1][threadIdx.x + 2][k];
			}
			if(i < imHeight - 1) //down
			{
				newval -= A[3 * nzSize + pixOffset] * xblock[threadIdx.y + 2][threadIdx.x + 1][k];
			}

			CHECKF(newval);
			newval += A[7 * nzSize + pixOffset];
			CHECKF(newval);
			//20130306: I tried replacing this switch with an array of offsets indexed by k; it added .6s/frame to runtime whichever way I did it
			const float* xpix = xblock[threadIdx.y + 1][threadIdx.x + 1];
			switch(k)
			{
				case 0:
					newval -= A[5 * nzSize + pixOffset] * xpix[1];
					CHECKF(newval);
					newval -= A[6 * nzSize + pixOffset] * xpix[2];
					CHECKF(newval);
					newval /= A[4 * nzSize + pixOffset];
					CHECKF(newval);
					break;
				case 1:
					newval -= A[4 * nzSize + pixOffset] * xpix[0];
					CHECKF(newval);
					newval -= A[6 * nzSize + pixOffset] * xpix[2];
					CHECKF(newval);
					newval /= A[5 * nzSize + pixOffset];
					CHECKF(newval);
					break;
				case 2:
					newval -= A[4 * nzSize + pixOffset] * xpix[0];
					CHECKF(newval);
					newval -= A[5 * nzSize + pixOffset] * xpix[1];
					CHECKF(newval);
					newval /= A[6 * nzSize + pixOffset];
					CHECKF(newval);
					break;
				default: assert(false);
			}

			const float omega = .7; //SOR parameter; .1 and .7 seem to work equally well on throwball1 when we run to convergence at the 1e-6 level, but omega = .8 fails to converge there; TODO ?
			x[(i * imWidth + j) * 3 + k] = omega * newval + (1 - omega) * xpix[k];
#if 1
			//TODO hack to avoid changing too much in one step
			if(x[(i * imWidth + j) * 3 + k] < xpix[k] - maxFlowStep) x[(i * imWidth + j) * 3 + k] = xpix[k] - maxFlowStep;
			else if(x[(i * imWidth + j) * 3 + k] > xpix[k] + maxFlowStep) x[(i * imWidth + j) * 3 + k] = xpix[k] + maxFlowStep;
#endif
		}
	}
}

/******************************************************************************************************************************************************/

/*
 * write coeffs of the system Ax = rhs into A (it'll hold both A and rhs elements)
 */
__global__ void fillSystemMtxKernel(const uint32_t imWidth, const uint32_t imHeight, const double alpha, const double gamma, const double* u,
	const double* ix2_d, const double* iy2_d, const double* ixy_d, const double* itx_d, const double* ity_d,
	const double* dx2_d, const double* dy2_d, const double* dxy_d, const double* dtx_d, const double* dty_d, const double* dx_d, const double* dy_d, const double* dt_d,
	const double* uL_d, const double* vL_d, const double* wL_d,
	const double* Phi1, const double* Pi1_d,
	const float* depthSmoothnessTermWeights, const float* depthMagTermWeights, const float* regularizationWeights,
	float* A)
{
	const int32_t i = blockIdx.y * blockDim.y + threadIdx.y, j = blockIdx.x * blockDim.x + threadIdx.x;

	if(i < imHeight && j < imWidth)
	{
		const uint32_t l = i * imWidth + j;
		const uint32_t nzSize = imWidth * imHeight * 3; //to help arrange elements of A so we can get coalesced accesses

		/*
		 * approximate second partials of du, dv, dw
		 *
		 * we don't need the else clauses here because the elements of A that are zero are always zero--it's only a function of pixel location--so we can instead initialize A to all zero before filling it over and over
		 */
		double weightSumXY = 0, weightSumZ = 0;
		if(j > 0) //left
		{
			int offset2 = l - 1;
			/*
			 * according to the Bruhn et al '05 paper, it'd be more correct math to do _weight = (phiData[offset] + phiData[offset2]) / 2, but this is faster
			 */
			const double wxy = Phi1[offset2] * regularizationWeights[(l - 1) * 2 + 0], wz = wxy * depthSmoothnessTermWeights[offset2];
			A[0 * nzSize + (l * 3 + 0)] = alpha * wxy;
			A[0 * nzSize + (l * 3 + 1)] = alpha * wxy;
			A[0 * nzSize + (l * 3 + 2)] = alpha * wz;
			weightSumXY += wxy;
			weightSumZ += wz;
		}
		if(i > 0) //up
		{
			int offset2 = l - imWidth;
			const double wxy = Phi1[offset2] * regularizationWeights[(l - imWidth) * 2 + 1], wz = wxy * depthSmoothnessTermWeights[offset2];
			A[1 * nzSize + (l * 3 + 0)] = alpha * wxy;
			A[1 * nzSize + (l * 3 + 1)] = alpha * wxy;
			A[1 * nzSize + (l * 3 + 2)] = alpha * wz;
			weightSumXY += wxy;
			weightSumZ += wz;
		}
		if(j < imWidth - 1) //right
		{
		//	int offset2 = l + 1;
			const double wxy = Phi1[l] * regularizationWeights[l * 2 + 0], wz = wxy * depthSmoothnessTermWeights[l];
			A[2 * nzSize + (l * 3 + 0)] = alpha * wxy;
			A[2 * nzSize + (l * 3 + 1)] = alpha * wxy;
			A[2 * nzSize + (l * 3 + 2)] = alpha * wz;
			weightSumXY += wxy;
			weightSumZ += wz;
		}
		if(i < imHeight - 1) //down
		{
		//	int offset2 = l + imWidth;
			const double wxy = Phi1[l] * regularizationWeights[l * 2 + 1], wz = wxy * depthSmoothnessTermWeights[l];
			A[3 * nzSize + (l * 3 + 0)] = alpha * wxy;
			A[3 * nzSize + (l * 3 + 1)] = alpha * wxy;
			A[3 * nzSize + (l * 3 + 2)] = alpha * wz;
			weightSumXY += wxy;
			weightSumZ += wz;
		}

		/*
		 * A is indexed by (item within row * nzSize + (pixelIdx * 3 + {0,1,2 for u,v,w variable}), where item within row is 0 for left nbr, 1 for up nbr, ..., 4 for same-pixel u, 5 for same-pixel v, 6 for same-pixel w, 7 for system RHS element
		 */

		A[4 * nzSize + (l * 3 + 0)] = -( ix2_d[l] + dx2_d[l] + alpha * weightSumXY );
		A[5 * nzSize + (l * 3 + 0)] = - ixy_d[l] - dxy_d[l];
		A[6 * nzSize + (l * 3 + 0)] = Pi1_d[l] * dx_d[l];
		A[7 * nzSize + (l * 3 + 0)] = -( -itx_d[l] - dtx_d[l] + alpha * - uL_d[l] );

		//flow magnitude penalty
		A[4 * nzSize + (l * 3 + 0)] += -( gamma );
		A[7 * nzSize + (l * 3 + 0)] += -( -gamma * u[l * 3 + 0] );


		A[4 * nzSize + (l * 3 + 1)] = - ixy_d[l] - dxy_d[l];
		A[5 * nzSize + (l * 3 + 1)] = -( iy2_d[l] + dy2_d[l] + alpha * weightSumXY );
		A[6 * nzSize + (l * 3 + 1)] = Pi1_d[l] * dy_d[l];
		A[7 * nzSize + (l * 3 + 1)] = -( -ity_d[l] - dty_d[l] + alpha * - vL_d[l] );

		//flow magnitude penalty
		A[5 * nzSize + (l * 3 + 1)] += -( gamma );
		A[7 * nzSize + (l * 3 + 1)] += -( -gamma * u[l * 3 + 1] );


		A[4 * nzSize + (l * 3 + 2)] = Pi1_d[l] * dx_d[l];
		A[5 * nzSize + (l * 3 + 2)] = Pi1_d[l] * dy_d[l];
		A[6 * nzSize + (l * 3 + 2)] = -( Pi1_d[l] + alpha * weightSumZ );
		A[7 * nzSize + (l * 3 + 2)] = -( Pi1_d[l] * dt_d[l] + alpha * - wL_d[l] );

		//flow magnitude penalty
		A[6 * nzSize + (l * 3 + 2)] += -( gamma * depthMagTermWeights[l] );
		A[7 * nzSize + (l * 3 + 2)] += -( -gamma * depthMagTermWeights[l] * u[l * 3 + 2] );
	}
}

/******************************************************************************************************************************************************/

__global__ void warpColorImgWithSceneFlowKernel(const uint32_t imgWidth, const uint32_t imgHeight, const uint32_t nChannels, const double* img1, const double* img2, const double* u, double* warpedImg2)
{
	const int32_t i = blockIdx.y * blockDim.y + threadIdx.y, j = blockIdx.x * blockDim.x + threadIdx.x;

	//TODO shared memory, or use textures

	if(i < imgHeight && j < imgWidth)
	{
		const uint32_t l = i * imgWidth + j;

		const double x = j + u[l * 3 + 0], y = i + u[l * 3 + 1]; //location to sample at
		if(x < 0 || x > imgWidth - 1 || y < 0 || y > imgHeight - 1)
		{
			for(int k=0;k<nChannels;k++) warpedImg2[l * nChannels + k] = img1[l * nChannels + k];
		}
		else
		{
			//bilerp
			const int32_t x0 = ::max(0, ::min((int32_t)imgWidth - 1, (int32_t)floor(x))), y0 = ::max(0, ::min((int32_t)imgHeight - 1, (int32_t)floor(y))), x1 = ::min((int32_t)imgWidth - 1, x0 + 1), y1 = ::min((int32_t)imgHeight - 1, y0 + 1);
			const float ax = x - x0, ay = y - y0;
			for(int k = 0; k < nChannels; k++)
				warpedImg2[l * nChannels + k] = (1 - ax) * (1 - ay) * img2[(y0 * imgWidth + x0) * nChannels + k]
				                                + (1 - ax) * ay * img2[(y1 * imgWidth + x0) * nChannels + k]
				                                + ax * (1 - ay) * img2[(y0 * imgWidth + x1) * nChannels + k]
				                                + ax * ay * img2[(y1 * imgWidth + x1) * nChannels + k];
		}
	}
}

__global__ void warpDepthImgWithSceneFlowKernel(const uint32_t imgWidth, const uint32_t imgHeight, const double* img1, const double* img2, const double* u, double* warpedImg2)
{
	const int32_t i = blockIdx.y * blockDim.y + threadIdx.y, j = blockIdx.x * blockDim.x + threadIdx.x;

	//TODO shared memory, or use textures

	if(i < imgHeight && j < imgWidth)
	{
		const uint32_t l = i * imgWidth + j;

		const double x = j + u[l * 3 + 0], y = i + u[l * 3 + 1]; //location to sample at
		if(x < 0 || x > imgWidth - 1 || y < 0 || y > imgHeight - 1)
		{
			warpedImg2[l] = img1[l] - u[l * 3 + 2];
		}
		else
		{
			//bilerp
			const int32_t x0 = ::max(0, ::min((int32_t)imgWidth - 1, (int32_t)floor(x))), y0 = ::max(0, ::min((int32_t)imgHeight - 1, (int32_t)floor(y))), x1 = ::min((int32_t)imgWidth - 1, x0 + 1), y1 = ::min((int32_t)imgHeight - 1, y0 + 1);
			const float ax = x - x0, ay = y - y0;
			warpedImg2[l] = (1 - ax) * (1 - ay) * img2[y0 * imgWidth + x0]
			                + (1 - ax) * ay * img2[y1 * imgWidth + x0]
			                + ax * (1 - ay) * img2[y0 * imgWidth + x1]
					          + ax * ay * img2[y1 * imgWidth + x1]
							- u[l * 3 + 2];
		}
	}
}

#ifdef MODEL_OCCLUSION
/******************************************************************************************************************************************************/

__global__ void fillOcclusionPenaltyKernel(const uint32_t imgWidth, const uint32_t imgHeight, const double* dm1, const double* dm2, const double* u, float* occPenalty)
{
	const int32_t i = blockIdx.y * blockDim.y + threadIdx.y, j = blockIdx.x * blockDim.x + threadIdx.x;
	if(i < imgHeight && j < imgWidth)
	{
		const uint32_t l = i * imgWidth + j;
		occPenalty[l] = (dm1[l] + u[l * 3 + 2] < dm2[l] + .02 * sqrCU(dm1[l]/* margin */)) ? 1 : .01/* TODO ? */;
	}
}
#endif

/******************************************************************************************************************************************************/

//#define CHECK_FLOATS //20130304: disabling gets us ~1s/frame on 320x240
#define CHECK_CONVERGENCE
/*
 * current experiments in scene flow
 *
 * regularizationWeights: two-channel for right and down nbr pixels
 */
void sceneFlowSingleScale(const rgbd::CameraParams& camParams, const DImage &Im1, const DImage& dm1, const DImage &Im2, const DImage& dm2, DImage &warpIm2, DImage& warpDm2, DImage &u, DImage &v, DImage& w,
	const double colorDataTermWeight, const DImage& depthDataTermWeights,
	const double alpha, const DImage& depthSmoothnessTermWeights,
	const double gamma, const DImage& depthMagTermWeights,
	const DImage& regularizationWeights,
	int nOuterFPIterations, int nInnerFPIterations, int nSORIterations, const bool allowEarlyTermination)
{
#ifdef CHECK_FLOATS
	CHECK(u);
	CHECK(v);
	CHECK(w);
	CHECK(depthDataTermWeights);
	CHECK(depthSmoothnessTermWeights);
	CHECK(depthMagTermWeights);
	CHECK(regularizationWeights);
#endif

	rgbd::timer t;
	const int imWidth=Im1.width(),imHeight=Im1.height(),nChannels=Im1.nchannels();

	thrust::device_vector<float> depthDataTermWeightsDev(imWidth * imHeight);
	thrust::device_vector<float> depthSmoothnessTermWeightsDev(imWidth * imHeight);
	thrust::device_vector<float> depthMagTermWeightsDev(imWidth * imHeight);
	thrust::device_vector<float> regularizationWeightsDev(imWidth * imHeight * 2);
	thrust::copy(depthDataTermWeights.data(), depthDataTermWeights.data() + imWidth * imHeight, depthDataTermWeightsDev.begin());
	thrust::copy(depthSmoothnessTermWeights.data(), depthSmoothnessTermWeights.data() + imWidth * imHeight, depthSmoothnessTermWeightsDev.begin());
	thrust::copy(depthMagTermWeights.data(), depthMagTermWeights.data() + imWidth * imHeight, depthMagTermWeightsDev.begin());
	thrust::copy(regularizationWeights.data(), regularizationWeights.data() + imWidth * imHeight * 2, regularizationWeightsDev.begin());

#ifdef CHECK_CONVERGENCE
	thrust::device_vector<float> convTestWtsDev; //allow to weight convergence in xy and in z separately; should be const
	const double convergenceThreshold = 4e-4;//1e-3; //TODO ?
{
	thrust::host_vector<float> convTestWts(imWidth * imHeight * 3);
	for(size_t i = 0, l = 0; i < imHeight; i++)
		for(size_t j = 0; j < imWidth; j++, l++)
		{
			convTestWts[l * 3 + 0] = 1;
			convTestWts[l * 3 + 1] = 1;
			convTestWts[l * 3 + 2] = camParams.focalLength / dm1(i, j);
		}
	convTestWtsDev = convTestWts;
}
#endif

	thrust::device_vector<double> Im1dev(imWidth * imHeight * nChannels), Im2dev(imWidth * imHeight * nChannels), warpIm2dev(imWidth * imHeight * nChannels);
	thrust::device_vector<double> dm1dev(imWidth * imHeight), dm2dev(imWidth * imHeight), warpDm2dev(imWidth * imHeight);
	thrust::copy(Im1.data(), Im1.data() + imWidth * imHeight * nChannels, Im1dev.begin());
	thrust::copy(Im2.data(), Im2.data() + imWidth * imHeight * nChannels, Im2dev.begin());
	thrust::copy(warpIm2.data(), warpIm2.data() + imWidth * imHeight * nChannels, warpIm2dev.begin());
	thrust::copy(dm1.data(), dm1.data() + imWidth * imHeight, dm1dev.begin());
	thrust::copy(dm2.data(), dm2.data() + imWidth * imHeight, dm2dev.begin());
	thrust::copy(warpDm2.data(), warpDm2.data() + imWidth * imHeight, warpDm2dev.begin());

	thrust::device_vector<double> imdxDev(imWidth * imHeight * nChannels), imdyDev(imWidth * imHeight * nChannels), imdtDev(imWidth * imHeight * nChannels);
	thrust::device_vector<double> dmdxDev(imWidth * imHeight), dmdyDev(imWidth * imHeight), dmdtDev(imWidth * imHeight);

	thrust::device_vector<float> duDev(imWidth * imHeight * 3);

	thrust::host_vector<double> uHost(imWidth * imHeight * 3);
	for(size_t i = 0, l = 0; i < imHeight; i++)
		for(size_t j = 0; j < imWidth; j++, l++)
		{
			uHost[l * 3 + 0] = u.data()[l];
			uHost[l * 3 + 1] = v.data()[l];
			uHost[l * 3 + 2] = w.data()[l];
		}
	thrust::device_vector<double> uDev(imWidth * imHeight * 3);
	thrust::copy(uHost.begin(), uHost.end(), uDev.begin());

	thrust::device_vector<double> ux(imWidth * imHeight), uy(imWidth * imHeight), vx(imWidth * imHeight), vy(imWidth * imHeight), wx(imWidth * imHeight), wy(imWidth * imHeight);
	thrust::device_vector<double> piDataDev(imWidth * imHeight),  //1st deriv of depth data energy
		phiDataDev(imWidth * imHeight),  //1st deriv of smoothness energy
		psiDataDev(imWidth * imHeight * nChannels); //1st deriv of color data energy

	thrust::device_vector<double> imdxyDev(imWidth * imHeight), imdx2Dev(imWidth * imHeight), imdy2Dev(imWidth * imHeight), imdtdxDev(imWidth * imHeight), imdtdyDev(imWidth * imHeight);
	thrust::device_vector<double> dmdxyDev(imWidth * imHeight), dmdx2Dev(imWidth * imHeight), dmdy2Dev(imWidth * imHeight), dmdtdxDev(imWidth * imHeight), dmdtdyDev(imWidth * imHeight);

#ifdef MODEL_OCCLUSION
	//occlusion penalty q
	thrust::device_vector<float> q(imWidth * imHeight);
#endif

	thrust::device_vector<double> uLaplacianDev(imWidth * imHeight), vLaplacianDev(imWidth * imHeight), wLaplacianDev(imWidth * imHeight);

	//linear system coefficients
	const uint32_t maxNZ = 8; /* max nonzeros per row of A, plus one for RHS */
	thrust::device_vector<float> Adev(imWidth * imHeight * 3 * maxNZ, 0.0f); //A and rhs in a single array

	const double epsD = 1e-6, epsZ = 1e-10/*4 better than 10 for synthetic data */, epsS = 1e-6; //for the psi functions: sqr(epsilon)
	t.stop("allocate imgs");

	//--------------------------------------------------------------------------
	// the outer fixed point iteration
	//--------------------------------------------------------------------------
	for(int count=0;count<nOuterFPIterations;count++)
	{
		rgbd::timer t2;

		t.restart();
		// compute image gradient
	{
		const dim3 blockSizeIm(BlockSize1D, BlockSize1D, 1);
		const dim3 nBlocksIm(ceil((float)imWidth / BlockSize1D), ceil((float)imHeight / BlockSize1D), nChannels);
		imgDerivativesKernel<<<nBlocksIm, blockSizeIm>>>(imWidth, imHeight, nChannels, Im1dev.data().get(), warpIm2dev.data().get(), imdxDev.data().get(), imdyDev.data().get(), imdtDev.data().get());

		const dim3 blockSizeDm(BlockSize1D, BlockSize1D, 1);
		const dim3 nBlocksDm(ceil((float)imWidth / BlockSize1D), ceil((float)imHeight / BlockSize1D), 1);
		const double maxGrad = .04; //TODO ?
		imgDerivativesKernelWithClamp<<<nBlocksDm, blockSizeDm>>>(imWidth, imHeight, 1, dm1dev.data().get(), warpDm2dev.data().get(), dmdxDev.data().get(), dmdyDev.data().get(), dmdtDev.data().get(), maxGrad);

	//	CUDA_CALL(cudaDeviceSynchronize());
	}

		t.stop("get dxs");

		thrust::fill(duDev.begin(), duDev.end(), 0.0f);
#ifdef MODEL_OCCLUSION
		//piecewise constant occlusion penalty -- TODO will this be useful?
	{
		const dim3 blockSize(BlockSize1D, BlockSize1D, 1);
		const dim3 nBlocks(ceil((float)imWidth / BlockSize1D), ceil((float)imHeight / BlockSize1D), 1);
		fillOcclusionPenaltyKernel<<<nBlocks, blockSize>>>(imWidth, imHeight, dm1dev.data().get(), dm2dev.data().get(), uDev.data().get(), q.data().get());
	//	CUDA_CALL(cudaDeviceSynchronize());
	}
#endif

		//--------------------------------------------------------------------------
		// the inner fixed point iteration
		//--------------------------------------------------------------------------
		for(int hh=0;hh<nInnerFPIterations;hh++)
		{
			// compute the derivatives of the current flow field
		{
			const dim3 blockSize(BlockSize1D, BlockSize1D, 1);
			const dim3 nBlocks(ceil((float)imWidth / BlockSize1D), ceil((float)imHeight / BlockSize1D), 1);
			flowDerivativesKernel<<<nBlocks, blockSize>>>(imWidth, imHeight, uDev.data().get(), duDev.data().get(), 0, ux.data().get(), uy.data().get());
			flowDerivativesKernel<<<nBlocks, blockSize>>>(imWidth, imHeight, uDev.data().get(), duDev.data().get(), 1, vx.data().get(), vy.data().get());
			flowDerivativesKernel<<<nBlocks, blockSize>>>(imWidth, imHeight, uDev.data().get(), duDev.data().get(), 2, wx.data().get(), wy.data().get());
		//	CUDA_CALL(cudaDeviceSynchronize());
		}

#ifdef CHECK_FLOATS
			CHECK(ux);
			CHECK(uy);
			CHECK(vx);
			CHECK(vy);
			CHECK(wx);
			CHECK(wy);
#endif

			/*
			 * calculate derivatives of penalty functions
			 */
		{
			const dim3 blockSize(BlockSize1D, BlockSize1D, 1);
			const dim3 nBlocks(ceil((float)imWidth / BlockSize1D), ceil((float)imHeight / BlockSize1D), 1);
			penaltyDerivativesKernel<<<nBlocks, blockSize>>>(imWidth, imHeight, nChannels, epsD, epsZ, epsS,
				imdxDev.data().get(), imdyDev.data().get(), imdtDev.data().get(), dmdxDev.data().get(), dmdyDev.data().get(), dmdtDev.data().get(), duDev.data().get(),
				ux.data().get(), uy.data().get(), vx.data().get(), vy.data().get(), wx.data().get(), wy.data().get(),
				colorDataTermWeight, depthDataTermWeightsDev.data().get(), depthSmoothnessTermWeightsDev.data().get(),
#ifdef MODEL_OCCLUSION
				q.data().get(),
#endif
				piDataDev.data().get(), psiDataDev.data().get(), phiDataDev.data().get());
		//	CUDA_CALL(cudaDeviceSynchronize());
		}

#ifdef CHECK_FLOATS
			CHECK(Phi_1st);
			CHECK(Psi_1st);
			CHECK(Pi_1st);
#endif

			/*
			 * calculate intermediate products for building the linear system
			 */
		{
			const dim3 blockSize(BlockSize1D, BlockSize1D, 1);
			const dim3 nBlocks(ceil((float)imWidth / BlockSize1D), ceil((float)imHeight / BlockSize1D), 1);

			//weighted products of derivatives of color
			multiply3CollapseKernel<<<nBlocks, blockSize>>>(imWidth, imHeight, nChannels, psiDataDev.data().get(), imdxDev.data().get(), imdyDev.data().get(), imdxyDev.data().get());
			multiply3CollapseKernel<<<nBlocks, blockSize>>>(imWidth, imHeight, nChannels, psiDataDev.data().get(), imdxDev.data().get(), imdxDev.data().get(), imdx2Dev.data().get());
			multiply3CollapseKernel<<<nBlocks, blockSize>>>(imWidth, imHeight, nChannels, psiDataDev.data().get(), imdyDev.data().get(), imdyDev.data().get(), imdy2Dev.data().get());
#ifdef MODEL_OCCLUSION
			multiply3Collapse2Kernel<<<nBlocks, blockSize>>>(imWidth, imHeight, nChannels, psiDataDev.data().get(), imdxDev.data().get(), imdtDev.data().get(), q.data().get(), q.data().get(), imdtdxDev.data().get());
			multiply3Collapse2Kernel<<<nBlocks, blockSize>>>(imWidth, imHeight, nChannels, psiDataDev.data().get(), imdyDev.data().get(), imdtDev.data().get(), q.data().get(), q.data().get(), imdtdyDev.data().get());
#else
			multiply3CollapseKernel<<<nBlocks, blockSize>>>(imWidth, imHeight, nChannels, psiDataDev.data().get(), imdxDev.data().get(), imdtDev.data().get(), imdtdxDev.data().get());
			multiply3CollapseKernel<<<nBlocks, blockSize>>>(imWidth, imHeight, nChannels, psiDataDev.data().get(), imdyDev.data().get(), imdtDev.data().get(), imdtdyDev.data().get());
#endif

			//weighted products of derivatives of depth
			multiply3Kernel<<<nBlocks, blockSize>>>(imWidth, imHeight, piDataDev.data().get(), dmdxDev.data().get(), dmdyDev.data().get(), dmdxyDev.data().get());
			multiply3Kernel<<<nBlocks, blockSize>>>(imWidth, imHeight, piDataDev.data().get(), dmdxDev.data().get(), dmdxDev.data().get(), dmdx2Dev.data().get());
			multiply3Kernel<<<nBlocks, blockSize>>>(imWidth, imHeight, piDataDev.data().get(), dmdyDev.data().get(), dmdyDev.data().get(), dmdy2Dev.data().get());
			multiply3Kernel<<<nBlocks, blockSize>>>(imWidth, imHeight, piDataDev.data().get(), dmdxDev.data().get(), dmdtDev.data().get(), dmdtdxDev.data().get());
			multiply3Kernel<<<nBlocks, blockSize>>>(imWidth, imHeight, piDataDev.data().get(), dmdyDev.data().get(), dmdtDev.data().get(), dmdtdyDev.data().get());

			// laplacian filtering of the current flow field, weighted by Phi_1st
			uvflowLaplacianKernel<<<nBlocks, blockSize>>>(imWidth, imHeight, uDev.data().get(), 0, phiDataDev.data().get(), regularizationWeightsDev.data().get(), uLaplacianDev.data().get());
			uvflowLaplacianKernel<<<nBlocks, blockSize>>>(imWidth, imHeight, uDev.data().get(), 1, phiDataDev.data().get(), regularizationWeightsDev.data().get(), vLaplacianDev.data().get());
			wflowLaplacianKernel<<<nBlocks, blockSize>>>(imWidth, imHeight, uDev.data().get(), 2, phiDataDev.data().get(), regularizationWeightsDev.data().get(), depthSmoothnessTermWeightsDev.data().get(), wLaplacianDev.data().get());

		//	CUDA_CALL(cudaDeviceSynchronize());
		}
#ifdef CHECK_FLOATS
			CHECK(uLaplacian);
			CHECK(vLaplacian);
			CHECK(wLaplacianTmp);
			CHECK(wLaplacian);
#endif

			/*
			 * fill the linear system
			 */
		{
			const dim3 blockSize(BlockSize1D, BlockSize1D, 1);
			const dim3 nBlocks(ceil((float)imWidth / BlockSize1D), ceil((float)imHeight / BlockSize1D), 1);
			fillSystemMtxKernel<<<nBlocks, blockSize>>>(imWidth, imHeight, alpha, gamma, uDev.data().get(),
				imdx2Dev.data().get(), imdy2Dev.data().get(), imdxyDev.data().get(), imdtdxDev.data().get(), imdtdyDev.data().get(),
				dmdx2Dev.data().get(), dmdy2Dev.data().get(), dmdxyDev.data().get(), dmdtdxDev.data().get(), dmdtdyDev.data().get(), dmdxDev.data().get(), dmdyDev.data().get(), dmdtDev.data().get(),
				uLaplacianDev.data().get(), vLaplacianDev.data().get(), wLaplacianDev.data().get(),
				phiDataDev.data().get(), piDataDev.data().get(),
				depthSmoothnessTermWeightsDev.data().get(), depthMagTermWeightsDev.data().get(), regularizationWeightsDev.data().get(),
				Adev.data().get());
		//	CUDA_CALL(cudaDeviceSynchronize());
		}
#ifdef CHECK_FLOATS
			for(size_t i = 0; i < A.size(); i++)
			{
				ASSERT_ALWAYS(!isnan(A[i]));
				ASSERT_ALWAYS(!isinf(A[i]));
			}
#endif
			t.stop("build system");

			/*
			 * solve the linear system
			 */
			t.restart();
		{
			const dim3 blockSize(BlockSize1D, BlockSize1D, 3);
			const dim3 nBlocks(ceil((float)imWidth / BlockSize1D), ceil((float)imHeight / BlockSize1D), 1);
			const uint32_t itersBetweenConvergenceChecks = 25;//100; //TODO ?
			thrust::device_vector<float> prevXDev(3 * imWidth * imHeight, 0.0f);
			for(int32_t i = 0; i < nSORIterations; i++)
			{
				if(i % itersBetweenConvergenceChecks == itersBetweenConvergenceChecks - 1) prevXDev = duDev;
				sceneFlowSingleScaleJacobiKernel<<<nBlocks, blockSize>>>(imWidth, imHeight, maxNZ, Adev.data().get(), duDev.data().get(), 0);
			//	CUDA_CALL(cudaDeviceSynchronize()); //20130314: unnecessary; apparently different kernels and other gpu calls in a single cuda stream run sequentially
				sceneFlowSingleScaleJacobiKernel<<<nBlocks, blockSize>>>(imWidth, imHeight, maxNZ, Adev.data().get(), duDev.data().get(), 1);
			//	CUDA_CALL(cudaDeviceSynchronize());
				if(i % itersBetweenConvergenceChecks == itersBetweenConvergenceChecks - 1)
				{
					const float maxdiff = thrust::transform_reduce(	thrust::make_zip_iterator(thrust::make_tuple(duDev.begin(), prevXDev.begin())),
																					thrust::make_zip_iterator(thrust::make_tuple(duDev.end(), prevXDev.end())),
																					thrustAbsDiffFunctor(), 0.0f, thrust::maximum<float>());
	//				cout << "iter " << i << ": maxdiff " << maxdiff << endl;
					if(maxdiff < 1e-5/*3e-3*//*1e-3*//* output with 1e-4 indistinguishable from with 1e-6 on throwball1 if I use underrelaxation with omega=.7; at 3e-3 it's only visually pretty close; TODO ? */) break;
					if(maxdiff >= maxFlowStep) break; //added 20130828 to try for speed; does speed things up and results are almost identical, so enabled for now
				}
			}
		}
			t.stop("solve");

#ifdef CHECK_FLOATS
			CHECK(du);
			CHECK(dv);
			CHECK(dw);
#endif
		}

		thrust::transform(uDev.begin(), uDev.end(), duDev.begin(), uDev.begin(), thrust::plus<double>()); //uDev += duDev
	{
		const dim3 blockSize(BlockSize1D, BlockSize1D, 1);
		const dim3 nBlocks(ceil((float)imWidth / BlockSize1D), ceil((float)imHeight / BlockSize1D), 1);
		warpColorImgWithSceneFlowKernel<<<nBlocks, blockSize>>>(imWidth, imHeight, nChannels, Im1dev.data().get(), Im2dev.data().get(), uDev.data().get(), warpIm2dev.data().get());
		warpDepthImgWithSceneFlowKernel<<<nBlocks, blockSize>>>(imWidth, imHeight, dm1dev.data().get(), dm2dev.data().get(), uDev.data().get(), warpDm2dev.data().get());
	//	CUDA_CALL(cudaDeviceSynchronize());
	}

		if(allowEarlyTermination)
		{
			/*
			 * check convergence
			 */
			const double statistic = thrust::transform_reduce(thrust::make_zip_iterator(thrust::make_tuple(duDev.begin(), convTestWtsDev.begin())), thrust::make_zip_iterator(thrust::make_tuple(duDev.end(), convTestWtsDev.end())), thrustSqrMultFunctor<float>(), 0.0f, thrust::plus<double>()) / (imWidth * imHeight * 3);
	//		cout << "iter " << count << ": stat " << statistic << endl;
			if(statistic < convergenceThreshold) break;
		}

		t2.stop("run one outer iter");
	}

	/*
	 * copy new flow back to cpu
	 */
	t.restart();
	uHost = uDev;
	for(size_t i = 0, l = 0; i < imHeight; i++)
		for(size_t j = 0; j < imWidth; j++, l++)
		{
			u.data()[l] = uHost[l * 3 + 0];
			v.data()[l] = uHost[l * 3 + 1];
			w.data()[l] = uHost[l * 3 + 2];
		}
	t.stop("copy flow to cpu");
}
