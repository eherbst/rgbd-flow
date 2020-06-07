/*
 * cudaUtils
 *
 * Evan Herbst
 * 1 / 28 / 13
 */

#include <cassert>
#include "cuda_util/cudaUtils.h"

namespace cuda
{

/*
 * adapted from the CUDA SDK's deviceQuery example
 *
 * return: max sizes of 3 dimensions of a thread block; max # threads per block total
 */
boost::array<unsigned long, 4> getMaxBlockSize()
{
	int deviceCount = 0;
	CUDA_CALL(cudaGetDeviceCount(&deviceCount));
	assert(deviceCount > 0);
	cudaDeviceProp deviceProp;
	CUDA_CALL(cudaGetDeviceProperties(&deviceProp, 0));
	const boost::array<unsigned long, 4> counts = {{deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2], deviceProp.maxThreadsPerBlock}};
	return counts;
}

} //namespace

