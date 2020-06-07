/*
 * cudaUtils
 *
 * Evan Herbst
 * 3 / 21 / 12
 */

#include <cstdio>
#include "cuda_util/cudaUtils.h"

void cudaCall(const cudaError_t err, const char* filename, const int line)
{
	const char* s = cudaGetErrorString(err);
	if(err != cudaSuccess)
		printf("cuda error at %s:%d: %s\n", filename, line, s);
}
