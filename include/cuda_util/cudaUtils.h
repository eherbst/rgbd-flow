/*
 * cudaUtils
 *
 * Evan Herbst
 * 3 / 21 / 12
 */

#ifndef EX_CUDA_UTILS_H
#define EX_CUDA_UTILS_H

#include <boost/array.hpp>
#include <cuda.h>
#include <cuda_runtime_api.h>

namespace cuda
{

/*
 * return: max sizes of 3 dimensions of a thread block; max # threads per block total
 */
boost::array<unsigned long, 4> getMaxBlockSize();

} //namespace

/*
 * wrap around cuda calls that return error codes
 */
void cudaCall(const cudaError_t err, const char* filename, const int line);

#define CUDA_CALL(e) cudaCall(e, __FILE__, __LINE__)

#endif //header
