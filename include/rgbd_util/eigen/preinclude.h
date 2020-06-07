/*
 * preinclude: deal with Eigen issues such as the SSE-requires-16-byte-alignment screwiness or having multiple Eigen versions installed
 *
 * include this file before any Eigen header(s)
 *
 * Evan Herbst
 * 4 / 5 / 10
 */

#ifndef EX_RGBD_EIGEN_WRAPPER_H
#define EX_RGBD_EIGEN_WRAPPER_H

#include <cstdint> //uint8_t

//force eigen to run various assertions to catch, for example, some aliasing issues
#undef EIGEN_NO_DEBUG

// electric, fuerte, groovy
#ifndef EIGEN_YES_I_KNOW_SPARSE_MODULE_IS_NOT_STABLE_YET
#define EIGEN_YES_I_KNOW_SPARSE_MODULE_IS_NOT_STABLE_YET 1
#endif

#if 0 //for now we'll let it use SSE
/*
 * avoid the whole SSE-requires-16-byte-alignment screwiness by not aligning anything ever
 */
#define EIGEN_DONT_ALIGN
/*
 * apparently defining EIGEN_DONT_ALIGN doesn't disable vectorization, but does give you a compile error
 */
#define EIGEN_DONT_VECTORIZE
#endif

// get the Eigen namespace defined
#ifdef RGBD_UTIL_USE_EIGEN3
#include <Eigen3/Core>
#else
#include <Eigen/Core>
#endif

/*
 * conveniences
 */
namespace Eigen
{
typedef Matrix<bool, Dynamic, 1> VectorXb;
typedef Matrix<uint32_t, Dynamic, 1> VectorXu;
typedef Matrix<uint8_t, 3, 1> Vector3uc;

/*
 * make storage order explicit, since eigen's default can be changed with a #define and sometimes we want to rely on storage order
 */

typedef Matrix<float, 4, 4, ColMajor> cmMatrix4f;
typedef Matrix<float, Dynamic, 1, ColMajor> cmVectorXf;
typedef Matrix<float, Dynamic, Dynamic, ColMajor> cmMatrixXf;
typedef Transform<float, 3, Affine, ColMajor> cmAffine3f;
typedef Matrix<double, 4, 4, ColMajor> cmMatrix4d;
typedef Matrix<double, Dynamic, 1, ColMajor> cmVectorXd;
typedef Matrix<double, Dynamic, Dynamic, ColMajor> cmMatrixXd;
typedef Transform<double, 3, Affine, ColMajor> cmAffine3d;

typedef Matrix<float, 4, 4, RowMajor> rmMatrix4f;
typedef Matrix<float, Dynamic, 1, RowMajor> rmVectorXf;
typedef Matrix<float, Dynamic, Dynamic, RowMajor> rmMatrixXf;
typedef Transform<float, 3, Affine, RowMajor> rmAffine3f;
typedef Matrix<double, 4, 4, RowMajor> rmMatrix4d;
typedef Matrix<double, Dynamic, 1, RowMajor> rmVectorXd;
typedef Matrix<double, Dynamic, Dynamic, RowMajor> rmMatrixXd;
typedef Transform<double, 3, Affine, RowMajor> rmAffine3d;
}

/*
 * a layer of indirection in case the name ros uses for the eigen namespace changes (which it did from cturtle to diamondback)
 */
namespace rgbd {
	namespace eigen = ::Eigen;
}

#endif //header
