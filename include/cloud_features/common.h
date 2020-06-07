/*
 * common: shared code for point-cloud features
 *
 * Evan Herbst
 * 1 / 20 / 12
 */

#ifndef EX_CLOUD_FEATURES_COMMON_H
#define EX_CLOUD_FEATURES_COMMON_H

#include <Eigen/Geometry>

/*
 * conveniences
 */
namespace Eigen
{
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
 * storage for descriptors
 */
typedef Eigen::rmMatrixXf cloudFeatDescs;

#endif //header
