/*
 * principalCurvatures: compute them for all points in a cloud
 *
 * Evan Herbst
 * 1 / 20 / 12
 */

#ifndef EX_COMPUTE_PRINCIPAL_CURVATURES_H
#define EX_COMPUTE_PRINCIPAL_CURVATURES_H

#include <cstdint>
#include <vector>
#include <tuple>
#include <pcl/point_cloud.h>
#include "cloud_features/common.h"

/*
 * pre: cloud has normals and is an organized cloud
 *
 * return: pt -> two principal curvatures; pt -> whether feature valid
 *
 * post: each element of the feature array has the same size, regardless of validity
 */
template <typename PointT>
std::tuple<cloudFeatDescs, std::vector<char>> computePrincipalCurvaturesOrganized(const typename pcl::PointCloud<PointT>::ConstPtr& organizedCloudPtr, const uint32_t nbrhoodRadiusInPixels);

#include "principalCurvatures.ipp"

#endif //header
