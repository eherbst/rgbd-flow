/*
 * principalCurvatures: compute them for all points in a cloud
 *
 * Evan Herbst
 * 1 / 20 / 12
 */

#include "pcl_rgbd/cloudNormals.h"

/*
 * like computePrincipalCurvatures() but should be faster
 */
template <typename PointT>
std::tuple<cloudFeatDescs, std::vector<char>> computePrincipalCurvaturesOrganized(const typename pcl::PointCloud<PointT>::ConstPtr& organizedCloudPtr, const uint32_t nbrhoodRadiusInPixels)
{
	std::tuple<cloudFeatDescs, std::vector<char>> result;
	Eigen::rmMatrixXf& descs = std::get<0>(result);
	descs.resize(organizedCloudPtr->points.size(), 2);
	descs.setZero();
	std::vector<char>& validity = std::get<1>(result);
	validity.resize(organizedCloudPtr->points.size(), true);

	//20120501: use our code, which is multithreaded, as there's no multithreaded curvature estimation in pcl-1.1
	std::vector<float> pc1, pc2;
	rgbd::computePrincipalCurvaturesOrganized(*organizedCloudPtr, nbrhoodRadiusInPixels, pc1, pc2);
	for(size_t i = 0; i < organizedCloudPtr->points.size(); i++)
	{
		descs(i, 0) = pc1[i];
		descs(i, 1) = pc2[i];
	}

	return result;
}
