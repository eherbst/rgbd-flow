/*
 * cloudNormals: computing and retrieving normals
 *
 * Evan Herbst
 * 3 / 24 / 10
 */

#ifndef EX_PCL_POINT_CLOUD_NORMALS_H
#define EX_PCL_POINT_CLOUD_NORMALS_H

#include <vector>
#include <string>
#include <boost/multi_array.hpp>
#include <opencv2/core/core.hpp>
#include <pcl/point_cloud.h>
#include <Eigen/Geometry>

namespace rgbd
{

	/*
	 * first and second principal curvatures, computed in parallel (unlike pcl as of 20120501)
	 *
	 * pc1, pc2 will be allocated
	 *
	 * pre: cloud has normals
	 */
	template <typename PointT>
	void computePrincipalCurvaturesOrganized(const pcl::PointCloud<PointT>& organizedCloud, const int32_t nbrhoodRadiusInPixels, std::vector<float>& pc1, std::vector<float>& pc2, const bool parallelize = true);

	/** \brief Estimate the point normals and surface curvatures for a given organized point cloud dataset (points)
		* using the data from a different point cloud (surface) for least-squares planar estimation.
		*
		* Copied from point_cloud_mapping/cloud_geometry/nearest.cpp to facilitate making changes
		*
		*
		* Key Differences:
		* 1) Ignore points with z <= 0
		* 2) no pragma omp parallel (causes segfaults)
		* 3) added missing j++ to if(nn_indices.size () < 4) --
		* 	still need to move on to next destination point
		*
		* \param points result is put here. note that the ONLY channels will be normals and curvature
		* also, this WILL HAVE THE DOWNSAMPLED SIZE
		* \param surface the point cloud data to use for least-squares planar estimation
		* \param k the windowing factor (i.e., how many pixels in the depth image in all directions should the neighborhood of a point contain)
		* \param downsample_factor factor for downsampling the input data, i.e., take every Nth row and column in the depth image
		* \param max_z maximum distance threshold (on Z) between the query point and its neighbors (set to -1 to ignore the check)
		* \param viewpoint the viewpoint where the cloud was acquired from (used for normal flip)
		*/
	template <typename PointT>
	void computeOrganizedPointCloudNormals (
		pcl::PointCloud<PointT> &points, const pcl::PointCloud<PointT> &surface,
		boost::multi_array<bool, 2>& surfaceValidity,
		unsigned int k, unsigned int downsample_factor, double max_z,
		const rgbd::eigen::Vector3f& viewpoint, const bool multithread = true);

	/**
		 * Sets the normals in an organized point cloud by calling computeOrganizedPointCloudNormals
		 * and then upsampling using interpolateNormals
		 *
		 * @param organizedCloud should have height, width set
		 * @param targetValidityGrid should be sized for organizedCloud and should have been filled
		 * @param normals_window_size
		 * @param normals_downsample
		 * @param normals_max_z_diff
		 * @param use_GPU_for_normals
		 */
	template <typename PointT>
	void setOrganizedNormals(
			pcl::PointCloud<PointT> &organizedCloud,
			boost::multi_array<bool, 2>& targetValidityGrid,
			unsigned int normals_window_size,
			unsigned int normals_downsample,
			float normals_max_z_diff,
			bool use_GPU_for_normals = false,
			const bool multithread = true);

	/*
	 * Extract the validity grid from an organized point cloud (use z > 0 to determine validity)
	 *
	 * validity_grid will be allocated (to be indexed with (x, y))
	 */
	void getValidityGrid(const cv::Mat_<float>& depthImg, boost::multi_array<bool, 2>& validity_grid);
	template <typename PointT>
	void getValidityGrid(const pcl::PointCloud<PointT> & organized_cloud, boost::multi_array<bool, 2>& validity_grid );

} //namespace

#include "cloudNormals.ipp"

#endif //header
