/*
 * cloudNormals: computing and retrieving normals
 *
 * Evan Herbst
 * 3 / 24 / 10
 */

#include <cassert>
#include <cmath>
#include <functional>
#include <boost/unordered_map.hpp>
#include <pcl/impl/pcl_base.hpp>
#include <pcl/features/normal_3d.h>
#include <pcl/features/impl/normal_3d.hpp>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/kdtree/impl/kdtree_flann.hpp>
#include <pcl/search/impl/search.hpp>
#include <pcl/search/impl/kdtree.hpp>
#include <pcl/search/impl/organized.hpp>
#include "rgbd_util/timer.h"
#include "rgbd_util/parallelism.h" //partitionEvenly()
#include "rgbd_util/threadPool.h"
#include "rgbd_util/primesensorUtils.h"

namespace rgbd
{
using std::vector;
using std::cout;
using std::endl;
using std::pair;
using boost::unordered_map;
using boost::optional;
using Eigen::Vector3d;
using Eigen::Vector4d;

namespace normalsAux
{

template <typename PointNT>
void computePointPrincipalCurvatures(const pcl::PointCloud<PointNT> &normals, int p_idx, const std::vector<int> &indices,
      float &pcx, float &pcy, float &pcz, float &pc1, float &pc2)
{
  EIGEN_ALIGN16 Eigen::Matrix3f I = Eigen::Matrix3f::Identity ();
  Eigen::Vector3f n_idx (normals.points[p_idx].normal[0], normals.points[p_idx].normal[1], normals.points[p_idx].normal[2]);
  EIGEN_ALIGN16 Eigen::Matrix3f M = I - n_idx * n_idx.transpose ();    // projection matrix (into tangent plane)

  // Project normals into the tangent plane
  Eigen::Vector3f normal;
  std::vector<Eigen::Vector3f> projected_normals_(indices.size ());
  Eigen::Vector3f xyz_centroid_;
  xyz_centroid_.setZero ();
  for (size_t idx = 0; idx < indices.size(); ++idx)
  {
    normal[0] = normals.points[indices[idx]].normal[0];
    normal[1] = normals.points[indices[idx]].normal[1];
    normal[2] = normals.points[indices[idx]].normal[2];

    projected_normals_[idx] = M * normal;
    xyz_centroid_ += projected_normals_[idx];
  }

  // Estimate the XYZ centroid
  xyz_centroid_ /= indices.size ();

  // Initialize to 0
  EIGEN_ALIGN16 Eigen::Matrix3f covariance_matrix_;
  covariance_matrix_.setZero ();

  double demean_xy, demean_xz, demean_yz;
  // For each point in the cloud
  for (size_t idx = 0; idx < indices.size (); ++idx)
  {
	  Eigen::Vector3f demean_ = projected_normals_[idx] - xyz_centroid_;

    demean_xy = demean_[0] * demean_[1];
    demean_xz = demean_[0] * demean_[2];
    demean_yz = demean_[1] * demean_[2];

    covariance_matrix_(0, 0) += demean_[0] * demean_[0];
    covariance_matrix_(0, 1) += demean_xy;
    covariance_matrix_(0, 2) += demean_xz;

    covariance_matrix_(1, 0) += demean_xy;
    covariance_matrix_(1, 1) += demean_[1] * demean_[1];
    covariance_matrix_(1, 2) += demean_yz;

    covariance_matrix_(2, 0) += demean_xz;
    covariance_matrix_(2, 1) += demean_yz;
    covariance_matrix_(2, 2) += demean_[2] * demean_[2];
  }

  // Extract the eigenvalues and eigenvectors
  //Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> ei_symm (covariance_matrix_);
  //eigenvalues_  = ei_symm.eigenvalues ();
  //eigenvectors_ = ei_symm.eigenvectors ();
  EIGEN_ALIGN16 Eigen::Matrix3f eigenvectors_;
  Eigen::Vector3f eigenvalues_;
  pcl::eigen33 (covariance_matrix_, eigenvectors_, eigenvalues_);

  pcx = eigenvectors_ (0, 2);
  pcy = eigenvectors_ (1, 2);
  pcz = eigenvectors_ (2, 2);
  pc1 = eigenvalues_ (2);
  pc2 = eigenvalues_ (1);
}

} //namespace

template <typename PointT>
rgbd::eigen::Vector3f getNormal(pcl::PointCloud<PointT> &organizedCloud, unsigned int x, unsigned int y)
{
	const PointT& pt = organizedCloud.points[y * organizedCloud.width + x];
	return ptNormal2eigen<rgbd::eigen::Vector3f>(pt);
}

template <typename PointT>
void markNormalInvalid(PointT& pt)
{
	pt.normal[0] = pt.normal[1] = pt.normal[2] = 0;
}

template <typename PointT>
float getCurvature(pcl::PointCloud<PointT> &organizedCloud, unsigned int x, unsigned int y)
{
	return organizedCloud.points[y * organizedCloud.width + x].curvature;
}

/*
 * pre: points has width, height set and points allocated
 */
template <typename PointT>
void computeOrganizedPointCloudNormalsThreadMain(pcl::PointCloud<PointT> &points, const pcl::PointCloud<PointT> &surface,
		boost::multi_array<bool, 2>& surfaceValidity, int k, unsigned int downsample_factor, double max_z,
		const rgbd::eigen::Vector3f& viewpoint, const uint32_t minPtsToBeValid, const unsigned int firstTgtRow, const unsigned int lastTgtRow)
{
    pcl::NormalEstimation<PointT, PointT> ester;
    for(unsigned int i = firstTgtRow; i < lastTgtRow; i++) //indices into target cloud
   	 for(unsigned int j = 0; j < points.width; j++)
		 {
   		 const unsigned int index = i * points.width + j;

   		 //indices into source cloud
   		 const unsigned int si = downsample_factor * i, sj = downsample_factor * j;
   		 const unsigned int sindex = si * surface.width + sj;

			//ignore points specifically marked as invalid
			if(!surfaceValidity[sj][si]) markNormalInvalid(points.points[index]);
			else
			{
				// Get all point neighbors in a local window
				uint32_t l = 0;
				std::vector<int> nn_indices((k + k + 1) * (k + k + 1));
				for(int y = std::max(0, (int)si - k); y <= std::min((int)surface.height - 1, (int)si + k); y++) //indices into surface
					for(int x = std::max(0, (int)sj - k); x <= std::min((int)surface.width - 1, (int)sj + k); x++)
					{
						unsigned int sindex2 = y * surface.width + x;
						// If the difference in Z (depth) between the query point and the current neighbor is smaller than max_z
						if(max_z != -1)
						{
							if(fabs(surface.points[sindex].z - surface.points[sindex2].z) < max_z * primesensor::stereoErrorRatio(surface.points[sindex].z))
								nn_indices[l++] = sindex2;
						}
						else nn_indices[l++] = sindex2;
					}

				if(l < minPtsToBeValid){
					markNormalInvalid(points.points[index]); //too few neighbors for a reliable estimate
					surfaceValidity[sj][si] = false;
				}
				else
				{
					nn_indices.resize (l);
					// Compute the point normals (nx, ny, nz), surface curvature estimates (c)
					rgbd::eigen::Vector4f plane_parameters;
					float curvature;
					ester.computePointNormal (surface, nn_indices, plane_parameters, curvature);

					/*
					 * sometimes pcl gives nans; haven't looked into why -- EVH 20101203
					 */
					bool invalid = false;
					for(unsigned int m = 0; m < 4; m++)
						if(isinf(plane_parameters[m]) || isnan(plane_parameters[m]))
						{
							invalid = true;
							break;
						}
					if(!invalid && (isinf(curvature) || isnan(curvature))) invalid = true;

					if(!invalid)
					{
						pcl::flipNormalTowardsViewpoint (surface.points[sindex], viewpoint.x(), viewpoint.y(), viewpoint.z(), plane_parameters);
						points.points[index] = surface.points[sindex];
						points.points[index].normal[0] = plane_parameters[0];
						points.points[index].normal[1] = plane_parameters[1];
						points.points[index].normal[2] = plane_parameters[2];
						points.points[index].curvature = curvature;
					}
					else
					{
						markNormalInvalid(points.points[index]);
						surfaceValidity[sj][si] = false;
					}
				}
			}
		 }
}

template <typename PointT>
void computeOrganizedPointCloudNormals (
    		pcl::PointCloud<PointT> &points, const pcl::PointCloud<PointT> &surface,
    		boost::multi_array<bool, 2>& surfaceValidity,
    		unsigned int k, unsigned int downsample_factor, double max_z,
    		const rgbd::eigen::Vector3f& viewpoint, const bool multithread)
{
	// Reduce by a factor of N
	points.width = lrint (ceil (surface.width / (double)downsample_factor));
	points.height = lrint (ceil (surface.height / (double)downsample_factor));
	points.points.resize(points.width * points.height);

	const uint32_t minPtsToBeValid = std::max((uint32_t)6, (2 * k + 1) * (2 * k + 1) / 3); //TODO ?

	if(multithread)
	{
		const unsigned int numThreads = getSuggestedThreadCount();
		const vector<unsigned int> rowIndices = partitionEvenly(points.height, numThreads);
		rgbd::threadGroup tg(numThreads);
		for(unsigned int i = 0; i < numThreads; i++)
			tg.addTask([&,i]()
				{
					computeOrganizedPointCloudNormalsThreadMain(points, surface, surfaceValidity, k, downsample_factor, max_z, viewpoint, minPtsToBeValid, rowIndices[i], rowIndices[i + 1]);
				});
		tg.wait();
	}
	else
	{
		computeOrganizedPointCloudNormalsThreadMain(points, surface, surfaceValidity, k, downsample_factor, max_z, viewpoint, minPtsToBeValid, 0, points.height);
	}
}

/*
 * like computePrincipalCurvatures() but should be faster
 */
template <typename PointT>
void computePrincipalCurvaturesOrganized(const pcl::PointCloud<PointT>& organizedCloud, const int32_t nbrhoodRadiusInPixels, std::vector<float>& pc1, std::vector<float>& pc2, const bool parallelize)
{
	pc1.resize(organizedCloud.points.size());
	pc2.resize(organizedCloud.points.size());

	const auto computePrincipalCurvaturesOrganizedThreadMain = [](const pcl::PointCloud<PointT>& organizedCloud, const int32_t nbrhoodRadiusInPixels, std::vector<float>& pc1, std::vector<float>& pc2, const int32_t firstIndex, const int32_t lastIndex)
		{
			for(uint32_t l = firstIndex; l <= lastIndex; l++)
			{
				const int32_t i = l / organizedCloud.width, j = l % organizedCloud.width;
				const float z0 = organizedCloud.points[l].z;
				const float depthFactor = primesensor::stereoErrorRatio(z0);
				vector<int> indices;
				for(int32_t ii = std::max(0, i - nbrhoodRadiusInPixels); ii <= std::min((int32_t)organizedCloud.height - 1, i + nbrhoodRadiusInPixels); ii++)
					for(int32_t jj = std::max(0, j - nbrhoodRadiusInPixels); jj <= std::min((int32_t)organizedCloud.width - 1, j + nbrhoodRadiusInPixels); jj++)
					{
						const int32_t ll = ii * organizedCloud.width + jj;
						if(fabs(organizedCloud.points[ll].z - z0) < .03/* TODO parameterize */ * depthFactor)
							indices.push_back(ll);
					}
				float pcx, pcy, pcz, _pc1, _pc2;
				normalsAux::computePointPrincipalCurvatures<PointT>(organizedCloud, l, indices, pcx, pcy, pcz, _pc1, _pc2);
				pc1[l] = _pc1;
				pc2[l] = _pc2;
			}
		};

	if(parallelize)
	{
		const uint32_t numThreads = getSuggestedThreadCount();
		rgbd::threadGroup tg(numThreads);
		const vector<unsigned int> indices = partitionEvenly(organizedCloud.points.size(), numThreads);
		for(uint32_t i = 0; i < numThreads; i++)
			tg.addTask([&,i](){computePrincipalCurvaturesOrganizedThreadMain(organizedCloud, nbrhoodRadiusInPixels, pc1, pc2, indices[i], indices[i + 1] - 1);});
		tg.wait();
	}
	else
	{
		computePrincipalCurvaturesOrganizedThreadMain(organizedCloud, nbrhoodRadiusInPixels, pc1, pc2, 0, organizedCloud.points.size() - 1);
	}
}

template <typename PointT>
void getValidityGrid(const pcl::PointCloud<PointT> & organized_cloud, boost::multi_array<bool, 2>& validity_grid )
{
	validity_grid.resize(boost::extents[organized_cloud.width][organized_cloud.height]);
	for (unsigned int y = 0; y < organized_cloud.height; y++) {
		for (unsigned int x = 0; x < organized_cloud.width; x++) {
			float depth_value = organized_cloud.points[y * organized_cloud.width + x].z;
			// depth_value must be positive
			validity_grid[x][y] = (depth_value > 0.0);
		}
	}
}

/**
 * Upsamples normals for an organized cloud that only has normals where
 * x%normals_downsample = 0 and y%normals_downsample = 0
 *
 * @param organizedCloud should have height, width set
 * @param targetValidityGrid should have been allocated
 * @param normals_downsample
 */
template <typename PointT>
void interpolateNormals(pcl::PointCloud<PointT> &organizedCloud, boost::multi_array<bool, 2>& targetValidityGrid, unsigned int normals_downsample)
{
	const unsigned int inc = normals_downsample;
	if(inc == 1) return;
	ASSERT_ALWAYS(inc > 0); //we divide by it later

	/*
	* compute the first x- and y-indices for which {x,y}_floor + inc is oob
	*/
	const unsigned int xRes = organizedCloud.width, yRes = organizedCloud.height;
	unsigned int xlast = xRes - (xRes % inc), ylast = yRes - (yRes % inc);
	if(xlast == xRes) xlast -= inc;
	if(ylast == yRes) ylast -= inc;

	for(unsigned int index = 0; index<organizedCloud.points.size(); index++)
	{
		const unsigned int x = index%xRes;
		const unsigned int y = index/xRes;

		//don't bother setting a normal for a point we don't care about
		if(!targetValidityGrid[x][y])
			continue;

		const unsigned int x_resid = x%inc;
		const unsigned int y_resid = y%inc;

		//normal already set for these
		if(x_resid == 0 && y_resid ==0)
			continue;

		const unsigned int x_floor = x - x_resid;
		const unsigned int y_floor = y - y_resid;

		/*
		 * avoid indexing out of bounds
		 */
		const unsigned int xinc = (x >= xlast) ? 0 : inc,
			yinc = (y >= ylast) ? 0 : inc;

		if(targetValidityGrid[x_floor][y_floor] &&
			targetValidityGrid[x_floor+xinc][y_floor] &&
			targetValidityGrid[x_floor][y_floor+yinc] &&
			targetValidityGrid[x_floor+xinc][y_floor+yinc])
		{
			//interpolate normal
			float x_r = ((float)x_resid)/inc; //in [0, 1)
			float y_r = ((float)y_resid)/inc;

			float coefCC = x_r*y_r;
			float coefCF = x_r*(1-y_r);
			float coefFC = (1-x_r)*y_r;
			float coefFF = (1-x_r)*(1-y_r);

			rgbd::eigen::Vector3f normalSum =
				coefFF * getNormal(organizedCloud,x_floor,y_floor) +
				coefFC * getNormal(organizedCloud,x_floor,y_floor+yinc) +
				coefCF * getNormal(organizedCloud,x_floor+xinc,y_floor) +
				coefCC * getNormal(organizedCloud,x_floor+xinc,y_floor+yinc);
			if(normalSum.squaredNorm() < 1e-6) //this does happen in practice -- EVH
			{
				targetValidityGrid[x][y] = false;
			}
			else
			{
				normalSum.normalize();

				float curvatureSum =
					coefFF * getCurvature(organizedCloud,x_floor,y_floor) +
					coefFC * getCurvature(organizedCloud,x_floor,y_floor+yinc) +
					coefCF * getCurvature(organizedCloud,x_floor+xinc,y_floor) +
					coefCC * getCurvature(organizedCloud,x_floor+xinc,y_floor+yinc);

				rgbd::eigen2ptNormal(organizedCloud.points[index], normalSum);
				organizedCloud.points[index].curvature = curvatureSum;
			}
		}
		else
		{
				//allow the normal to be set if there are for instance 3 valid
				//neighbors with similar depth values -- MSK
			targetValidityGrid[x][y] = false;
		}
	}
}

template <typename PointT>
    void setOrganizedNormals(
    		pcl::PointCloud<PointT> &organizedCloud,
    		boost::multi_array<bool, 2>& targetValidityGrid,
    		unsigned int normals_window_size,
    		unsigned int normals_downsample,
    		float normals_max_z_diff,
		bool use_GPU_for_normals, const bool multithread)
    {
//    	rgbd::timer t;

    	//organized normals
    	rgbd::eigen::Vector3f origin(0, 0, 0);
    	pcl::PointCloud<PointT> withNormals;

#if ENABLE_OPENCL
	static int first_time = 1;
	static bool using_GPU_for_normals = false;

	ROS_INFO("cloudNormals: opencl enabled: about to computeOrganizedPointCloudNormals");
	if ( first_time == 1 )
	{
		first_time = 0;

		if ( use_GPU_for_normals == true )
		{
			bool ret = GPUcomputeOrganizedPointCloudNormals(withNormals, organizedCloud,normals_window_size,normals_downsample, organizedCloud.width,organizedCloud.height,normals_max_z_diff,origin);
			if ( ret == false )
			{
				ROS_INFO("cloudNormals: configured to use GPU for normals calculation, but can't execute kernel. Falling back to CPU computation.");
			}
			else
			{
				using_GPU_for_normals = true;
			}
		}
		else
		{
			//computeOrganizedPointCloudNormals(withNormals, organizedCloud,normals_window_size,normals_downsample, normals_max_z_diff,origin);
			computeOrganizedPointCloudNormals(withNormals, organizedCloud, targetValidityGrid, normals_window_size,normals_downsample, normals_max_z_diff,origin);
		}
	}
	else
	{
		if ( using_GPU_for_normals == true )
		{
			bool ret = GPUcomputeOrganizedPointCloudNormals(withNormals, organizedCloud,normals_window_size,normals_downsample, organizedCloud.width,organizedCloud.height,normals_max_z_diff,origin);
			if ( ret == false )
			{
				ROS_INFO("cloudNormals: configured to use GPU for normals calculation, but can't execute kernel. Falling back to CPU computation.");
				using_GPU_for_normals = false;
			}
		}
		else
		{
			//computeOrganizedPointCloudNormals(withNormals, organizedCloud,normals_window_size,normals_downsample, normals_max_z_diff,origin);
			computeOrganizedPointCloudNormals(withNormals, organizedCloud, targetValidityGrid, normals_window_size,normals_downsample, normals_max_z_diff,origin);
		}
	}
#else
	computeOrganizedPointCloudNormals(withNormals, organizedCloud, targetValidityGrid, normals_window_size,normals_downsample, normals_max_z_diff,origin, multithread);
#endif

    	//put the normals into organizedCloud because withNormals is the wrong dimensions

#if ENABLE_OPENCL
    	int smIncr = 1;

    	unsigned int smX, smY, lgX, lgY, lgIndex;
      for(unsigned int smIndex=0; smIndex<withNormals.points.size(); smIndex += smIncr) {
			if ( using_GPU_for_normals == true )
			{
				// GPU computation calculates all 307200 points, so we need to simulate any
				// requested downsampling here
				smX = smIndex%(withNormals.width*normals_downsample);
				smY = smIndex/(withNormals.width*normals_downsample);
				if ( smX % normals_downsample != 0 ) continue;
				if ( smY % normals_downsample != 0 ) continue;
				lgX = smX;
				lgY = smY;
				lgIndex = smIndex;
			}
			else
			{
				smX = smIndex%withNormals.width;
				smY = smIndex/withNormals.width;
				lgX = smX*normals_downsample;
				lgY = smY*normals_downsample;
				lgIndex = lgX + lgY*organizedCloud.width;
			}

		//ROS_INFO("smX,Y=%d,%d smIndex=%d lgX,Y=%d,%d lgIndex=%d", smX, smY, smIndex, lgX, lgY, lgIndex );
#else
		for(unsigned int smY = 0; smY < withNormals.height; smY++)
			for(unsigned int smX = 0; smX < withNormals.width; smX++)
			{
				const unsigned int smIndex = smY * withNormals.width + smX;
				unsigned int lgX = smX*normals_downsample;
				unsigned int lgY = smY*normals_downsample;
				unsigned int lgIndex = lgX + lgY*organizedCloud.width;
#endif
				if(targetValidityGrid[lgX][lgY])
				{
					organizedCloud.points[lgIndex].normal[0] = withNormals.points[smIndex].normal[0];
					organizedCloud.points[lgIndex].normal[1] = withNormals.points[smIndex].normal[1];
					organizedCloud.points[lgIndex].normal[2] = withNormals.points[smIndex].normal[2];
					organizedCloud.points[lgIndex].curvature = withNormals.points[smIndex].curvature;
				}
			}

    	interpolateNormals(organizedCloud, targetValidityGrid, normals_downsample);

//    	t.stop("setOrganizedNormals");
    }

} //namespace
