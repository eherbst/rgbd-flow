cmake_minimum_required(VERSION 2.6)

add_definitions(-std=c++0x -Wall -O2 -g -fPIC -pthread)
include_directories(${PROJECT_SOURCE_DIR}/include)

find_package(Boost 1.49 COMPONENTS system filesystem program_options REQUIRED)
include_directories(${Boost_INCLUDE_DIR})

find_package(Eigen)
include_directories(${EIGEN_INCLUDE_DIR})

find_package(PCL)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

find_package(OpenCV)

FIND_PACKAGE(CUDA)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -arch=sm_20 -Xcompiler -fPIC -Xcompiler -mno-sse2 -Xcompiler -O2)#-g -G are needed if using cuda-gdb; -mno-sse2 seems necessary always

#set the default path for built executables to the "bin" directory
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
#set the default path for built libraries to the "lib" directory
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

#################################################################
# libs

add_library(rgbd_util
				src/assert.cpp
				src/mathUtils.cpp
				src/parallelism.cpp
				src/primesensorUtils.cpp
				src/threadPool.cpp
				)
				
add_library(pcl_rgbd
				src/pointTypes.cpp
				)
				
add_library(primesensor_tools
				src/primeSense_quantization_v5.cpp
				)
				
add_library(image_features
				src/halfDiskFeatures.cpp
				)
							
cuda_add_library(rgbd_cuda_util_cu SHARED
						src/cudaUtils.cu
						src/cudaUtils.cpp
						)
						
cuda_add_library(scene_flow_jacobi_cuda SHARED
						src/sceneFlowSingleScaleJacobiCUDA.cu
						)
target_link_libraries(scene_flow_jacobi_cuda rgbd_cuda_util_cu)

add_library(rgbd_frame_common
				src/staticDepthNoiseModeling.cpp
				)

add_library(boundary_probs_heuristic
							src/heuristicBoundaries.cpp
							)
							
add_library(optical_flow_utils
				src/middleburyFlowIO.cpp
				src/sceneFlowIO.cpp
				)

add_library(rgbd_flow
							src/ceLiuOpticalFlow.cpp
							src/depthMapPyramid.cpp
							src/GaussianPyramid.cpp
							src/heuristicProbBoundary.cpp
							src/OpticalFlow.cpp
							src/opticalFlowUtils.cpp
							src/recursiveMedianFiltering.cpp
							src/rgbdFrameUtils.cpp
							src/runFlowTwoFrames.cpp
							src/Stochastic.cpp
							)
target_link_libraries(rgbd_flow scene_flow_jacobi_cuda boundary_probs_heuristic rgbd_frame_common optical_flow_utils pcl_rgbd image_features primesensor_tools rgbd_util)

#################################################################
# drivers

#run your choice of flow algorithm
add_executable(rgbdFlowDemo src/main.cpp)
target_link_libraries(rgbdFlowDemo rgbd_flow -lpthread opencv_highgui opencv_imgproc opencv_core ${PCL_LIBRARIES} ${Boost_LIBRARIES})
