/*
 * opticalFlowUtils: non-main functions, separated out so CUDA can see these but not main functions using c++0x features
 *
 * Evan Herbst
 * 10 / 29 / 12
 */

#ifndef EX_CELIU_OPTICAL_FLOW_UTILS_H
#define EX_CELIU_OPTICAL_FLOW_UTILS_H

#include <opencv2/core/core.hpp>
#include "celiu_flow/Image.h"
#include "celiu_flow/NoiseModel.h"

namespace scene_flow
{

void getDxs(DImage& imdx,DImage& imdy,DImage& imdt,const DImage& im1,const DImage& im2);

/***************************************************************************************************************************************************************************************/

void warpFL(DImage& warpIm2,const DImage& Im1,const DImage& Im2,const DImage& vx,const DImage& vy);
void warpFL(DImage& warpIm2,const DImage& Im1,const DImage& Im2,const DImage& flow);
void warpBy3DFlow(const DImage& Im1, const DImage& dm1, const DImage& Im2, const DImage& dm2, DImage& warpIm2, DImage& warpDm2, const DImage& vx, const DImage& vy, const DImage& vz);

/***************************************************************************************************************************************************************************************/

void estGaussianMixture(const DImage& Im1,const DImage& Im2,GaussianMixture& para,double prior = 0.9);
void estLaplacianNoise(const DImage& Im1,const DImage& Im2,Vector<double>& para);

void weightedLaplacian(DImage& output,const DImage& input,const DImage& weight);
//added by EVH: include the inhomogeneous second partials, in Hessian row-major order
void hessian(cv::Mat& output, const DImage &input, const DImage& weight);
/*
 * normWeight gets multiplied into numerator and denominator
 */
void Laplacian(DImage &output, const DImage &input, const DImage& normWeights);
/*
 * nonnormWeight gets multiplied in; normWeight gets multiplied into numerator and denominator
 *
 * normWeights: 2-channel, for right and down pixel nbrs
 */
void Laplacian(DImage &output, const DImage &input, const DImage& nonnormWeights, const DImage& normWeights);

} //namespace

#endif //header
