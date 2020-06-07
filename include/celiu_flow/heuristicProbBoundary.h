/*
 * heuristicProbBoundary: cues for spatial regularization weights
 *
 * Evan Herbst
 * 8 / 29 / 12
 */

#ifndef EX_OPTICAL_FLOW_HEURISTIC_PB_H
#define EX_OPTICAL_FLOW_HEURISTIC_PB_H

#include <boost/multi_array.hpp>
#include "celiu_flow/Image.h"

/*
 * not Malik's Pb, but a very heuristic rough boundary-prob map to identify all possible boundary pts in an rgbd frame
 *
 * img: first 3 channels should be rgb
 *
 * depth should not have been renormalized (eg to [0, 1])
 *
 * return: p(boundary) for each pixel pair in row-major order
 * (first two output dimensions are y, x; third is [right, down] nbrs)
 */
boost::multi_array<float, 3> computeHeuristicPb(const DImage& img, const DImage& depth);

#endif //header
