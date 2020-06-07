/*
 * recursiveMedianFiltering: of depth maps
 *
 * Evan Herbst
 * 8 / 27 / 12
 */

#ifndef EX_RECURSIVE_MEDIAN_FILTERING_H
#define EX_RECURSIVE_MEDIAN_FILTERING_H

#include <cstdint>
#include "celiu_flow/Image.h"

namespace rgbd
{

/*
 * fill in invalid regions (values <= 0)
 *
 * windowHalfwidth: the nbrhood over which we take the median
 *
 * data: row-major and contiguous
 */
void recursiveMedianFilter(const uint32_t width, const uint32_t height, float* const data, const int32_t windowHalfwidth = 1);
void recursiveMedianFilter(const uint32_t width, const uint32_t height, double* const data, const int32_t windowHalfwidth = 1);

} //namespace

/*
 * fill in invalid regions
 *
 * windowHalfwidth: the nbrhood over which we take the median
 */
void recursiveMedianFilter(DImage& depth, const int32_t windowHalfwidth = 1);

/*
 * fill in only small invalid regions
 *
 * we'll use the first three channels of col to get a color similarity measure; each channel should be in [0, 1)
 */
void recursiveMedianFilterSmallInvalidRegionsUsingColor(DImage& depth, const DImage& col, const uint32_t maxRegionSizeToFill);

#endif //header
