/*
 * mathUtils
 *
 * Evan Herbst
 * 3 / 3 / 10
 */

#include "rgbd_util/mathUtils.h"

/*
 * absolute value
 */
double dabs(double d)
{
	return (d < 0) ? -d : d;
}

/*
 * linear interpolation
 *
 * alpha: in [0, 1]
 */
double linterp(const double v0, const double v1, const double alpha)
{
	return (1 - alpha) * v0 + alpha * v1;
}

