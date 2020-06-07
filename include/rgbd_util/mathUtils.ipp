/*
 * mathUtils
 *
 * Evan Herbst
 * 3 / 3 / 10
 */

#include <cstdlib>

template <typename T>
T clamp(const T x, const T min, const T max)
{
	return std::min(max, std::max(min, x));
}
