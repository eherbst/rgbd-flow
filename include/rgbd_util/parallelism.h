/*
 * parallelism: making multithreading easier
 *
 * Evan Herbst
 * 3 / 25 / 10
 */

#ifndef EX_RGBD_PARALLELISM_H
#define EX_RGBD_PARALLELISM_H

#if __GNUC__ == 4 && __GNUC_MINOR__ == 5 && defined(__GXX_EXPERIMENTAL_CXX0X__)
//use std thread because boost.thread 1.42 is broken under gcc 4.5 c++0x mode
#elif __GNUC__ == 4 && __GNUC_MINOR__ > 5 && defined(__GXX_EXPERIMENTAL_CXX0X__)
//use std thread because it should work (what does this mean? the library isn't fully implemented yet -- EVH 20120130)
#else
#define USE_BOOST_THREAD
#endif

#ifndef USE_BOOST_THREAD
#include <thread>
#include <functional>
namespace rgbd
{
	using std::thread;
	using std::ref;
	using std::cref;
	using std::bind;
}
#else
#include <boost/ref.hpp>
#include <boost/thread.hpp>
#include <boost/bind.hpp>
namespace rgbd
{
	using boost::thread;
	using boost::ref;
	using boost::cref;
	using boost::bind;
}
#endif

/*******************************************************************************/

#include <vector>

/*
 * divide the range [0, numItems) more or less evenly into numPartitions parts and return the start index of each subsequence, plus an element equal to numItems
 * (ie, each jth partition's index range is [result[j], result[j + 1]) )
 *
 * numItems can be 0; numPartitions cannot
 */
std::vector<unsigned int> partitionEvenly(const unsigned int numItems, const unsigned int numPartitions);

/*
 * Get a suggested number of threads based on hardware concurrency
 *
 * @param min_threads  The minimum value returned, regardless of system (minimum is always non-zero regardless of this value)
 * @param remaining_threads  Subtracted from hardware_concurrency (leave threads left over for other tasks)
 */
unsigned int getSuggestedThreadCount(unsigned int min_threads = 2, unsigned int remaining_threads = 0);

#endif //header
