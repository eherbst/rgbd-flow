/*
 * parallelism: making multithreading easier
 *
 * Evan Herbst
 * 3 / 25 / 10
 */

#include <cassert>
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include "rgbd_util/assert.h"
#include "rgbd_util/parallelism.h"
using std::vector;
using std::cout;
using std::endl;

/*
 * divide the range [0, numItems) more or less evenly into numPartitions parts and return the start index of each subsequence, plus an element equal to numItems
 * (ie, each jth partition's index range is [result[j], result[j + 1]) )
 *
 * numItems can be 0; numPartitions cannot
 */
vector<unsigned int> partitionEvenly(const unsigned int numItems, const unsigned int numPartitions)
{
	vector<unsigned int> indices(numPartitions + 1);
	indices[0] = 0;
	const unsigned int quotient = numItems / numPartitions, remainder = numItems - numPartitions * quotient;
	for(unsigned int i = 1; i < numPartitions; i++) indices[i] = indices[i - 1] + quotient + (remainder >= i); //divide leftovers evenly
	indices[numPartitions] = numItems;
	return indices;
}


unsigned int getSuggestedThreadCount(unsigned int min_threads, unsigned int remaining_threads)
{
	if (min_threads < 1) min_threads = 1;
#ifdef USE_BOOST_THREAD
	const int nthreads = boost::thread::hardware_concurrency();
#else
	/*
	 * EVH 20111201: std::thread::hardware_concurrency() returns 0 in gcc 4.5 and 4.6; /proc/cpuinfo is the most portable alternative
	 */
	static int previous_nthreads = 0; //keep from having to run the command multiple times (although this is not thread-safe -- TODO don't use static?)
	if(previous_nthreads == 0)
	{
		//use popen() rather than system() to avoid writing a file
		FILE* fp = popen("cat /proc/cpuinfo | grep processor | wc -l", "r");
		fscanf(fp, "%d", &previous_nthreads);
		ASSERT_ALWAYS(pclose(fp) != -1);
		ASSERT_ALWAYS(previous_nthreads >= 1);
	}
	const int nthreads = previous_nthreads;
#endif
	return (unsigned int) std::max( (int)min_threads, nthreads - (int)remaining_threads);
}
