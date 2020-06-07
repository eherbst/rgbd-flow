/*
 * timer: as-accurate-as-we-know-how timing with minimal coding
 *
 * Evan Herbst
 * 3 / 25 / 10
 */

#ifndef EX_RGBD_TIMER_H
#define EX_RGBD_TIMER_H

#include <ctime>
#include <iostream>
#include <stdexcept>

namespace rgbd
{

/*
 * simple wall-time reporter (ie only use it when you're single-threaded <-- EVH: actually seems fine multithreaded):
 *
 * timer t;
 * <do stuff>
 * t.stop("do stuff"); //prints "time to do stuff: 1.02s"
 *
 * t.restart();
 * <do stuff>
 * t.stop("do other stuff");
 */
class timer
{
	public:

		timer(bool verb = true) : verbose(verb)
		{
			restart();
		}

		void restart()
		{
			const int err = clock_gettime(CLOCK_REALTIME, &start);
			if(err) throw std::runtime_error("clock_gettime failed");
		}

		void stop(const std::string& desc)
		{
			timespec end;
			const int err = clock_gettime(CLOCK_REALTIME, &end);
			if(err) throw std::runtime_error("clock_gettime failed");
			if(verbose) std::cout << "time to " << desc << ": " << (((long)end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9) << "s" << std::endl;
		}

	private:

		timespec start;
		bool verbose;
};

} //namespace

#endif //header
