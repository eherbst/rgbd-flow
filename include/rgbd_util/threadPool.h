/*
 * threadPool: allocate threads once and use them repeatedly, to reduce multithreading overhead
 *
 * Evan Herbst
 * 1 / 23 / 12
 */

#ifndef EX_RGBD_THREADPOOL_H
#define EX_RGBD_THREADPOOL_H

#include <vector>
#include <memory>
#include <boost/thread.hpp>
#include <boost/bind.hpp>
#include <boost/asio/io_service.hpp>

namespace rgbd
{

/*
 * usage pattern:
 * ---
 * threadGroup tg(NTHREADS);
 * for(i = 0; i < NTHREADS; i++) tg.addTask([i](){run(i);});
 * tg.wait();
 * for(i = 0; i < NTHREADS; i++) useResult(i);
 * ---
 *
 * 20120521: should now be thread-safe (will run all tasks in a single thread unless this object owns the class-global thread set)
 */
class threadGroup
{
	public:

		threadGroup(const uint32_t numThreads)
		{
			havePool = mux.try_lock();
			if(havePool)
			{
				if(!io_service)
				{
					io_service.reset(new boost::asio::io_service);
					work.reset(new boost::asio::io_service::work(*io_service));
				}
				//add threads until we have enough
				const size_t curNumThreads = threads.size();
				for(size_t i = curNumThreads; i < numThreads; i++)
					threads.create_thread(boost::bind(&boost::asio::io_service::run, io_service.get()));
			}
		}

		~threadGroup()
		{
//			io_service.stop(); //TODO will need to call this somewhere?
			if(havePool) mux.unlock();
		}

		/*
		 * to be called from the main thread
		 */
		template <typename Func>
		void addTask(Func&& f)
		{
			if(havePool) //add to task queue
			{
				tasks.push_back(std::shared_ptr<boost::packaged_task<void>>(new boost::packaged_task<void>(std::move(f))));
				futures.push_back(tasks.back()->get_future());
				io_service->post(boost::bind(&boost::packaged_task<void>::operator (), tasks.back().get())); //could use the lambda [t](){(*t)();} instead of a bind() here if I copy tasks.back() to a variable t first
			}
			else //run immediately in our single thread
			{
				f();
			}
		}

		/*
		 * to be called from the main thread
		 */
		void wait()
		{
			if(havePool)
			{
				boost::wait_for_all(futures.begin(), futures.end());
				futures.clear();
				tasks.clear();
			}
		}

	private:

		static boost::mutex mux;
		static std::shared_ptr<boost::asio::io_service> io_service;
		static std::shared_ptr<boost::asio::io_service::work> work;
		static boost::thread_group threads;

		bool havePool; //do we currently own the thread set?
		std::vector<std::shared_ptr<boost::packaged_task<void>>> tasks; //could remove the ptrs if I preallocate the vector instead of push_back()ing
		std::vector<boost::unique_future<void>> futures;
};

} //namespace

#endif //header
