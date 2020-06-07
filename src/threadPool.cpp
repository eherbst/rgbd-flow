/*
 * threadPool: allocate threads once and use them repeatedly, to reduce multithreading overhead
 *
 * Evan Herbst
 * 1 / 23 / 12
 */

#include "rgbd_util/threadPool.h"

namespace rgbd
{

boost::mutex threadGroup::mux;
std::shared_ptr<boost::asio::io_service> threadGroup::io_service;
std::shared_ptr<boost::asio::io_service::work> threadGroup::work;
boost::thread_group threadGroup::threads;

} //namespace
