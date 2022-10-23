#ifndef THREADSPOOL_H
#define THREADSPOOL_H
#pragma once

#include <vector>
#include <thread>
#include <future>
#include <queue>
#include <condition_variable>

using namespace std;

class ThreadPool
{
private:
	// need to keep track of threads so we can join them
	vector<thread> workers;
	queue<function<void()>> tasks;
	mutex queue_mutex;
	condition_variable condition;
	bool stop;
public:
	ThreadPool(size_t);
	template<class F, class... Args>
	auto enqueue(F&& f, Args&&... args)->future<typename result_of<F(Args...)>::type>;
	~ThreadPool();
};

#include "ThreadsPool.h"

inline ThreadPool::ThreadPool(size_t threads) :stop(false)
{
	for (size_t i = 0; i < threads; i++)
		workers.emplace_back([this]
			{
				for (;;)
				{
					function<void()> task;
					{
						unique_lock<mutex> lock(this->queue_mutex);
						this->condition.wait(lock, [this]
							{
								return this->stop || !this->tasks.empty();
							});
						if (this->stop && this->tasks.empty())
							return;
						task = move(this->tasks.front());
						this->tasks.pop();
					}
					task();
				}
			});
}

// add new work item to the pool
template<class F, class... Args>
auto ThreadPool::enqueue(F&& f, Args&&... args) -> future<typename result_of<F(Args...)>::type>
{
	using return_type = typename result_of<F(Args...)>::type;

	auto task = make_shared<packaged_task<return_type()>>(bind(forward<F>(f), forward<Args>(args)...));

	future<return_type> res = task->get_future();
	{
		unique_lock<mutex> lock(queue_mutex);

		// don't allow enqueueing after stopping the pool
		if (stop)
			throw runtime_error("enqueue on stopped ThreadPool");

		tasks.emplace([task]() { (*task)(); });
	}
	condition.notify_one();
	return res;
}

// the destructor joins all threads
inline ThreadPool::~ThreadPool()
{
	{
		unique_lock<mutex> lock(queue_mutex);
		stop = true;
	}
	condition.notify_all();
	for (thread& worker : workers)
		worker.join();
}

#endif //THREADSPOOL_H
