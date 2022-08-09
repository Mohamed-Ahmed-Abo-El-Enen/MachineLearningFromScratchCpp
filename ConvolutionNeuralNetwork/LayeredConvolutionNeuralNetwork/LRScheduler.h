#ifndef LRSCHEDULER_H 
#define LRSCHEDULER_H

class LRScheduler
{
public:
	double learningRate;
	virtual void OnIterationEnd(int iteraation) = 0;
	virtual ~LRScheduler() = default;
};
#endif // !LRSCHEDULER_H 
