#ifndef LINEARLRSCHEDULER_H
#define LINEARLRSCHEDULER_H
#include "LRScheduler.h"

class LinearLRScheduler : public LRScheduler
{
public:
	double step;
	LinearLRScheduler(double initialLearningRate, double step);
	virtual ~LinearLRScheduler() = default;
	void OnIterationEnd(int iterations) override;
};
#endif // !LINEARLRSCHEDULER_H
