#include "LinearLRScheduler.h"

LinearLRScheduler::LinearLRScheduler(double initialLearningRate, double step)
{
	this->learningRate = initialLearningRate;
	this->step = step;
}
void LinearLRScheduler::OnIterationEnd(int iterations)
{
	learningRate += step;
}