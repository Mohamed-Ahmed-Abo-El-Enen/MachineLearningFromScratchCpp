#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "Tensor.h"
#include <map>

using namespace std;

class Optimizer
{
protected:
	double m_LearningRate = 0.005;

public:
	Optimizer(double learningRate) { m_LearningRate = learningRate; }
	virtual void UpdateParameters(map<string, Tensor<double>>& parameters, map<string, Tensor<double>> grads) = 0;
	virtual void InitializeParams(map<string, Tensor<double>> parameters) = 0;
	virtual ~Optimizer() = default;
};

#endif // !OPTIMIZER_H
