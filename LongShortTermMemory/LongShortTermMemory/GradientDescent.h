#ifndef GRADIENTDESCENT_H
#define GRADIENTDESCENT_H

#include "Optimizer.h"

using namespace std;

class GradientDescent : public Optimizer
{

public:
	GradientDescent() = default;
	GradientDescent(double learningRate);
	void UpdateParameters(map<string, Tensor<double>>& parameters, map<string, Tensor<double>> grads) override;
	void InitializeParams(map<string, Tensor<double>> parameters) override;
};

#endif // !GRADIENTDESCENT_H
