#ifndef RELU_H
#define RELU_H

#include "Module.h"

class Relu : public Module
{
private:
	Tensor<double> relu();
	Tensor<double> reluGradient();

public:
	Relu();
	virtual ~Relu() = default;
	void Compile(unique_ptr<Tensor<double>>& input) override;
	unique_ptr<Tensor<double>> ForwardPropagate(unique_ptr<Tensor<double>>& input) override;
	unique_ptr<Tensor<double>> BackwardPropagate(unique_ptr<Tensor<double>>& chainGradient, double learningRate) override;
	void load(FILE* fileModel) override;
	void save(FILE* fileModel) override;
};

#endif // !RELU_H
