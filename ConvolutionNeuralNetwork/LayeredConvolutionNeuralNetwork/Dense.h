#ifndef DENSE_H
#define DENSE_H

#include "Module.h"
/*
	Dense Layer
	Wx+b
*/
class Dense :public Module
{
private:
	unique_ptr<Tensor<double>> weights;
	unique_ptr<Tensor<double>> bias;
	int inputDims[4];
	int inputNumDims;

public:
	Dense(int inputSize, int outputSize, int seed = 0);
	virtual ~Dense() = default;

	void Compile(unique_ptr<Tensor<double>>& input) override;
	unique_ptr<Tensor<double>> ForwardPropagate(unique_ptr<Tensor<double>>& input) override;
	unique_ptr<Tensor<double>> BackwardPropagate(unique_ptr<Tensor<double>>& chainGradient, double learningRate) override;
	void load(FILE* fileModel) override;
	void save(FILE* fileModel) override;
};

#endif // !DENSE_H
