#ifndef CONV2D_H
#define CONV2D_H

#include "Module.h"
#include "Utils.h"

class Conv2D : public Module
{
private:
	int stride, padding;
public:
	unique_ptr<Tensor<double>> kernels;
	unique_ptr<Tensor<double>> bias;

	Conv2D(FilterShape filters, MatShape kernelSize, int stride, int padding, int seed = 0);
	virtual ~Conv2D() = default;

	void Compile(unique_ptr<Tensor<double>>& input) override;
	unique_ptr<Tensor<double>> ForwardPropagate(unique_ptr<Tensor<double>>& input) override;
	unique_ptr<Tensor<double>> BackwardPropagate(unique_ptr<Tensor<double>>& chainGradient, double learningRate) override;
	void load(FILE* fileModel) override;
	void save(FILE* fileModel) override;
};
#endif // !CONV2D_H
