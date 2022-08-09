#ifndef MAXPOOL_H
#define MAXPOOL_H

#include "Module.h"

class MaxPool : public Module
{
private:
	unique_ptr<Tensor<int>> m_Indexes;
	int m_Stride;
	int m_Size;

public:
	MaxPool(int size, int stride);
	virtual ~MaxPool() = default;
	void Compile(unique_ptr<Tensor<double>>& input) override;
	unique_ptr<Tensor<double>> ForwardPropagate(unique_ptr<Tensor<double>>& input) override;
	unique_ptr<Tensor<double>> BackwardPropagate(unique_ptr<Tensor<double>>& chainGradient, double learningRate) override;
	void load(FILE* fileModel) override;
	void save(FILE* fileModel) override;
};

#endif // !MAXPOOL_H
