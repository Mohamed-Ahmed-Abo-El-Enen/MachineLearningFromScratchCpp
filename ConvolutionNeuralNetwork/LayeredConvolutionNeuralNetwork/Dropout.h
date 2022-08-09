#ifndef DROPOUT_H
#define DROPOUT_H
#include "Module.h"

class Dropout : public Module
{
private:
	double p;
	int seed;
	void DropoutCutOp(default_random_engine generator, uniform_real_distribution<> distribution, double p);

public:
	Dropout(double p = 0.5, int seed = 0);
	virtual ~Dropout() = default;
	void Compile(unique_ptr<Tensor<double>>& input) override;
	unique_ptr<Tensor<double>> ForwardPropagate(unique_ptr<Tensor<double>>& input) override;
	unique_ptr<Tensor<double>> BackwardPropagate(unique_ptr<Tensor<double>>& chainGradient, double learningRate) override;
	void load(FILE* fileModel) override;
	void save(FILE* fileModel) override;
};
#endif // !DROPOUT_H
