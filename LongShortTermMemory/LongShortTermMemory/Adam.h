#ifndef ADAM_H
#define ADAM_H

#include "Optimizer.h"

using namespace std;

class Adam : public Optimizer
{
private:
	double m_Beta1;
	double m_Beta2;
	map<string, Tensor<double>> V;
	map<string, Tensor<double>> S;

	void InitializeV(map<string, Tensor<double>> parameters);
	void InitializeS(map<string, Tensor<double>> parameters);
public:
	Adam() = default;
	Adam(double beta1, double beta2, double learningRate);
	void UpdateParameters(map<string, Tensor<double>>& parameters, map<string, Tensor<double>> grads) override;
	void InitializeParams(map<string, Tensor<double>> parameters) override;
};

#endif // !ADAM_H
