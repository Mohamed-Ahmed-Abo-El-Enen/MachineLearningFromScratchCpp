#ifndef RNN_H
#define RNN_H

#include <string>
#include <map>
#include "Tensor.h"
#include "Optimizer.h"
#include "Utils.h"

using namespace std;

class RNN
{
private:
	Optimizer* m_Optimizer;

	int seed = 0;
	double mean = 0;
	double std = 0.1;
	default_random_engine generator;

	//number of hidden neurons
	int m_HiddenUnits;

	//number of output units i.e vocab size
	int m_OutputUnits;

	map<string, Tensor<double>> parameters;

	map<string, vector<Tensor<double>>> updatedParameters;

	map<string, Tensor<double>> grads;

	void ForwardPropagation(Tensor<double>& sample, Tensor<double> hiddenState);
	double BackwardPropagation(Tensor<double>& inputs, Tensor<double>& targets);
	void ClipGradientNorm(double maxNorm=0.25);
	void UpdateParameters();

public:
	RNN(int hiddenUnits, int outputUnits, Optimizer* optimizer);
	void Train(Tensor<double>& X_tensor, Tensor<double>& y_tensor, int epochs);
	vector<vector<Tensor<double>>> Predict(Tensor<double>& X_tensor);
};

#endif // !RNN_H
