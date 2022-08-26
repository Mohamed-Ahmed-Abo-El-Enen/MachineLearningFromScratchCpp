#ifndef LSTM_H
#define LSTM_H

#include <string>
#include <map>
#include "Tensor.h"
#include "Optimizer.h"
#include "Utils.h"

using namespace std;

class LSTM 
{
private:
	Optimizer* m_Optimizer;
	int seed = 0;
	double mean = 0;
	double std = 0.1;
	default_random_engine generator;

	//number of input neurons
	int m_InputUnits;

	//number of hidden neurons
	int m_HiddenUnits; 

	//number of output units i.e vocab size
	int m_OutputUnits;

	bool m_HaveEmbedding;

	map<string, Tensor<double>> parameters;

	map<string, vector<Tensor<double>>> updatedParameters;

	map<string, Tensor<double>> grads;
	
	void ForwardPropagation(Tensor<double>& sample, Tensor<double> h, Tensor<double> c);
	double BackwardPropagation(Tensor<double>& targets);
	//void ClipGradientNorm(double maxNorm = 0.25);
	void UpdateParameters();
public:
	LSTM(int inputUnits, int hiddenUnits, int outputUnits, Optimizer* optimizer, bool haveEmbedding=false);
	void Train(Tensor<double>& X_tensor, Tensor<double>& y_tensor, int epochs);
	vector<vector<Tensor<double>>> Predict(Tensor<double>& X_tensor);
};

#endif // !LSTM_H
