#ifndef Embedding_H
#define Embedding_H

#include <map>
#include "Tensor.h"
#include "Optimizer.h"
#include "Utils.h"

using namespace std;

class Embedding
{
private:
	int seed = 0;
	double mean = 0;
	double std = 1;
	default_random_engine generator;
	int m_VocabSize;
	int m_NumEmbedding;
	int m_WindowSize;

	Optimizer* m_Optimizer;

	map<string, Tensor<double>> cache;

	map<string, Tensor<double>> parameters;

	map<string, Tensor<double>> grads;

	void ForwardPropagation(Tensor<double> X);
	double BackwardPropagation(Tensor<double> X, Tensor<double> y);
	double CrossEntropy(Tensor<double> z, Tensor<double> y);
	void MakeXYWindow(Tensor<double>& X_tensor, Tensor<double>& X, Tensor<double>& y);
	void UpdateParameters();
public:
	Embedding() = default;
	Embedding(int vocabSize, int numEmbedding, int windowSize, Optimizer* optimizer);
	void Train(Tensor<double>& X_tensor, int epochs);
	Tensor<double> GetEmbedding(Tensor<double>& X_tensor);
};

#endif // !Embedding_H
