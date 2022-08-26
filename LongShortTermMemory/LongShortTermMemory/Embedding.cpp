#include "Embedding.h"

Embedding::Embedding(int vocabSize, int numEmbedding, int windowSize, Optimizer* optimizer)
{
	m_Optimizer = optimizer;
	m_VocabSize = vocabSize;
	m_NumEmbedding = numEmbedding;
	m_WindowSize = windowSize;

	normal_distribution<double> distribution(mean, std);

	int dimsW1[] = { vocabSize, numEmbedding };
	Tensor<double> W1(2, dimsW1);
	generator.seed(seed);
	//W1.randn(generator, distribution, 1);
	for (int i = 0; i < W1.GetSize(); i++)
		W1.SetListIndex((i + 1) * 0.1, i);
	

	int dimsW2[] = { numEmbedding, vocabSize };
	Tensor<double> W2(2, dimsW2);
	generator.seed(seed+1);
	//W2.randn(generator, distribution, 1);
	for (int i = 0; i < W2.GetSize(); i++)
		W2.SetListIndex((i + 1) * 0.05, i);

	parameters["W1"] = W1;
	parameters["W2"] = W2;
}

void Embedding::ForwardPropagation(Tensor<double> X)
{
	cache["A1"] = X.matmul(parameters["W1"]);
	cache["A2"] = cache["A1"].matmul(parameters["W2"]);
	cache["Z"] = cache["A2"].transpose();
	cache["Z"] = ActivationFunctions::Softmax(cache["Z"]).transpose();
}

double Embedding::BackwardPropagation(Tensor<double> X, Tensor<double> y)
{
	Tensor<double> dA2 = cache["Z"] - y;
	Tensor<double> W2_d = cache["A1"].transpose().matmul(dA2);

	Tensor<double> dA1 = dA2.matmul(parameters["W2"].transpose());
	Tensor<double> W1_d = X.transpose().matmul(dA1);

	grads["W1_d"] = W1_d;
	grads["W2_d"] = W2_d;

	double loss = CrossEntropy(cache["Z"], y);

	return loss;
}

void Embedding::UpdateParameters()
{
	m_Optimizer->UpdateParameters(parameters, grads);
}

double Embedding::CrossEntropy(Tensor<double> z, Tensor<double> y)
{
	return -1 * (z.log() * y).sum();
}

void Embedding::MakeXYWindow(Tensor<double>& X_tensor, Tensor<double>& X, Tensor<double>& y)
{
	vector<vector<Tensor<double>>>Xvec;
	vector<vector<Tensor<double>>>yvec;

	for (int i = 0; i < X_tensor.m_Dims[0]; i++)
	{
		Tensor<double> sample = X_tensor[i];

		vector<Tensor<double>> Xsample;
		vector<Tensor<double>> ysample;

		for (int j = 0; j < sample.m_Dims[0]; j++)
		{
			vector<int> windowIdx;

			for (int idx = max(0, (j - m_WindowSize)); idx < j; idx++)
				windowIdx.push_back(idx);

			for (int idx = j; idx < min(sample.m_Dims[0], (j + m_WindowSize + 1)); idx++)
				windowIdx.push_back(idx);

			for (int k = 0; k < windowIdx.size(); k++)
			{
				if (j == windowIdx[k])
					continue;

				Xsample.push_back(sample[j]);
				ysample.push_back(sample[k]);
			}
		}
		Xvec.push_back(Xsample);
		yvec.push_back(ysample);
	}	

	int dimsX[] = { Xvec.size(), Xvec[0].size(), m_VocabSize };
	X = Tensor<double>(3, dimsX);

	int dimsy[] = { yvec.size(), yvec[0].size(), m_VocabSize };
	y = Tensor<double>(3, dimsy);

	for (size_t i = 0; i < X.m_Dims[0]; i++)
	{
		for (size_t  j= 0; j < X.m_Dims[1]; j++)
		{
			for (size_t k = 0; k < X.m_Dims[2]; k++)
			{
				X.set(Xvec[i][j].get(k), i, j, k);
				y.set(yvec[i][j].get(k), i, j, k);
			}
		}
	}
}

Tensor<double> Embedding::GetEmbedding(Tensor<double>& X_tensor)
{
	int dimsResults[] = { X_tensor.m_Dims[0], X_tensor.m_Dims[1], m_NumEmbedding };
	Tensor<double> results(3, dimsResults);
	for (size_t i = 0; i < X_tensor.m_Dims[0]; i++)
	{
		Tensor<double> res = X_tensor[i].matmul(parameters["W1"]);
		for (size_t j = 0; j < res.m_Dims[0]; j++)
			for (size_t k = 0; k < res.m_Dims[1]; k++)
				results.set(res.get(j, k), i, j, k);
	}
	return results;
}


void Embedding::Train(Tensor<double>& X_tensor, int epochs)
{
	vector<double> trainLoss;
	Tensor<double> X;
	Tensor<double> y;
	MakeXYWindow(X_tensor, X, y);

	m_Optimizer->InitializeParams(parameters);

	for (size_t itr = 0; itr < epochs; itr++)
	{
		double epochTrainLoss = 0;
		for (size_t i = 0; i < X.m_Dims[0]; i++)
		{
			ForwardPropagation(X[i]);
			epochTrainLoss += BackwardPropagation(X[i], y[i]);		
			UpdateParameters();
		}		
		trainLoss.push_back(epochTrainLoss);

		if (itr % 5 == 0)
			cout << "Epoch " << itr << ", training loss: " << trainLoss[itr] << endl;
	}
}