#include "Softmax.h"

Softmax::Softmax()
{
	costFunctionName = "Softmax";
}

void Softmax::Compile(unique_ptr<Tensor<double>>& input)
{
	for (int i = 0; i < 2; i++)
		standardOutputTensorDims.push_back(input->m_Dims[i]);
}

unique_ptr<Tensor<double>> Softmax::GetSoftmaxProbabilities(unique_ptr<Tensor<double>>& input)
{
	assert(input->numDims == 2);
	int rows = input->m_Dims[0];
	int cols = input->m_Dims[1];
	unique_ptr<Tensor<double>> probabilities = make_unique<Tensor<double>>(2, input->m_Dims);
	for (int i = 0; i < rows; i++)
	{
		double rowMax = -1;
		for (int j = 0; j < cols; j++)
		{
			if (j == 0 || input->get(i, j) > rowMax)
				rowMax = input->get(i, j);
		}

		double denominator = 0;
		for (int j = 0; j < cols; j++)
		{
			double x = input->get(i, j);
			denominator += exp(input->get(i, j) - rowMax);
		}

		for (int j = 0; j < cols; j++)
			probabilities->set(exp(input->get(i, j) - rowMax) / denominator, i, j);
	}
	return probabilities;
}

pair<double, unique_ptr<Tensor<double>>> Softmax::BackwardPropagate(unique_ptr<vector<int>>& yTrue)
{
	double loss = CrossEntropy(m_Output, *yTrue);

	return make_pair(loss, make_unique<Tensor<double>>(CrossEntropyPrime(m_Output, *yTrue)));
}

unique_ptr<Tensor<double>> Softmax::predict(unique_ptr<Tensor<double>>& input)
{
	m_Output = move(GetSoftmaxProbabilities(input));
	return make_unique<Tensor<double>>(*m_Output);
}

double Softmax::CrossEntropy(unique_ptr<Tensor<double>>& yHat, vector<int>& yTrue)
{
	double total = 0;
	for (int i = 0; i < yTrue.size(); i++)
	{
		double x = yHat->get(i, yTrue[i]);
		// Sets a minimum value to prevent division by zero (log(0))
		total += -log(x < epslion ? epslion : x);
	}
	return total / yTrue.size();
}

unique_ptr<Tensor<double>> Softmax::CrossEntropyPrime(unique_ptr<Tensor<double>>& yHat, vector<int> yTrue)
{
	unique_ptr<Tensor<double>> prime = make_unique<Tensor<double>>(yHat);
	for (int i = 0; i < yTrue.size(); i++)
		prime->set(yHat->get(i, yTrue[i]) - 1, i, yTrue[i]);
	
	return make_unique<Tensor<double>>(*prime / yHat->m_Dims[0]);
}

