#include "LeakyRelu.h"

LeakyRelu::LeakyRelu()
{
	layerName = "LeakyRelu";
}

Tensor<double> LeakyRelu::LeakyReluFunc()
{
	Tensor<double> result(m_Input->numDims, m_Input->m_Dims);
	for (int i = 0; i < m_Input->GetSize(); i++)
	{
		double x = m_Input->GetListIndex(i);
		double value = x > 0 ? x : negativeSlope * x;
		result.SetListIndex(value, i);
	}
	return result;
}

Tensor<double> LeakyRelu::LeakyReluGradient()
{
	Tensor<double> result(m_Input->numDims, m_Input->m_Dims);
	for (int i = 0; i < m_Input->GetSize(); i++)
	{
		double x = m_Input->GetListIndex(i);
		double value = x > 0 ? 1 : negativeSlope;
		result.SetListIndex(value, i);
	}
	return result;
}

void LeakyRelu::Compile(unique_ptr<Tensor<double>>& input)
{
	for (int i = 0; i < input->numDims; i++)
		standardInputTensorDims.push_back(input->m_Dims[i]);

	for (int i = 0; i < input->numDims; i++)
		standardOutputTensorDims.push_back(input->m_Dims[i]);
}

unique_ptr<Tensor<double>> LeakyRelu::ForwardPropagate(unique_ptr<Tensor<double>>& input)
{
	m_Input = make_unique<Tensor<double>>();
	m_Input = move(input);

	m_Output = make_unique<Tensor<double>>(LeakyReluFunc());
	return make_unique<Tensor<double>>(*m_Output);
}

unique_ptr<Tensor<double>> LeakyRelu::BackwardPropagate(unique_ptr<Tensor<double>>& chainGradient, double learningRate)
{
	return make_unique<Tensor<double>>((*chainGradient) * LeakyReluGradient());
}

void LeakyRelu::load(FILE* fileModel)
{}

void LeakyRelu::save(FILE* fileModel)
{}