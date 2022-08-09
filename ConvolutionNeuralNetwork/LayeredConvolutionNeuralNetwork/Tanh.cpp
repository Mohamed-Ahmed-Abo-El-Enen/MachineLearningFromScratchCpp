#include "Tanh.h"

Tanh::Tanh()
{
	layerName = "Tanh";
}

double GetTanhVal(double x)
{
	return (2.0 / (1.0 + exp(-2 * x)) - 1);
}

Tensor<double> Tanh::TanhFunc()
{
	Tensor<double> result(m_Input->numDims, m_Input->m_Dims);
	for (int i = 0; i < m_Input->GetSize(); i++)
	{
		double x = m_Input->GetListIndex(i);
		double value = GetTanhVal(x);
		result.SetListIndex(value, i);
	}
	return result;
}

Tensor<double> Tanh::TanhGradient()
{
	Tensor<double> result(m_Input->numDims, m_Input->m_Dims);
	for (int i = 0; i < m_Input->GetSize(); i++)
	{
		double x = m_Input->GetListIndex(i);
		double value = 1 - pow(GetTanhVal(x), 2);
		result.SetListIndex(value, i);
	}
	return result;
}

void Tanh::Compile(unique_ptr<Tensor<double>>& input)
{
	for (int i = 0; i < input->numDims; i++)
		standardInputTensorDims.push_back(input->m_Dims[i]);

	for (int i = 0; i < input->numDims; i++)
		standardOutputTensorDims.push_back(input->m_Dims[i]);
}

unique_ptr<Tensor<double>> Tanh::ForwardPropagate(unique_ptr<Tensor<double>>& input)
{
	m_Input = make_unique<Tensor<double>>(input);
	m_Input = move(input);
	m_Output = make_unique<Tensor<double>>(TanhFunc());
	return make_unique<Tensor<double>>(*m_Output);
}

unique_ptr<Tensor<double>> Tanh::BackwardPropagate(unique_ptr<Tensor<double>>& chainGradient, double learningRate)
{
	return make_unique<Tensor<double>>((*chainGradient) * TanhGradient());
}

void Tanh::load(FILE* fileModel)
{}

void Tanh::save(FILE* fileModel)
{}