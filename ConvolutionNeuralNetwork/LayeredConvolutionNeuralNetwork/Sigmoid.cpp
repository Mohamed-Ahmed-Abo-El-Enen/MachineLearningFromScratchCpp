#include "Sigmoid.h"

Sigmoid::Sigmoid()
{
	layerName = "Sigmoid";
}

double SigmoidVal(double x)
{
	return  1.0 / (1.0 + exp(-x));
}

Tensor<double> Sigmoid::SigmoidFunc()
{
	Tensor<double> result(m_Input->numDims, m_Input->m_Dims);
	for (int i = 0; i < m_Input->GetSize(); i++)
	{
		double x = m_Input->GetListIndex(i);
		double value = SigmoidVal(x);
		result.SetListIndex(value, i);
	}
	return result;
}

Tensor<double> Sigmoid::SigmoidGradient()
{
	Tensor<double> result(m_Input->numDims, m_Input->m_Dims);
	for (int i = 0; i < m_Input->GetSize(); i++)
	{
		double x = m_Input->GetListIndex(i);
		x = SigmoidVal(x);
		double value = (x * (1.0 - x));
		result.SetListIndex(value, i);
	}
	return result;
}

void Sigmoid::Compile(unique_ptr<Tensor<double>>& input)
{
	for (int i = 0; i < input->numDims; i++)
		standardInputTensorDims.push_back(input->m_Dims[i]);

	for (int i = 0; i < input->numDims; i++)
		standardOutputTensorDims.push_back(input->m_Dims[i]);
}

unique_ptr<Tensor<double>> Sigmoid::ForwardPropagate(unique_ptr<Tensor<double>>& input)
{
	m_Input = make_unique<Tensor<double>>();
	m_Input = move(input);
	m_Output = make_unique<Tensor<double>>(SigmoidFunc());
	return make_unique<Tensor<double>>(*m_Output);
}

unique_ptr<Tensor<double>> Sigmoid::BackwardPropagate(unique_ptr<Tensor<double>>& chainGradient, double learningRate)
{
	return make_unique<Tensor<double>>((*chainGradient) * SigmoidGradient());
}

void Sigmoid::load(FILE* fileModel)
{}

void Sigmoid::save(FILE* fileModel)
{}