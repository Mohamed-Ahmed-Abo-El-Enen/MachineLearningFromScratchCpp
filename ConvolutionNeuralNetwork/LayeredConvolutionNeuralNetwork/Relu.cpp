#include "Relu.h"

Relu::Relu()
{
	layerName = "Relu";
}

Tensor<double> Relu::relu()
{
	Tensor<double> result(m_Input->numDims, m_Input->m_Dims);
	for (int i = 0; i < m_Input->GetSize(); i++)
	{
		double x = m_Input->GetListIndex(i);
		double value = x > 0 ? x : 0;
		result.SetListIndex(value, i);
	}
	return result;
}

Tensor<double> Relu::reluGradient()
{
	Tensor<double> result(m_Input->numDims, m_Input->m_Dims);
	double value = 0;
	for (int i = 0; i < m_Input->GetSize(); i++)
	{
		value = m_Input->GetListIndex(i) > 0 ? 1 : 0;
		result.SetListIndex(value, i);
	}
	return result;
}

void Relu::Compile(unique_ptr<Tensor<double>>& input)
{
	for (int i = 0; i < input->numDims; i++)
		standardInputTensorDims.push_back(input->m_Dims[i]);

	for (int i = 0; i < input->numDims; i++)
		standardOutputTensorDims.push_back(input->m_Dims[i]);
}

unique_ptr<Tensor<double>> Relu::ForwardPropagate(unique_ptr<Tensor<double>>& input)
{
	m_Input = make_unique<Tensor<double>>();
	m_Input = move(input);
	m_Output = make_unique<Tensor<double>>(relu());
	return make_unique<Tensor<double>>(*m_Output);;
}

unique_ptr<Tensor<double>> Relu::BackwardPropagate(unique_ptr<Tensor<double>>& chainGradient, double learningRate)
{
	return make_unique<Tensor<double>>((*chainGradient) * reluGradient());
}

void Relu::load(FILE* fileModel)
{}

void Relu::save(FILE* fileModel)
{}