#include "Dropout.h"
#include "Tensor.h"

Dropout::Dropout(double p, int seed)
{
	layerName = "Dropout";
	this->p = p;
	this->seed = seed;
}

void Dropout::DropoutCutOp(default_random_engine generator, uniform_real_distribution<> distribution, double p)
{
	for (int i = 0; i < m_Input->GetSize(); i++)
		m_Input->SetListIndex((distribution(generator) < p) / p, i);
}

void Dropout::Compile(unique_ptr<Tensor<double>>& input)
{
	for (int i = 0; i < input->numDims; i++)
		standardInputTensorDims.push_back(input->m_Dims[i]);

	for (int i = 0; i < input->numDims; i++)
		standardOutputTensorDims.push_back(input->m_Dims[i]);
}

unique_ptr<Tensor<double>> Dropout::ForwardPropagate(unique_ptr<Tensor<double>>& input)
{
	m_Input = make_unique<Tensor<double>>(input->numDims, input->m_Dims);

	default_random_engine generator(seed);
	uniform_real_distribution<> distribution(0.0, 1.0);

	DropoutCutOp(generator, distribution, p);
	m_Output = make_unique<Tensor<double>>((*input) * (*m_Input));
	return make_unique<Tensor<double>>(*m_Output);
}

unique_ptr<Tensor<double>> Dropout::BackwardPropagate(unique_ptr<Tensor<double>>& chainGradient, double learningRate)
{
	return make_unique<Tensor<double>>((*chainGradient) * (*m_Input));
}

void Dropout::load(FILE *fileModel){}

void Dropout::save(FILE* fileModel) {}