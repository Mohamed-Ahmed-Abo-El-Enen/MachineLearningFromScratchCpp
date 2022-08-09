#include "Dense.h"

Dense::Dense(int inputSize, int outputSize, int seed)
{
	layerName = "Dense";
	default_random_engine generator(seed);
	normal_distribution<double> distribution(0.0, 1.0);
	int weightsDims[] = { inputSize, outputSize };
	weights = make_unique<Tensor<double>>(2, weightsDims);
	weights->randn(generator, distribution, sqrt(2.0 / inputSize));
	int biasDims[] = { outputSize };
	bias = make_unique<Tensor<double>>(1, biasDims);
	bias->randn(generator, distribution, 0);
}

void Dense::Compile(unique_ptr<Tensor<double>>& input)
{
	for (int i = 0; i < input->numDims; i++)
		standardInputTensorDims.push_back(input->m_Dims[i]);

	int resultDims[] = { input->m_Dims[0], weights->m_Dims[1] };
	for (int i = 0; i < 2; i++)
		standardOutputTensorDims.push_back(resultDims[i]);
}

unique_ptr<Tensor<double>> Dense::ForwardPropagate(unique_ptr<Tensor<double>>& input)
{	
	m_Input = make_unique<Tensor<double>>();
	m_Input = move(input);

	inputNumDims = m_Input->numDims;
	copy(m_Input->m_Dims, m_Input->m_Dims + m_Input->numDims, inputDims);
	if (m_Input->numDims != 2)
	{
		// flatten tensor
		int flattenSize = 1;
		for (int i = 1; i < m_Input->numDims; i++)
			flattenSize *= m_Input->m_Dims[i];
		
		int m_Dims[] = { m_Input->m_Dims[0], flattenSize };
		m_Input->view(2, m_Dims);
	}
	m_Output = make_unique<Tensor<double>>(m_Input->matmul(weights) + bias);

	return make_unique<Tensor<double>>(*m_Output);
}

unique_ptr<Tensor<double>> Dense::BackwardPropagate(unique_ptr<Tensor<double>>& chainGradient, double learningRate)
{
	Tensor<double> weightGradient = m_Input->transpose().matmul(*chainGradient);
	Tensor<double> biasGradient = chainGradient->ColumnWiseSum();
	Tensor<double> gradient = chainGradient->matmul(weights->transpose());
	gradient.view(inputNumDims, inputDims);

	*weights -= weightGradient * learningRate;
	*bias -= biasGradient * learningRate;

	return make_unique<Tensor<double>>(gradient);
}

void Dense::load(FILE* fileModel)
{
	double value;
	for (int i = 0; i < weights->m_Dims[0]; i++)
	{
		for (int j = 0; j < weights->m_Dims[1]; j++)
		{
			int read = fscanf_s(fileModel, "%lf", &value);
			if (read != 1)
				throw runtime_error("Invalid model file");
			
			weights->set(value, i, j);
		}
	}

	for (int i = 0; i < bias->m_Dims[0]; i++)
	{
		int read = fscanf_s(fileModel, "%lf", &value);
		if (read != 1)
			throw runtime_error("Invalid model file");

		bias->set(value, i);
	}
}

void Dense::save(FILE* fileModel)
{
	for (int i = 0; i < weights->m_Dims[0]; i++)	
		for (int j = 0; j < weights->m_Dims[1]; j++)		
			fprintf(fileModel, "%18lf", weights->get(i, j));		
	

	for (int i = 0; i < bias->m_Dims[0]; i++)
	{
		fprintf(fileModel, "%18lf", bias->get(i));
	}
}