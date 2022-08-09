#include "MaxPool.h"

MaxPool::MaxPool(int size, int stride)
{
	layerName = "MaxPool";
	m_Size = size;
	m_Stride = stride;
}

void MaxPool::Compile(unique_ptr<Tensor<double>>& input)
{
	for (int i = 0; i < input->numDims; i++)
		standardInputTensorDims.push_back(input->m_Dims[i]);

	int h = ((input->m_Dims[2] - (m_Size - 1) - 1) / m_Stride) + 1;
	int w = ((input->m_Dims[3] - (m_Size - 1) - 1) / m_Stride) + 1;
	int resultDim[] = { input->m_Dims[0], input->m_Dims[1], h, w };
	for (int i = 0; i < 4; i++)
		standardOutputTensorDims.push_back(resultDim[i]);
}

unique_ptr<Tensor<double>> MaxPool::ForwardPropagate(unique_ptr<Tensor<double>>& input)
{
	m_Input = make_unique<Tensor<double>>();
	m_Input = move(input);

	int h = ((m_Input->m_Dims[2] - (m_Size - 1) - 1) / m_Stride) + 1;
	int w = ((m_Input->m_Dims[3] - (m_Size - 1) - 1) / m_Stride) + 1;
	int m_Dims[] = { m_Input->m_Dims[0], m_Input->m_Dims[1], h,w };
	m_Output = make_unique<Tensor<double>>(4, m_Dims);
	m_Indexes = make_unique<Tensor<int>>(4, m_Dims);

	for (int i = 0; i < m_Input->m_Dims[0]; i++) // for each batch image
	{
		for (int j = 0; j < m_Input->m_Dims[1]; j++) // for each image
		{
			for (int k = 0; k < m_Dims[2]; k++) // for each output y 
			{
				for (int l = 0; l < m_Dims[3]; l++) // for each output x
				{
					double max = -DBL_MAX;
					int index = 0;
					for (int m = 0; m < m_Size; m++)
					{
						for (int n = 0; n < m_Size; n++)
						{
							int input_y = k * m_Stride + m;
							int input_x = l * m_Stride + n;
							double value = m_Input->get(i, j, input_y, input_x);
							if (value > max)
							{
								index = m * m_Size + n;
								max = value;
							}
						}
					}
					m_Output->set(max, i, j, k, l);
					m_Indexes->set(index, i, j, k, l);
				}
			}
		}
	}

	return make_unique<Tensor<double>>(*m_Output);
}

unique_ptr<Tensor<double>> MaxPool::BackwardPropagate(unique_ptr<Tensor<double>>& chainGradient, double learningRate)
{
	unique_ptr<Tensor<double>> inputGradient = make_unique<Tensor<double>>(m_Input->numDims, m_Input->m_Dims);
	inputGradient->zero();

	for (int i = 0; i < m_Input->m_Dims[0]; i++) // for each batch image
	{
		for (int j = 0; j < m_Input->m_Dims[1]; j++) // for each image
		{
			for (int k = 0; k < m_Output->m_Dims[2]; k++) // for each output y
			{
				for (int l = 0; l < m_Output->m_Dims[3]; l++) // for each output y
				{
					double chainGrad = chainGradient->get(i, j, k, l);
					int index = m_Indexes->get(i, j, k, l);
					int m = index / m_Size;
					int n = index % m_Size;
					int input_y = k * m_Stride + m;
					int input_x = l * m_Stride + n;
					inputGradient->set(chainGrad, i, j, input_y, input_x);
				}
			}
		}
	}
	return inputGradient;
}

void MaxPool::load(FILE* fileModel)
{}

void MaxPool::save(FILE* fileModel)
{}