#include "Conv2D.h"

Conv2D::Conv2D(FilterShape inoutChannels, MatShape kernelSize, int stride, int padding, int seed)
{
	layerName = "Conv2D";
	default_random_engine generator(seed);
	normal_distribution<double> distribution(0.0, 1.0);

	int kernelDims[] = { inoutChannels.out, inoutChannels.in, kernelSize.rows, kernelSize.columns };
	kernels = make_unique<Tensor<double>>(4, kernelDims);
	kernels->randn(generator, distribution, sqrt(2.0 / (kernelSize.rows * kernelSize.columns * inoutChannels.out)));

	int biasDims[] = { inoutChannels.out };
	bias = make_unique<Tensor<double>>( 1, biasDims );
	bias->randn(generator, distribution, 0);
	this->stride = stride;
	this->padding = padding;
}

void Conv2D::Compile(unique_ptr<Tensor<double>>& input)
{
	assert(kernels->m_Dims[1] == input->m_Dims[1]);
	int h = ((input->m_Dims[2] + 2 * padding - (kernels->m_Dims[2] - 1) - 1) / stride) + 1;
	int w = ((input->m_Dims[3] + 2 * padding - (kernels->m_Dims[3] - 1) - 1) / stride) + 1;

	for (int i = 0; i < input->numDims; i++)
		standardInputTensorDims.push_back(input->m_Dims[i]);
	
	int resultDims[] = { input->m_Dims[0], kernels->m_Dims[0], h, w };
	for (int i = 0; i < 4; i++)
		standardOutputTensorDims.push_back(resultDims[i]);
}

unique_ptr<Tensor<double>> Conv2D::ForwardPropagate(unique_ptr<Tensor<double>>& input)
{
	assert(kernels->m_Dims[1] == input->m_Dims[1]);
	m_Input = make_unique<Tensor<double>>();
	m_Input = move(input);

	int w = ((m_Input->m_Dims[3] + 2 * padding - (kernels->m_Dims[3] - 1) - 1) / stride) + 1;
	int h = ((m_Input->m_Dims[2] + 2 * padding - (kernels->m_Dims[2] - 1) - 1) / stride) + 1;
	int resultDims[] = { m_Input->m_Dims[0], kernels->m_Dims[0], h, w };
	m_Output = make_unique <Tensor<double>>(4, resultDims);

	for (int i = 0; i < m_Input->m_Dims[0]; ++i) // for each batch img
	{ 
		for (int j = 0; j < kernels->m_Dims[0]; ++j) // for each output volume
		{ 
			for (int k = 0; k < h; ++k) // for every vertical k in the output volume
			{ 
				for (int l = 0; l < w; ++l) // for every horizontal l in the output volume
				{ 
					int imgStride_i = stride * k - padding;
					int imgStride_j = stride * l - padding;
					double total = 0;
					for (int m = 0; m < kernels->m_Dims[1]; ++m) //for each filter 
					{ 
						for (int n = 0; n < kernels->m_Dims[2]; ++n) 
						{
							for (int o = 0; o < kernels->m_Dims[3]; ++o)
							{
								int x = imgStride_i + n, y = imgStride_j + o;

								if (x < 0 || x >= m_Input->m_Dims[2] || y < 0 || y >= m_Input->m_Dims[3])
									continue; // if it is padding region, skip(sum 0)

								double a = m_Input->get(i, m, x, y);
								double b = kernels->get(j, m, n, o);
								total += a * b;
							}
						}
					}
					m_Output->set(total + bias->get(j), i, j, k, l);
				}
			}
		}
	}

	return make_unique<Tensor<double>>(*m_Output);
}

unique_ptr<Tensor<double>> Conv2D::BackwardPropagate(unique_ptr<Tensor<double>>& chainGradient, double learningRate)
{
	unique_ptr<Tensor<double>> inputGradient = make_unique<Tensor<double>>(m_Input->numDims, m_Input->m_Dims);
	unique_ptr<Tensor<double>> kernelsGradient = make_unique<Tensor<double>>(kernels->numDims, kernels->m_Dims);
	unique_ptr<Tensor<double>> biasGradient = make_unique<Tensor<double>>(1, bias->m_Dims);
	inputGradient->zero();
	kernelsGradient->zero();
	biasGradient->zero();

	for (int i = 0; i < m_Input->m_Dims[0]; ++i) // for each batch img
	{ 
		for (int f = 0; f < kernels->m_Dims[0]; f++) // for each filter
		{ 
			int x = -padding;
			for (int cx = 0; cx < chainGradient->m_Dims[2]; x += stride, cx++) // for each x in the chain gradient
			{ 
				int y = -padding;
				for (int cy = 0; cy < chainGradient->m_Dims[3]; y += stride, cy++) // for each y in the chain gradient
				{ 
					double chain_grad = chainGradient->get(i, f, cx, cy);
					for (int fx = 0; fx < kernels->m_Dims[2]; fx++) // for each x in the filter
					{ 
						int ix = x + fx; // input x
						if (ix >= 0 && ix < m_Input->m_Dims[2]) {
							for (int fy = 0; fy < kernels->m_Dims[3]; fy++) // for each y in the filter
							{ 
								int iy = y + fy; // input y
								if (iy >= 0 && iy < m_Input->m_Dims[3]) 
								{
									for (int fc = 0; fc < kernels->m_Dims[1]; fc++) // for each channel in the filter
									{ 
										kernelsGradient->add(m_Input->get(i, fc, ix, iy) * chain_grad, f, fc, fx, fy);
										inputGradient->add(kernels->get(f, fc, fx, fy) * chain_grad, i, fc, ix, iy);

									}
								}
							}
						}
					}
					biasGradient->add(chain_grad, f);
				}
			}
		}
	}

	*kernels -= (*kernelsGradient) * learningRate;
	*bias -= (*biasGradient) * learningRate;

	return make_unique<Tensor<double>>(*inputGradient);
}

void Conv2D::load(FILE* fileModel)
{
	double value;
	for (int i = 0; i < kernels->m_Dims[0]; i++)
	{
		for (int j = 0; j < kernels->m_Dims[1]; j++)
		{
			for (int k = 0; k < kernels->m_Dims[2]; k++)
			{
				for (int l = 0; l < kernels->m_Dims[3]; l++)
				{
					int read = fscanf_s(fileModel, "%lf", &value);
					if (read != 1)
						throw runtime_error("Invalid model file");
					kernels->set(value, i, j, k, l);
				}
			}
		}
	}
	for (int m = 0; m < bias->m_Dims[0]; m++)
	{
		int read = fscanf_s(fileModel, "%lf", &value);
		if (read != 1)
			throw runtime_error("Invalid model file");
		bias->set(value, m);
	}
}

void Conv2D::save(FILE* fileModel)
{
	for (int i = 0; i < kernels->m_Dims[0]; i++)	
		for (int j = 0; j < kernels->m_Dims[1]; j++)		
			for (int k = 0; k < kernels->m_Dims[2]; k++)			
				for (int l = 0; l < kernels->m_Dims[3]; l++)				
					fprintf(fileModel, "%18lf", kernels->get(i, j, k, l));							
			
	for (int m = 0; m < bias->m_Dims[0]; m++)	
		fprintf(fileModel, "%18lf", bias->get(m));
	
}