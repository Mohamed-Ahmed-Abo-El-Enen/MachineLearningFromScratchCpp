#include "RNN.h"

RNN::RNN(int hiddenUnits, int outputUnits, Optimizer* optimizer)
{
	normal_distribution<double> distribution(mean, std);

	m_HiddenUnits = hiddenUnits;
	m_OutputUnits = outputUnits;
	m_Optimizer = optimizer;

	// Weight matrix(input to hidden state)
	int dimsU[] = { m_HiddenUnits, m_OutputUnits};
	Tensor<double> U(2, dimsU);
	generator.seed(seed);
	U.randn(generator, distribution, 1);

	// Weight matrix (recurrent computation)
	int dimsV[] = { m_HiddenUnits, m_HiddenUnits };
	Tensor<double> V(2, dimsV);
	generator.seed(seed+1);
	V.randn(generator, distribution, 1);

	// Weight matrix (hidden state to output)
	int dimsW[] = { m_OutputUnits, m_HiddenUnits };
	Tensor<double> W(2, dimsW);
	generator.seed(seed+2);
	W.randn(generator, distribution, 1);

	// Bias (hidden state)
	int dimshidden_b[] = { m_HiddenUnits, 1 };
	Tensor<double> hidden_b(2, dimshidden_b);
	hidden_b.zero();

	// Bias (output)
	int dimsout_b[] = { m_OutputUnits, 1 };
	Tensor<double> out_b(2, dimsout_b);
	out_b.zero();
	
	parameters["U"] = U;
	parameters["V"] = V;
	parameters["W"] = W;

	parameters["hidden_b"] = hidden_b;
	parameters["out_b"] = out_b;
}

void RNN::ForwardPropagation(Tensor<double>& inputs, Tensor<double> hiddenState)
{
	Tensor<double> U = parameters["U"];
	Tensor<double> V = parameters["V"];
	Tensor<double> W = parameters["W"];
	Tensor<double> hidden_b = parameters["hidden_b"];
	Tensor<double> out_b = parameters["out_b"];

	vector<Tensor<double>> outputs;
	vector<Tensor<double>> hiddenStates;

	for (int t = 0; t < inputs.m_Dims[0]; t++)
	{
		// Compute new hidden state
		Tensor<double> result = U.matmul(inputs[t]);
		result = result + V.matmul(hiddenState);
		result = result + hidden_b;
		hiddenState = ActivationFunctions::Tanh(result);

		// Compute output
		result = W.matmul(hiddenState);
		result = result + out_b;
		Tensor<double> output = ActivationFunctions::Softmax(result);

		// Save resultsand continue
		outputs.push_back(output);
		hiddenStates.push_back(hiddenState);
	}

	updatedParameters["hiddenStates_s"] = hiddenStates;
	updatedParameters["output_s"] = outputs;
}

void RNN::ClipGradientNorm(double maxNorm)
{
	double totalNorm = 0;
	map<string, Tensor<double>>::iterator it;

	for (it = grads.begin(); it != grads.end(); it++)
	{
		double gradNorm = it->second.pow(2).sum();
		totalNorm += gradNorm;
	}
	totalNorm = std::sqrt(totalNorm);

	double clipCoef = maxNorm / (totalNorm + 1e-6);

	if (clipCoef < 1)
	{
		for (it = grads.begin(); it != grads.end(); it++)		
			it->second = it->second * clipCoef;		
	}
}

Tensor<double> Softmax(Tensor<double> m_Input)
{
	assert(m_Input.numDims == 2);
	Tensor<double> probabilities(m_Input.numDims, m_Input.m_Dims);
	double expSum = m_Input.exp().sum();
	for (size_t i = 0; i < m_Input.m_Dims[0]; i++)
	{
		double exp = m_Input.exp().get(i);
		probabilities.set(exp / expSum, i, 0);
	}
	return probabilities;
}

double RNN::BackwardPropagation(Tensor<double>& inputs, Tensor<double>& targets)
{
	Tensor<double> U = parameters["U"];
	Tensor<double> V = parameters["V"];
	Tensor<double> W = parameters["W"];
	Tensor<double> hidden_b = parameters["hidden_b"];
	Tensor<double> out_b = parameters["out_b"];

	Tensor<double> U_d(parameters["U"].numDims, parameters["U"].m_Dims);
	U_d.zero();

	Tensor<double> V_d(parameters["V"].numDims, parameters["V"].m_Dims);
	V_d.zero();

	Tensor<double> W_d(parameters["W"].numDims, parameters["W"].m_Dims);
	W_d.zero();

	Tensor<double> hidden_b_d(parameters["hidden_b"].numDims, parameters["hidden_b"].m_Dims);
	hidden_b_d.zero();

	Tensor<double> out_b_d(parameters["out_b"].numDims, parameters["out_b"].m_Dims);
	out_b_d.zero();

	vector<Tensor<double>> hiddenStates_s = updatedParameters["hiddenStates_s"];
	vector<Tensor<double>> output_s = updatedParameters["output_s"];

	Tensor<double> h_next_d(hiddenStates_s[0].numDims, hiddenStates_s[0].m_Dims);
	h_next_d.zero();

	double loss = 0;

	for (int t = output_s.size() - 1; t >= 0; t--)
	{
		// Compute cross - entropy loss(as a scalar)
		loss += -1 * (output_s[t].log() * targets[t]).mean2D(0).get(0, 0);

		// Backpropagate into output(derivative of cross - entropy)
		Tensor<double> _do = output_s[t];
		int index = targets[t].argmax(0).get(0, 0);
		double val = _do.get(index, 0) - 1;
		_do.set(val, index, 0);

		// Backpropagate into W
		W_d = W_d + _do.matmul(hiddenStates_s[t].transpose());
		out_b_d = out_b_d + _do;

		// Backpropagate into h
		Tensor<double> h_d = W.transpose().matmul(_do) + h_next_d;

		// Backpropagate into W
		W_d = W_d + _do.matmul(hiddenStates_s[t].transpose());
		out_b_d = out_b_d + _do;

		// Backpropagate into h
		h_d = W.transpose().matmul(_do) + h_next_d;

		// Backpropagate through non-linearity
		Tensor<double> f_d = ActivationFunctions::TanhDerivative(hiddenStates_s[t]) * h_d;
		hidden_b_d = hidden_b_d + f_d;

		// Backpropagate into U
		U_d = U_d + f_d.matmul(inputs[t].transpose());

		// Backpropagate into V
		index = t - 1 >= 0 ? t - 1 : output_s.size() - 1;
		V_d = V_d + f_d.matmul(hiddenStates_s[index].transpose());
		h_next_d = V.transpose().matmul(f_d);
	}
	grads["U_d"] = U_d;
	grads["V_d"] = V_d;
	grads["W_d"] = W_d;
	grads["hidden_b_d"] = hidden_b_d;
	grads["out_b_d"] = out_b_d;

	ClipGradientNorm();

	return loss;
}

void RNN::UpdateParameters()
{
	m_Optimizer->UpdateParameters(parameters, grads);
}

void RNN::Train(Tensor<double>& X_tensor, Tensor<double>& y_tensor, int epochs)
{
	vector<double> trainLoss;
	m_Optimizer->InitializeParams(parameters);

	for (int itr = 0; itr < epochs; itr++)
	{
		double epochTrainLoss = 0;
		for (size_t i = 0; i < X_tensor.m_Dims[0]; i++)
		{
			Tensor<double> x_Sample = X_tensor[i];
			Tensor<double> y_Sample = y_tensor[i];

			int dimsH[] = { m_HiddenUnits, 1 };
			Tensor<double> h(2, dimsH);
			h.zero();		

			ForwardPropagation(x_Sample, h);
			updatedParameters["output_s"];
			double loss = BackwardPropagation(x_Sample, y_Sample);

			UpdateParameters();

			epochTrainLoss += loss;
		}
		trainLoss.push_back(epochTrainLoss);

		if (itr % 5 == 0)
			cout << "Epoch " << itr << ", training loss: " << trainLoss[itr] << endl;
	}
}

vector<vector<Tensor<double>>> RNN::Predict(Tensor<double>& X_tensor)
{
	vector<vector<Tensor<double>>> yHat;
	for (size_t i = 0; i < X_tensor.m_Dims[0]; i++)
	{
		Tensor<double> x_Sample = X_tensor[i];

		int dimsH[] = { m_HiddenUnits, 1 };
		Tensor<double> h(2, dimsH);
		h.zero();

		ForwardPropagation(x_Sample, h);

		vector<Tensor<double>> outputs = updatedParameters["output_s"];
		yHat.push_back(outputs);
	}
	return yHat;
}
