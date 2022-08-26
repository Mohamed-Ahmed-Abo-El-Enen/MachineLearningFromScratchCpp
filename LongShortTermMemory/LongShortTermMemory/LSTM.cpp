#include "LSTM.h"

LSTM::LSTM(int inputUnits, int hiddenUnits, int outputUnits, Optimizer* optimizer, bool haveEmbedding)
{
	m_Optimizer = optimizer;

	normal_distribution<double> distribution(mean, std);

	m_InputUnits = inputUnits;
	m_HiddenUnits = hiddenUnits;
	m_OutputUnits = outputUnits;
	m_HaveEmbedding = haveEmbedding;

	// lstm cell weights
	///--------------------------------------------------------------------------------------
	int dimsW_f[] = { m_HiddenUnits, m_HiddenUnits + m_OutputUnits };
	if (m_HaveEmbedding)
		dimsW_f[1] = m_HiddenUnits + m_InputUnits;

	Tensor<double> W_f(2, dimsW_f);
	generator.seed(seed);
	W_f.randn(generator, distribution, 1);

	int dimsb_f[] = { m_HiddenUnits, 1 };
	Tensor<double> b_f(2, dimsb_f);
	b_f.zero();

	///--------------------------------------------------------------------------------------
	int dimsW_i[] = { m_HiddenUnits, m_HiddenUnits + m_OutputUnits };
	if (m_HaveEmbedding)
		dimsW_i[1] = m_HiddenUnits + m_InputUnits;

	Tensor<double> W_i(2, dimsW_i);
	generator.seed(seed + 1);
	W_i.randn(generator, distribution, 1);

	int dimsb_i[] = { m_HiddenUnits, 1 };
	Tensor<double> b_i(2, dimsb_i);
	b_i.zero();

	///--------------------------------------------------------------------------------------
	int dimsW_g[] = { m_HiddenUnits, m_HiddenUnits + m_OutputUnits };
	if (m_HaveEmbedding)
		dimsW_g[1] = m_HiddenUnits + m_InputUnits;

	Tensor<double> W_g(2, dimsW_g);
	generator.seed(seed + 2);
	W_g.randn(generator, distribution, 1);

	int dimsb_g[] = { m_HiddenUnits, 1 };
	Tensor<double> b_g(2, dimsb_g);
	b_g.zero();

	///--------------------------------------------------------------------------------------
	int dimsW_o[] = { m_HiddenUnits, m_HiddenUnits + m_OutputUnits };
	if (m_HaveEmbedding)
		dimsW_o[1] = m_HiddenUnits + m_InputUnits;

	Tensor<double> W_o (2, dimsW_o);
	generator.seed(seed + 3);
	W_o.randn(generator, distribution, 1);

	int dimsb_o[] = { m_HiddenUnits, 1 };
	Tensor<double> b_o(2, dimsb_o);
	b_o.zero();

	///--------------------------------------------------------------------------------------
	int dimsW_v[] = { m_OutputUnits, m_HiddenUnits };
	Tensor<double> W_v(2, dimsW_v);
	generator.seed(seed + 4);
	W_v.randn(generator, distribution, 1);

	int dimsb_v[] = { m_OutputUnits, 1 };
	Tensor<double> b_v(2, dimsb_v);
	b_v.zero();
	

	parameters["W_f"] = W_f;
	parameters["b_f"] = b_f;

	parameters["W_i"] = W_i;
	parameters["b_i"] = b_i;

	parameters["W_g"] = W_g;
	parameters["b_g"] = b_g;

	parameters["W_o"] = W_o;
	parameters["b_o"] = b_o;

	parameters["W_v"] = W_v;
	parameters["b_v"] = b_v;	
}

void LSTM::ForwardPropagation(Tensor<double>& sample, Tensor<double> h_prev, Tensor<double> c_prev)
{
	assert(h_prev.numDims == 2 && c_prev.numDims == 2);
	assert(h_prev.m_Dims[0] == m_HiddenUnits && c_prev.m_Dims[0] == m_HiddenUnits);
	assert(h_prev.m_Dims[1] == 1 && c_prev.m_Dims[1] == 1);

	Tensor<double> W_f = parameters["W_f"];
	Tensor<double> b_f = parameters["b_f"];
								   	 
	Tensor<double> W_i = parameters["W_i"];
	Tensor<double> b_i = parameters["b_i"];
								   	 
	Tensor<double> W_g = parameters["W_g"];
	Tensor<double> b_g = parameters["b_g"];
								   	 
	Tensor<double> W_o = parameters["W_o"];
	Tensor<double> b_o = parameters["b_o"];

	Tensor<double> W_v = parameters["W_v"];
	Tensor<double> b_v = parameters["b_v"];

	vector<Tensor<double>> x_s;
	vector<Tensor<double>> z_s;
	vector<Tensor<double>> f_s;
	vector<Tensor<double>> i_s;
	vector<Tensor<double>> g_s;
	vector<Tensor<double>> c_s;
	vector<Tensor<double>> o_s;
	vector<Tensor<double>> h_s;
	vector<Tensor<double>> v_s;
	vector<Tensor<double>> output_s;

	// Append the initial celland hidden state to their respective lists
	h_s.push_back(h_prev);
	c_s.push_back(c_prev);

	for (int t = 0; t < sample.m_Dims[0]; t++)
	{		
		// Concatenate inputand hidden state
		Tensor<double> z = h_prev.ConcatenateAxis0(sample[t]);
		z_s.push_back(z);

		Tensor<double> res;

		//Calculate forget gate
		res = W_f.matmul(z) + b_f;
		Tensor<double> f = ActivationFunctions::Sigmoid(res);
		f_s.push_back(f);
		
		//Calculate input gate
		res = W_i.matmul(z) + b_i;
		Tensor<double> i = ActivationFunctions::Sigmoid(res);
		i_s.push_back(i);

		//Calculate candidate
		res = W_g.matmul(z) + b_g;
		Tensor<double> g = ActivationFunctions::Tanh(res);
		g_s.push_back(g);

		//Calculate memory state
		res = f * c_prev;
		c_prev = res + i * g;
		c_s.push_back(c_prev);

		//Calculate output gate
		res = W_o.matmul(z) + b_o;
		Tensor<double> o = ActivationFunctions::Sigmoid(res);
		o_s.push_back(o);

		// Calculate hidden state
		h_prev = o * (ActivationFunctions::Tanh(c_prev));
		h_s.push_back(h_prev);

		// Calculate logits
		Tensor<double> v = W_v.matmul(h_prev) + b_v;
		v_s.push_back(v);

		// Calculate softmax
		Tensor<double> output = ActivationFunctions::Softmax(v);
		output_s.push_back(output);
	}

	updatedParameters["z_s"] = z_s;
	updatedParameters["f_s"] = f_s;
	updatedParameters["i_s"] = i_s;
	updatedParameters["g_s"] = g_s;
	updatedParameters["c_s"] = c_s;
	updatedParameters["o_s"] = o_s;
	updatedParameters["h_s"] = h_s;
	updatedParameters["v_s"] = v_s;
	updatedParameters["output_s"] = output_s;
}

double LSTM::BackwardPropagation(Tensor<double>& targets)
{
	Tensor<double> W_f = parameters["W_f"];
	Tensor<double> b_f = parameters["b_f"];

	Tensor<double> W_i = parameters["W_i"];
	Tensor<double> b_i = parameters["b_i"];

	Tensor<double> W_g = parameters["W_g"];
	Tensor<double> b_g = parameters["b_g"];

	Tensor<double> W_o = parameters["W_o"];
	Tensor<double> b_o = parameters["b_o"];

	Tensor<double> W_v = parameters["W_v"];
	Tensor<double> b_v = parameters["b_v"];


	vector<Tensor<double>> z_s = updatedParameters["z_s"];
	vector<Tensor<double>> f_s = updatedParameters["f_s"];
	vector<Tensor<double>> i_s = updatedParameters["i_s"];
	vector<Tensor<double>> g_s = updatedParameters["g_s"];
	vector<Tensor<double>> c_s = updatedParameters["c_s"];
	vector<Tensor<double>> o_s = updatedParameters["o_s"];
	vector<Tensor<double>> h_s = updatedParameters["h_s"];
	vector<Tensor<double>> v_s = updatedParameters["v_s"];
	vector<Tensor<double>> output_s = updatedParameters["output_s"];

	Tensor<double> W_f_d(W_f.numDims, W_f.m_Dims);
	W_f_d.zero();
	Tensor<double> b_f_d(b_f.numDims, b_f.m_Dims);
	b_f_d.zero();

	Tensor<double> W_i_d(W_i.numDims, W_i.m_Dims);
	W_i_d.zero();
	Tensor<double> b_i_d(b_i.numDims, b_i.m_Dims);
	b_i_d.zero();

	Tensor<double> W_g_d(W_g.numDims, W_g.m_Dims);
	W_g_d.zero();
	Tensor<double> b_g_d(b_g.numDims, b_g.m_Dims);
	b_g_d.zero();

	Tensor<double> W_o_d(W_o.numDims, W_o.m_Dims);
	W_o_d.zero();
	Tensor<double> b_o_d(b_o.numDims, b_o.m_Dims);
	b_o_d.zero();

	Tensor<double> W_v_d(W_v.numDims, W_v.m_Dims);
	W_v_d.zero();
	Tensor<double> b_v_d(b_v.numDims, b_v.m_Dims);
	b_v_d.zero();

	Tensor<double> dh_next(h_s[0].numDims, h_s[0].m_Dims);
	dh_next.zero();

	Tensor<double> dc_next(c_s[0].numDims, c_s[0].m_Dims);
	dc_next.zero();

	double loss = 0;

	for (int t = output_s.size()-1; t >= 0; t--)
	{		
		//Compute the cross entropy
		loss += -1 * (output_s[t].log() * targets[t]).mean2D(0).get(0, 0);
		size_t index = t - 1 >= 0 ? t - 1 : output_s.size() - 1;
		Tensor<double> c_prev = c_s[t];

		// Compute the derivative of the relation of the hidden - state to the output gate
		Tensor<double> dv = output_s[t];
		index = targets[t].argmax(0).get(0, 0);
		double val = dv.get(index, 0) - 1;
		dv.set(val, index, 0);
	
		//Update the gradient of the relation of the hidden - state to the output gate
		W_v_d = W_v_d + dv.matmul(h_s[t].transpose());
		b_v_d = b_v_d + dv;

		//Compute the derivative of the hidden state and output gate
		Tensor<double> dh = W_v.transpose().matmul(dv);
		dh = dh + dh_next;		
		Tensor<double> _do = dh * ActivationFunctions::Tanh(c_s[t+1]);
		_do = ActivationFunctions::SigmoidDerivative(o_s[t]) * _do;

		// Update the gradients with respect to the output gate
		W_o_d = W_o_d + _do.matmul(z_s[t].transpose());
		b_o_d = b_o_d + _do;

		// Compute the derivative of the cell state and candidate g
		Tensor<double> dc = dc_next;
		Tensor<double> tanh = ActivationFunctions::Tanh(c_s[t+1]);
		dc = dc + (dh * o_s[t] * ActivationFunctions::TanhDerivative(tanh));
		Tensor<double> dg = dc * i_s[t];
		dg = ActivationFunctions::TanhDerivative(g_s[t]) * dg;

		// Update the gradients with respect to the candidate
		W_g_d = W_g_d + dg.matmul(z_s[t].transpose());
		b_g_d = b_g_d + dg;

		// Compute the derivative of the input gate and update its gradients
		Tensor<double> di = dc * g_s[t];
		di = ActivationFunctions::SigmoidDerivative(i_s[t]) * di;
		W_i_d = W_i_d + di.matmul(z_s[t].transpose());
		b_i_d = b_i_d + di;

		// Compute the derivative of the forget gate and update its gradients
		Tensor<double> df = dc * c_prev;
		df = ActivationFunctions::Sigmoid(f_s[t]) * df;
		W_f_d = W_f_d + df.matmul(z_s[t].transpose());
		b_f_d = b_f_d + df;

		//Compute the derivative of the inputand update the gradients of the previous hiddenand cell state
		Tensor<double> dz = W_f.transpose().matmul(df) +
			W_i.transpose().matmul(di) +
			W_g.transpose().matmul(dg) +
			W_o.transpose().matmul(_do);
		Tensor<double> dh_prev = dz.endAt(m_HiddenUnits, 0);
		Tensor<double> dc_prev = f_s[t] * dc;		
	}
	grads["W_f_d"] = W_f_d;
	grads["b_f_d"] = b_f_d;

	grads["W_i_d"] = W_i_d;
	grads["b_i_d"] = b_i_d;

	grads["W_g_d"] = W_g_d;
	grads["b_g_d"] = b_g_d;

	grads["W_o_d"] = W_o_d;
	grads["b_o_d"] = b_o_d;

	grads["W_v_d"] = W_v_d;
	grads["b_v_d"] = b_v_d;

	return loss;
}

void LSTM::UpdateParameters()
{
	m_Optimizer->UpdateParameters(parameters, grads);
}

void LSTM::Train(Tensor<double>& X_tensor, Tensor<double>& y_tensor, int epochs)
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

			int dimsC[] = { m_HiddenUnits, 1 };
			Tensor<double> c(2, dimsC);
			c.zero();

			ForwardPropagation(x_Sample, h, c);
			double loss = BackwardPropagation(y_Sample);

			UpdateParameters();

			epochTrainLoss += loss;
		}
		trainLoss.push_back(epochTrainLoss);

		if (itr % 5 == 0)
			cout << "Epoch " << itr << ", training loss: " << trainLoss[itr] << endl;
	}
}

vector<vector<Tensor<double>>> LSTM::Predict(Tensor<double>& X_tensor)
{
	vector<vector<Tensor<double>>> yHat;
	for (size_t i = 0; i < X_tensor.m_Dims[0]; i++)
	{
		Tensor<double> x_Sample = X_tensor[i];

		int dimsH[] = { m_HiddenUnits, 1 };
		Tensor<double> h(2, dimsH);
		h.zero();

		int dimsC[] = { m_HiddenUnits, 1 };
		Tensor<double> c(2, dimsC);
		c.zero();

		ForwardPropagation(x_Sample, h, c);

		vector<Tensor<double>> outputs = updatedParameters["output_s"];
		yHat.push_back(outputs);
	}
	return yHat;
}