#include "SVM.h"

SVM::SVM(const bool _verbose)
{
	this->kernel = kernel::linear;
	this->params = vector<double>(2,0);
	this->verbose = _verbose;
	this->isHardMargin = true;
}

SVM::SVM(const kernelFunction _kernel, const vector<double> _params,const bool _verbose)
{
	this->kernel = _kernel;
	this->params = _params;
	this->verbose = _verbose;
}

SVM::~SVM()
{
}

void SVM::log(const string str)
{
	if (this->verbose)
		cout << str << flush;
	return;
}

//void SVM::train(const vector<vector<double>> class1Data, const vector<vector<double>> class2Data, const size_t D, const double C, const double lr, const double limit)
void SVM::train(const vector<vector<double>> x, const vector<int> y, const size_t D, const double C, const double lr, const double limit)
{
	// Set Lagrange Multiplier and Parameters
	size_t N = x.size();
	vector<double> alpha = vector<double>(N, 0.0);
	double beta = 1.0;
	double delta;

	//Training 
	this->log("\n");
	this->log("/////////////////////// Training ///////////////////////\n");

	bool judge = true;
	double error;

	constexpr double eps = 0.0000001;
	const double breakEps = 0.0000001;
	double previousError = -INFINITY;

	int iterations = 0;
	int maxIterations = 128;

	double item1, item2, item3;
	while (judge)
	{
		judge = false;
		error = 0.0;

		// updata Alpha
		// Compute the partial derivative about alpha i as
		// dl/dalpha = 1 - sum_j(alpha_j * y_i * y_j * K(X_i^T, X_j)) - beta * sum(alpha_j * y_i * y_j)
		for (size_t i = 0; i < N; i++)
		{
			// Set item1
			item1 = 0.0;
			for (size_t j = 0; j < N; j++)
				item1 += alpha[j] * (double)y[i] * (double)y[j] * this->kernel(x[i], x[j], this->params);

			// Set item2
			item2 = 0.0;
			for (size_t j = 0; j < N; j++)
				item2 += alpha[j] * (double)y[i] * (double)y[j];

			// Set delta
			delta = 1.0 - item1 - beta * item2;
			
			// Update
			// alpha = alpha + lr * dl/dalpha
			alpha[i] += lr * delta;

			// check stopping delta limit
			if (alpha[i] < 0.0)
				alpha[i] = 0.0;

			// Regularization
			else if (alpha[i] > C)
				alpha[i] = C;

			else if (abs(delta) > limit)
			{
				judge = true;
				error += abs(delta) - limit;
			}				
		}

		// update beta
		// beta = beta + 1/2 * sum(alpha_i * y_i)^2
		item3 = 0.0;
		for (size_t i = 0; i < N; i++)		
			item3 +=  alpha[i] * (double)y[i];
		
		beta +=  1 / 2 * (item3 * item3);

		// Output Residual Error
		this->log("\r error: " + to_string(error));

		// Break early 
		if (abs(error - previousError) <= breakEps)
			iterations++;
		else
		{
			previousError = error;
			iterations = 0;
		}

		if (iterations >= maxIterations)
			judge = false;
	}

	this->log("\n");
	this->log("////////////////////////////////////////////////////////\n");

	// Descripe for support vetors
	// points on the decision boundary
	int Ns = 0;
	this->xs = vector<vector<double>>();
	this->ys = vector<int>();
	this->alpha_s = vector<double>();

	int Ns_in = 0;
	this->xs_in = vector<vector<double>>();
	this->ys_in = vector<int>();
	this->alpha_s_in = vector<double>();

	for (int i = 0; i < N; i++)
	{
		if ((alpha[i] > eps) && (alpha[i] < C-eps))
		{
			this->xs.push_back(x[i]);
			this->ys.push_back(y[i]);
			this->alpha_s.push_back(alpha[i]);
			Ns++;
		}

		else if (alpha[i] >= C - eps)
		{
			this->xs_in.push_back(x[i]);
			this->ys_in.push_back(y[i]);
			this->alpha_s_in.push_back(alpha[i]);
			Ns_in++;
		}
	}

	this->log("Ns (number of support vectors on margin) = " + to_string(Ns) + "\n");
	this->log("Ns (number of support vectors inside margin) = " + to_string(Ns_in) + "\n");

	if (this->isHardMargin)
	{
		// Description for w 
		// sum(alpah_i * y_i * x_i_d)
		this->log("Margin Weight = [");
		this->w = vector<double>(D, 0.0);
		for (size_t d = 0; d < D; d++)
		{
			for (size_t j = 0; j < Ns; j++)			
				this->w[d] += alpha_s[j] * (double)ys[j] * x[j][d];
			
			this->log(to_string(this->w[d]) + ", ");
		}
		this->log("]\n");
	}
	// Descripe for b
	// 1/n * sum(y_i - w^T K(x_i, x_j))
	this->b = 0.0;
	for (size_t i = 0; i < Ns; i++)
	{
		this->b += (double)ys[i];
		for (size_t j = 0; j < Ns; j++)
			this->b -= alpha_s[j] * (double)ys[j] * this->kernel(this->xs[j], this->xs[i], this->params);

		for (size_t j = 0; j < Ns_in; j++)
			this->b -= alpha_s_in[j] * (double)ys_in[j] * this->kernel(this->xs_in[j], this->xs[i], this->params);
	}
	this->b /= (double)Ns;

	this->log("Bias = " + to_string(this->b) + "\n");
	this->log("////////////////////////////////////////////////////////\n\n");
}

void SVM::Evaluation(const vector<vector<double>> class1Data, const vector<vector<double>> class2Data)
{
	this->correctC1 = 0;
	for (size_t i = 0; i < class1Data.size(); i++)
	{
		if (this->g(class1Data[i]) == 1)
			this->correctC1++;
	}

	this->correctC2 = 0;
	for (size_t i = 0; i < class2Data.size(); i++)
	{
		if (this->g(class2Data[i]) == -1)
			this->correctC2++;
	}

	this->accuaracy = (double)(this->correctC1 + this->correctC2) / (double)(class1Data.size() + class2Data.size());
	this->accuracyC1 = (double)this->correctC1 / (double)class1Data.size();
	this->accuracyC2 = (double)this->correctC2 / (double)class2Data.size();
}

vector<int> SVM::predict(vector<vector<double>> X)
{
	vector<int> yHat;
	for (size_t i = 0; i < X.size(); i++)
		yHat.push_back(this->g(X[i]));

	return yHat;
}

double SVM::f(const vector<double> x)
{
	// Decision Score
	double result = 0.0;
	for (size_t i = 0; i < xs.size(); i++)	
		result += alpha_s[i] * (double)ys[i] * this->kernel(this->xs[i], x, this->params);
	
	for (size_t i = 0; i < xs_in.size(); i++)
		result += alpha_s_in[i] * (double)ys_in[i] * this->kernel(this->xs_in[i], x, this->params);

	result += this->b;

	return result;
}

double SVM::g(const vector<double>x)
{
	double fx;
	int gx;
	fx = this->f(x);

	// get sample label
	if (fx >= 0.0)
		gx = 1;
	else
		gx = -1;
	return gx;
}
