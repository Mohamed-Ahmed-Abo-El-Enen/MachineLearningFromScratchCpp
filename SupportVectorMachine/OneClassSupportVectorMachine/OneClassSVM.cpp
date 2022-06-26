#include "OneClassSVM.h"

OneClassSVM::OneClassSVM(const kernelFunction _kernel, const vector<double> _params,const bool _verbose)
{
	this->kernel = _kernel;
	this->params = _params;
	this->verbose = _verbose;
}

OneClassSVM::~OneClassSVM()
{
}

void OneClassSVM::log(const string str)
{
	if (this->verbose)
		cout << str << flush;
	return;
}

void OneClassSVM::pairsSort(vector<pair<double, int>>& data)
{
	sort(data.begin(), data.end(), [](auto& left, auto& right) 
		{
			return left.first > right.first;
		});
}

void OneClassSVM::train(const vector<vector<double>> x, const size_t D, const double nu, const double lr, const double limit)
{
	constexpr double eps = 0.0000001;
	size_t N = x.size();
	vector<double> alpha = vector<double>(N, 0.0);
	double C = 1.0 / ((double)N * nu);
	double beta = 1.0;
	double delta;

	//Training 
	this->log("\n");
	this->log("/////////////////////// Training ///////////////////////\n");

	bool judge = true;
	double error;

	const double breakEps = 0.000001;
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
		// dl/dalpha = sum_j(alpha_j * K(X_i^T, X_j)) + beta * sum(alpha_j - 1)
		for (size_t i = 0; i < N; i++)
		{
			// Set item1
			item1 = 0.0;
			for (size_t j = 0; j < N; j++)
				item1 += alpha[j] * this->kernel(x[i], x[j], this->params);

			// Set item2
			item2 = 0.0;
			for (size_t j = 0; j < N; j++)
				item2 += (alpha[j] - 1);

			// Set delta
			delta = item1 + beta * item2;
			
			// Update
			// alpha = alpha - lr * dl/dalpha
			alpha[i] -= lr * delta;

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
		// beta = beta + 1/2 * sum(alpha_i - 1)^2
		item3 = 0.0;
		for (size_t i = 0; i < N; i++)		
			item3 +=  (alpha[i]-1);
		
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
	this->alpha_s = vector<double>();

	int Ns_in = 0;
	this->xs_in = vector<vector<double>>();
	this->alpha_s_in = vector<double>();

	for (int i = 0; i < N; i++)
	{
		if ((alpha[i] > eps) && (alpha[i] < C-eps))
		{
			this->xs.push_back(x[i]);
			this->alpha_s.push_back(alpha[i]);
			Ns++;
		}

		else if (alpha[i] >= C - eps)
		{
			this->xs_in.push_back(x[i]);
			this->alpha_s_in.push_back(alpha[i]);
			Ns_in++;
		}
	}

	this->log("Ns (number of support vectors on margin) = " + to_string(Ns) + "\n");
	this->log("Ns (number of support vectors inside margin) = " + to_string(Ns_in) + "\n");


	// Descripe for b
	// 1/ns * sum(alpha * K(x_i, x_j)) where S = {s|0<alpha_s<1/nu}
	this->b = 0.0;
	for (size_t i = 0; i < Ns; i++)
	{
		for (size_t j = 0; j < Ns; j++)
			this->b += alpha_s[j] * this->kernel(this->xs[j], this->xs[i], this->params);

		for (size_t j = 0; j < Ns_in; j++)
			this->b += alpha_s_in[j] * this->kernel(this->xs_in[j], this->xs[i], this->params);
	}
	this->b /= (double)Ns;

	this->log("Bias = " + to_string(this->b) + "\n");
	this->log("////////////////////////////////////////////////////////\n\n");
}

void OneClassSVM::Evaluation(const vector<vector<double>> normalData, const vector<vector<double>> anomalyData)
{
	vector<pair<double, int>> score;
	this->correctN = 0;
	for (size_t i = 0; i < normalData.size(); i++)
	{
		score.push_back({ this->f(normalData[i]), 1 });
		if (this->g(normalData[i]) == 1)
			this->correctN++;
	}

	this->correctA = 0;
	for (size_t i = 0; i < anomalyData.size(); i++)
	{
		score.push_back({ this->f(anomalyData[i]), -1 });
		if (this->g(anomalyData[i]) == -1)
			this->correctA++;
	}

	this->roc(score);

	this->accuaracy = (double)(this->correctN + this->correctA) / (double)(normalData.size() + anomalyData.size());
	this->accuracyN = (double)this->correctN / (double)normalData.size();
	this->accuracyA = (double)this->correctA / (double)anomalyData.size();

	cout << "///////////////////////// Test /////////////////////////" << endl;
	cout << "accuracy-all: " << this->accuaracy << " (" << this->correctN + this->correctA << "/" << normalData.size() + anomalyData.size() << ")" << endl;
	cout << "accuracy-normal: " << this->accuracyN << " (" << this->correctN << "/" << normalData.size() << ")" << endl;
	cout << "accuracy-anomaly: " << this->accuracyA << " (" << this->correctA << "/" << anomalyData.size() << ")" << endl;
	cout << "AUROC: " << this->auroc << endl;
	cout << "////////////////////////////////////////////////////////" << endl;
}

vector<int> OneClassSVM::predict(vector<vector<double>> X)
{
	vector<int> yHat;
	for (size_t i = 0; i < X.size(); i++)
		yHat.push_back(this->g(X[i]));

	return yHat;
}

// f(X) = sum(alpha * kernel(X, X)) - b 
double OneClassSVM::f(const vector<double> x)
{
	// Decision Score
	double result = 0.0;
	for (size_t i = 0; i < xs.size(); i++)	
		result += alpha_s[i] * this->kernel(this->xs[i], x, this->params);
	
	for (size_t i = 0; i < xs_in.size(); i++)
		result += alpha_s_in[i] * this->kernel(this->xs_in[i], x, this->params);

	result -= this->b;

	return result;
}

// if  f(x) > 0 then 1 else -1
double OneClassSVM::g(const vector<double>x)
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

void OneClassSVM::roc(const vector<pair<double, int>> score)
{
	vector<pair<double, int>> sortedScore = score;
	this->pairsSort(sortedScore);

	size_t Np = 0, Nn = 0;
	for (size_t i = 0; i < sortedScore.size(); i++)
	{
		if (sortedScore[i].second == -1)
			Np++;
		else
			Nn++;
	}

	size_t TP=0, FP=0;
	size_t TPRate = 0, FPRate = 0;
	size_t preTPRate = 0, preFPRate = 0;
	for (size_t i = 0; i < sortedScore.size()-1; i++)
	{
		if (sortedScore[i].second == -1)
			TP++;
		else
			FP++;

		if (sortedScore[i].first != sortedScore[i + 1].first)
		{
			TPRate = (double)TP / (double)Np;
			FPRate = (double)FP / (double)Nn;

			this->auroc += (TPRate + preTPRate) * (FPRate - preFPRate) * 0.5;
			preTPRate = TPRate;
			preFPRate = FPRate;
		}
	}
	this->auroc += (1.0 + preTPRate) * (1.0 - preFPRate) * 0.5;
}

