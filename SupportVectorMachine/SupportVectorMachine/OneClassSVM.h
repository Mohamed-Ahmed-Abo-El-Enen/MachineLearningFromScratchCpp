#pragma once
#ifndef SVM_HPP
#define SVM_HPP

#include <string>
#include "Kernel.h"

using namespace std;

class OneClassSVM
{
private:
	bool verbose;
	double b;
	vector<vector<double>> xs;
	vector<double> alpha_s;

	vector<vector<double>> xs_in;
	vector<double>alpha_s_in;

	kernelFunction kernel;
	vector<double> params;

	void log(const string str);
	double dotProduct(const vector<double> x1, const vector<double> x2);
	void sort(vector<pair<double, int>>& data);
public:
	double accuaracy;
	double accuracyN, accuracyA;
	double auroc;
	size_t correctN, correctA;

	OneClassSVM() = delete;
	OneClassSVM(const kernelFunction _kernel = kernel::rbf, const vector<double> _params = {1.0}, const bool _verbose=true);
	~OneClassSVM();

	void train(const vector<vector<double>> X, const size_t D, const double nu, const double lr, const double limit = 0.0001);
	void Evaluation(const vector<vector<double>> normalData, const vector<vector<double>> anomalyData);
	void roc(const vector<pair<double, int>> score);
	vector<int> predict(const vector<vector<double>> X);
	double f(const vector<double>x);
	double g(const vector<double> x);
};

#endif