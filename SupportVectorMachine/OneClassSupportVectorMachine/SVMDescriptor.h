#pragma once
#ifndef SUPPORTVECTORDESCRIPTOR_HPP
#define SUPPORTVECTORDESCRIPTOR_HPP

#include <string>
#include <algorithm>
#include "Kernel.h"

using namespace std;

class SVMDescriptor
{
private:
	bool verbose;
	double b;
	double R;

	vector<vector<double>> xs;
	vector<double> alpha_s;

	vector<vector<double>> xs_out;
	vector<double>alpha_s_out;

	kernelFunction kernel;
	vector<double> params;

	void log(const string str);
	void pairsSort(vector<pair<double, int>>& data);
public:
	double accuaracy;
	double accuracyN, accuracyA;
	double auroc;
	size_t correctN, correctA;

	SVMDescriptor() = delete;
	SVMDescriptor(const kernelFunction _kernel = kernel::rbf, const vector<double> _params = { 1.0 }, const bool _verbose = true);
	~SVMDescriptor();

	void train(const vector<vector<double>> X, const size_t D, const double nu, const double lr, const double limit = 0.0001);
	void Evaluation(const vector<vector<double>> normalData, const vector<vector<double>> anomalyData);
	void roc(const vector<pair<double, int>> score);
	vector<int> predict(const vector<vector<double>> X);
	double f(const vector<double>x);
	double g(const vector<double> x);
};

#endif