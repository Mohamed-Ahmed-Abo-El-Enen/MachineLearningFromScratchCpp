#pragma once
#ifndef SVM_HPP
#define SVM_HPP

#include <string>
#include "Kernel.h"

using namespace std;

class SVM
{
private:
	bool verbose;
	kernelFunction kernel;
	vector<double> params;
	vector<double> w;
	vector<double> w_in;
	bool isHardMargin;
	double b;

	vector<vector<double>> xs;
	vector<int> ys;
	vector<double> alpha_s;

	vector<vector<double>> xs_in;
	vector<int> ys_in;
	vector<double>alpha_s_in;

	void log(const string str);
public:
	double accuaracy;
	double accuracyC1, accuracyC2;
	size_t correctC1, correctC2;

	SVM() = delete;
	SVM(const bool _verbose = true);
	SVM(const kernelFunction _kernel, const vector<double> _params, const bool _verbose);
	~SVM();

	void train(const vector<vector<double>> X, const vector<int> y, const size_t D, const double C, const double lr, const double limit = 0.0001);
	void Evaluation(const vector<vector<double>> class1Data, const vector<vector<double>> class2Data);
	vector<int> predict(const vector<vector<double>> X);
	double f(const vector<double>x);
	double g(const vector<double> x);
};

#endif