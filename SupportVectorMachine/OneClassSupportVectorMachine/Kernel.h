#pragma once
#ifndef KERNEL_HPP
#define KERNEL_HPP

#include <iostream>
#include <vector>
#include <functional>
#include <math.h>

using namespace std;

namespace kernel
{
	double linear(const vector<double> x1, vector<double> x2, const vector<double> params);
	double polynomial(const vector<double> x1, vector<double> x2, const vector<double> params);
	double rbf(const vector<double> x1, vector<double> x2, const vector<double> params);
}
typedef function<double(const vector<double>, const vector<double>, const vector<double>)> kernelFunction;

#endif