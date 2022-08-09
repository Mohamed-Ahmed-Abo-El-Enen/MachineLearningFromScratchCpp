#ifndef UTIL_H
#define UTIL_H

#include "Matrix.h"

struct Shape
{
	size_t rows;
	size_t columns;
};

namespace fns
{
	double relu(double x);
	double sigmoid(double x);
	double tan(double x);
	double relu_gradient(double x);
	double sigmoid_gradient(double x);
	double tan_gradient(double x);
	double softmax(double x);
}

namespace Prerocess
{
	void ProcessMNISTImage(const char* filePath, vector<unique_ptr<Matrix>>& xSample, vector<unique_ptr<vector<double>>>& ySample, size_t nImages = 100);
	void ProcessMNISTCSV(const char* filePath, vector<vector<double>>& xSample, vector<vector<double>>& ySample);
	void ProcessImage(const char* filePath);
}

#endif // !UTIL_H
