#include "Kernel.h"

double kernel::linear(const vector<double> x1, const vector<double> x2, const vector<double> params)
{
	if (x1.size() != x2.size())
	{
		cerr << "ERROR: Could't match the number of elements for Dot Product" << endl;
		exit(-1);
	}

	double result = 0.0;

	for (size_t i = 0; i < x1.size(); i++)
		result += x1[i] * x2[i];

	return result;
}

double kernel::polynomial(const vector<double> x1, const vector<double> x2, const vector<double> params)
{
	if (x1.size() != x2.size())
	{
		cerr << "ERROR: Could't match the number of elements for Dot Product" << endl;
		exit(-1);
	}

	if (params.size() != 2)
	{
		cerr << "ERROR: Could't match the number of hyper-parameters" << endl;
		exit(-1);
	}

	double result = 0.0;

	for (size_t i = 0; i < x1.size(); i++)
		result += x1[i] * x2[i];

	result += params[0];
	result = pow(result, params[1]);

	return result;
}

double kernel::rbf(const vector<double> x1, const vector<double> x2, const vector<double> params)
{
	if (x1.size() != x2.size())
	{
		cerr << "ERROR: Could't match the number of elements for Dot Product" << endl;
		exit(-1);
	}

	if (params.size() != 1)
	{
		cerr << "ERROR: Could't match the number of hyper-parameters" << endl;
		exit(-1);
	}

	double result = 0.0;

	for (size_t i = 0; i < x1.size(); i++)
		result += (x1[i] - x2[i]) * (x1[i] - x2[i]);

	
	result = exp(-params[0] * result);

	return result;
}