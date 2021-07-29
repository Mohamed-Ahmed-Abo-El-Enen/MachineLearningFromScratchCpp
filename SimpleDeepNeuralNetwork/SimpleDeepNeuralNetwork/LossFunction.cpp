#include"LossFunction.h"

double LossFunction::RMS(const vector<double>& targetVals, vector<Neuron>& currentVals)
{
	double error = 0;
	for (unsigned n = 0; n < currentVals.size() - 1; n++)
	{

		double delta = targetVals[n] - currentVals[n].GetOutputVal();
		error += pow(delta, 2);
	}
	error /= currentVals.size();
	return sqrt(error);
}

double LossFunction::MSE(const vector<double>& targetVals, vector<Neuron>& currentVals)
{
	double error = 0;
	for (unsigned n = 0; n < currentVals.size() - 1; n++)
	{

		double delta = targetVals[n] - currentVals[n].GetOutputVal();
		error += pow(delta, 2);
	}
	error /= currentVals.size();
	return error;
}