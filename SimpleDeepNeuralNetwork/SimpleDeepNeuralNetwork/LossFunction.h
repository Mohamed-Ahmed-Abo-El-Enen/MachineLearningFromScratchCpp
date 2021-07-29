#pragma once
#include<vector>
#include"Neuron.h"

using namespace std;

class LossFunction
{
public:
	static double RMS(const vector<double>& targetVals, vector<Neuron>& currentVals);
	static double MSE(const vector<double>& targetVals, vector<Neuron>& currentVals);
	static enum LossFunctionTags
	{
		MSETag=0,
		RMSTag=1
	};
};
