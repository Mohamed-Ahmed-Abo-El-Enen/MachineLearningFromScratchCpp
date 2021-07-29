#pragma once
#ifndef NEURON_H
#define NEURON_H

#include<vector>
#include"Connection.h"
#include<cstdlib>
#include"Activations.h"

using namespace std;

class Neuron
{
private:
	double m_outputtVal;
	unsigned m_neuronIndex;
	vector<Connection> m_OutputWeights;
	ActivationFunctions m_ActivationFunction;
	double m_Gradient;
	double m_LearningRate;
	double m_AlphaMomentum;
	double RandomWeight();
	double ActivationFunction(double x);
	double ActivationFunctionDerivative(double x);
	double SumDOW(const vector<Neuron>& nextLayer) const;

public:
	Neuron(unsigned numOutputs, unsigned neuronIndex, ActivationFunctions activationFunctions = ActivationFunctions::SigmoidTag, double learningRate = 0.5, double alphaMomentum=0.9);
	~Neuron();
	void FeedForward(const vector<Neuron>& prevLayer);
	double GetOutputVal() const;
	void SetOutputVal(double val);
	void SetActivationFunction(ActivationFunctions ActivationFunction);
	void CalcualteOutputGradient(double targetVal);
	void CalculateHiddenLayerGradient(const vector<Neuron>& nextLayer);
	void UpdateInputWeights(vector<Neuron>& prevLayer);
};

#endif