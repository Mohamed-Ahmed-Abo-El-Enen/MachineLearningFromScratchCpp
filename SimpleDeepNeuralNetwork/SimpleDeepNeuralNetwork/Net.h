#pragma once
#ifndef NET_H
#define NET_H

#include<vector>
#include<iostream>
#include<cassert>
#include<string>
#include "Neuron.h"
#include "LayerCharacteristic.h"
#include "LossFunction.h"

using namespace std;

typedef vector<Neuron> Layer;

class Net
{
private:
	vector<Layer> m_layers; //m_layer[layerNum][neuroNum]
	void FeedFirstLayer(const vector<double>& inputVals);
	void ForwardPropagate();
	double m_error;
	double m_RecentAverageError;
	double m_ReccentAverageSmothingFactor;
	LossFunction::LossFunctionTags m_LossfuntionTag;
	void UpdateWeight();
	void GradientFirstLayer(Layer& outputLayer, const vector<double>& targetVals);
	void GradientHiddenLayer();
	void CalculateGradient(Layer& outputLayer, const vector<double>& targetVals);
	void FeedForward(const vector<double>& inputVals);
	void BackPropagate(const vector<double>& targeVals);
	void GetResults(vector<double>& resultVals) const;
	void ShowVectorVals(string label, vector<double>& v);
	void GetLossFunctionError(const vector<double>& targetVals, vector<Neuron>& outputLayer);

public:
	Net(const vector<LayerCharacteristic>& layers);
	double GetRecentAverageError(void) const { return m_RecentAverageError; }
	void Compile(LossFunction::LossFunctionTags lossFunctionTag = LossFunction::RMSTag);
	unsigned Fit(vector<vector<double>>& samplesFeatures, vector<vector<double>>& samplesTarget, int epochs, bool verbose);
	unsigned Predict(vector<vector<double>>& samplesFeatures, vector<double>& predictedVals, bool printWhilePredict);
};

#endif