#ifndef NETWORKMODEL_H
#define NETWORKMODEL_H

#include "Module.h"
#include "OutputLayer.h"
#include "LRScheduler.h"
#include "MNISTDataLoader.h"

class NetworkModel
{
private:
	vector<Module*> m_Modules;
	OutputLayer* m_OutputLayer;
	LRScheduler* m_LearnScheduler;
	int iteration = 0;

	unique_ptr<Tensor<double>> Forward(unique_ptr<Tensor<double>>& x);
	double Backward(vector<int>& y);
	double TrainStep(unique_ptr<Tensor<double>>& xSmaples, vector<int>& ySamples);
public:
	NetworkModel(vector<Module*>& modules, OutputLayer* outputLayer, LRScheduler* learnScheduler);
	virtual ~NetworkModel() = default;
	vector<int> predict(Tensor<double> xSamples);
	void compile(Tensor<double> xSamples, bool verbos=true);
	void train(int epochs, MNISTDataLoader trainLoader, bool verbos = true);
	void eval();
	void load(string path);
	void save(string path);
};
#endif // !NETWORKMODEL_H
