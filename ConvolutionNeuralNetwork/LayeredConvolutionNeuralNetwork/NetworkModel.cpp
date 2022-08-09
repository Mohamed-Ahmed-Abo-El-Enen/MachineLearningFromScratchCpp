#include "NetworkModel.h"

NetworkModel::NetworkModel(vector<Module*>& modules, OutputLayer* outputLayer, LRScheduler* learnScheduler)
{
	m_Modules = modules;
	m_OutputLayer = outputLayer;
	m_LearnScheduler = learnScheduler;
}

void NetworkModel::compile(Tensor<double> x, bool verbos)
{
	unique_ptr<Tensor<double>> xPtr = make_unique<Tensor<double>>(x);
	for (int i = 0; i < m_Modules.size(); i++)
	{
		m_Modules[i]->Compile(xPtr);
		vector<int> tensorShape = m_Modules[i]->GetStandardOutputTensorDims();
		xPtr = make_unique<Tensor<double>>(tensorShape);
		xPtr->zero();
		if(verbos)
			cout << "Layer Number " << i + 1 << " : " << m_Modules[i]->GetLayerInfo(xPtr) << endl;
	}
	m_OutputLayer->Compile(xPtr);
	if (verbos)
		cout << "Cost Function : " << m_OutputLayer->GetcostFunctionInfo() << endl;
}

unique_ptr<Tensor<double>> NetworkModel::Forward(unique_ptr<Tensor<double>>& x)
{
	for (int i = 0; i < m_Modules.size(); i++)
	{
		x = move(m_Modules[i]->ForwardPropagate(x));
	}
	return m_OutputLayer->predict(x);
}

double NetworkModel::Backward(vector<int>& y)
{
	unique_ptr<vector<int>> yPtr = make_unique<vector<int>>(y);
	auto loss_costGradient = m_OutputLayer->BackwardPropagate(yPtr);
	unique_ptr<Tensor<double>> chainGradient = make_unique<Tensor<double>>(*loss_costGradient.second);
	for (int i = m_Modules.size() - 1; i >= 0; i--)
	{
		chainGradient = move(m_Modules[i]->BackwardPropagate(chainGradient, m_LearnScheduler->learningRate));
	}
	return loss_costGradient.first;
}

double NetworkModel::TrainStep(unique_ptr<Tensor<double>>& xSmaples, vector<int>& ySamples)
{
	// forward
	unique_ptr<Tensor<double>> output = move(Forward(xSmaples));

	// backward
	double cost = Backward(ySamples);

	iteration++;
	m_LearnScheduler->OnIterationEnd(iteration);
	
	return cost;
}

void PrintProgress(float progress, int iterationNumber, int numTrainBatches, double loss)
{
	int barWidth = 50;
	cout << "[";
	int pos = barWidth * progress;
	for (int i = 0; i < barWidth; ++i) 
	{
		if (i < pos) 
			cout << "=";
		else if (i == pos) 
			cout << ">";
		else 
			cout << " ";
	}
	cout << "] " << int(progress * 100.0) << " Iteration " << iterationNumber << "/" << numTrainBatches << " - Batch loss: " << loss << "\r";
	cout.flush();

}

void NetworkModel::train(int epochs, MNISTDataLoader trainLoader, bool verbos)
{
	printf("Training for %d epochs(s).\n", epochs);

	int numTrainBatches = trainLoader.GetNumBatches();
	float progressStep = 1.0 / numTrainBatches;
	float progress = 0;
	double lastLoss = DBL_MAX;
	for (int i = 0; i < epochs; i++)
	{
		progress = progressStep;
		printf("Epochs %d \n", i + 1);
		for (int j = 0; j < numTrainBatches; j++)
		{
			auto xy = trainLoader.NextBatch();
			unique_ptr<Tensor<double>> x = move(xy.first);
			vector<int> y = xy.second;
			double loss = TrainStep(x, y);
			lastLoss = loss;
			if (verbos)
			{
				PrintProgress(progress, j + 1, numTrainBatches, lastLoss);
				progress += progressStep;
			}			
		}
		printf("\n");
	}
}

vector<int> NetworkModel::predict(Tensor<double> xSamples)
{
	unique_ptr<Tensor<double>> xPtr = make_unique<Tensor<double>>(xSamples);
	Tensor<double> output = Forward(xPtr);
	vector<int> predictions; 
	for (int i = 0; i < output.m_Dims[0]; i++)
	{
		int argmax = -1;
		double max = -1;
		for (int j = 0; j < output.m_Dims[1]; j++)
		{
			if (output.get(i, j) > max)
			{
				max = output.get(i, j);
				argmax = j;
			}
		}
		predictions.push_back(argmax);
	}
	return predictions;
}

void NetworkModel::eval()
{
	for (auto& module:m_Modules)
		module->eval();	
}

void NetworkModel::save(string path)
{
	FILE* modelFile;
	errno_t err = fopen_s(&modelFile, path.c_str(), "w");
	if (err!=0)
		throw runtime_error("ERROR: saving model file.");

	for (int i = 0; i < m_Modules.size(); i++)
		m_Modules[i]->save(modelFile);
	fclose(modelFile);
}

void NetworkModel::load(string path)
{
	FILE* modelFile;
	errno_t err = fopen_s(&modelFile, path.c_str(), "r");
	if (err!=0)
		throw runtime_error("ERROR: laoding model file.");
	for (int i = 0; i < m_Modules.size(); i++)
		m_Modules[i]->load(modelFile);
	fclose(modelFile);
}
