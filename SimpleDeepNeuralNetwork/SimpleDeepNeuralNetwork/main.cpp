#include"Net.h"
#include"Activations.h"
#include"FileHandler.h"
#include <cassert>
#include "Utils.h"

using namespace std;

int main()
{
	vector<LayerCharacteristic> layers;
	layers.push_back(LayerCharacteristic(2, ActivationFunctions::TanhTag, 0.15, 0.5));
	layers.push_back(LayerCharacteristic(4, ActivationFunctions::TanhTag, 0.15, 0.5));
	layers.push_back(LayerCharacteristic(3, ActivationFunctions::TanhTag, 0.15, 0.5));
	layers.push_back(LayerCharacteristic(1, ActivationFunctions::SigmoidTag, 0.15, 0.5));
	Net net(layers);

	string datasetFilePath = "..\\Debug\\XORData.txt";
	vector<vector<double>> datasetFeatures;
	vector<vector<double>> datasetTargets;

	FileHandler fileHandler;
	fileHandler.ReadDatasetFile(datasetFilePath, datasetFeatures, datasetTargets);

	net.Compile(LossFunction::LossFunctionTags::RMSTag);

	int epochs = 5;
	net.Fit(datasetFeatures, datasetTargets, epochs, true);
	// Report how well the training is working, averaged over the recent score error
	cout << "Model recent average error: " << net.GetRecentAverageError() << endl;

	vector<double> predictedVals;
	net.Predict(datasetFeatures, predictedVals, true);
	return 0;
}