#include "data.h"
#include "decision_tree.h"
#include "random_forest.h"
#include "pandas.h"
#include "utils.h"
#include "config.h"

using namespace random_forest;

int main()
{
	clock_t startTime, endTime;
	startTime = clock();

	string datasetPath = "../Resources/pima indians.csv";
	pandas::Dataset dataset = pandas::ReadCSV(datasetPath, ',', -1);
	Data trainData(true);
	trainData.PrepareData(dataset.features, dataset.labels);

	Config config;
	config.m_NumEstimatiors = 10;
	config.m_MaxDepth = -1;
	config.m_MinSamplesSplit = 150;
	config.m_MinSamplesLeaf = 1;
	config.m_EachTreeSamplesNum = -1;
	config.m_n_Jobs = 8;
	config.m_Criterion = "gini";
	config.m_MaxFeatures = "logs";

	RandomForest randomForest(config);

	randomForest.fit(trainData);

	datasetPath = "../Resources/pima indians.csv";
	dataset = pandas::ReadCSV(datasetPath, ',', -1);
	Data testData(true);
	testData.PrepareData(dataset.features, dataset.labels);

	auto resultsProbability = randomForest.PredictProbability(testData);
	auto yHat = randomForest.PredictClass(resultsProbability);

	cout << "ACC: " << utils::CalcualteACC(dataset.labels, yHat) << endl;

	endTime = clock();
	cout << "Total Time: " << (double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << endl;

	utils::WriteDataToCSV(yHat, testData, "../Resources/Results.csv", true);

	system("pause");

	return 0;
}