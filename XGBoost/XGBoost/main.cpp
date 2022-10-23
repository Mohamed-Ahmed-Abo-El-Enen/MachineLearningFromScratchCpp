#include <vector>
#include <iostream>
#include <fstream>
#include <time.h>
#include <list>
#include "config.h"
#include "pandas.h"
#include "numpy.h"
#include "xgboost.h"
#include "tree.h"
#include "utils.h"

using namespace std;
using namespace xgboost;
using namespace pandas;
using namespace numpy;

int main()
{
	clock_t startTime, endTime;
	startTime = clock();

	Config config;
	config.m_NumEstimators = 5;
	config.m_LearningRate = 0.4;
	config.m_MaxDepth = 6;
	config.m_MinSamplesSplit = 50;
	config.m_MinDataInLeaf = 20;
	config.m_RegGamma = 0.3;
	config.m_RegLambda = 0.3;
	config.m_ColSampleByTree = 0.8;
	config.m_MinChildWeight = 5;
	config.m_MaxBin = 100;

	string datasetPath = "../Resources/pima indians.csv";
	pandas::Dataset dataset = pandas::ReadCSV(datasetPath, ',', -1);

	XGBoost xgboost = XGBoost(config);
	xgboost.fit(dataset.features, dataset.labels);

	vector<double> yHat;
	for (size_t i = 0; i < dataset.labels.size(); i++)	
		yHat.push_back(xgboost.PredictProba(dataset.features[i])[1]);

	cout << "AUC: " << CalcualteAUC(dataset.labels, yHat) << endl;
	cout << "KS: " << CalcualteKS(dataset.labels, yHat) << endl;
	cout << "ACC: " << CalcualteACC(dataset.labels, yHat) << endl;

	endTime = clock();
	cout << "Total Time: " << (double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << endl;

	system("pause");

	return 0;
}