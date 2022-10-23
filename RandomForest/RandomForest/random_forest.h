#ifndef RANDOM_FOREST_H
#define RANDOM_FOREST_H
#pragma once

#include "decision_tree.h"
#include "ThreadsPool.h"
#include "config.h"

using namespace std;
using namespace random_forest;

class RandomForest
{
private:
	vector<DecisionTree> m_DecisionTrees;
	int m_NumEstimatiors;
	string m_Criterion;
	string m_MaxFeatures;
	int m_MaxDepth;
	int m_MinSamplesSplit;
	int m_MinSamplesLeaf;
	int m_EachTreeSamplesNum;
	int m_n_Jobs;

	void norm(vector<double>& total);

public:
	RandomForest(int nEstimators, string criterion, string maxFeatures, int maxDepth, int minSamplesSplit, int minSamplesLeaf, int eachTreeSamplesNum, int nJobs);
	RandomForest(Config config);
	void fit(Data& trainData);
	vector<double> PredictProbability(Data& testData);
	vector<int> PredictClass(const vector<double>& probability);
};

#endif // !RANDOM_FOREST_H