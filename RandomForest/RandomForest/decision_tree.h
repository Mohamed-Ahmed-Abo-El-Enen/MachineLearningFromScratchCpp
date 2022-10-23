#ifndef DECISION_TREE_H
#define DECISION_TREE_H
#pragma once

#include <set>
#include <functional>
#include <string>
#include "utils.h"
#include "node.h"

using namespace std;

class DecisionTree
{
private:
	shared_ptr<Node> m_Root;
	int m_NumFeatures;
	int m_MaxDepth;
	int m_MinSamplesSplit;
	int m_MinSamplesLeaf;
	int m_SampleNum;

	function<double(int&, int&, int&, int&)> CriterionFunction;
	function<int(int)> MaxFeatureFunction;

	set<double> GetValuesRange(int& featureIndex, vector<int>& samplesVec, Data& data);
	void SplitSamplesVec(int& featureIndex, double& threshold, vector<int>& samplesVec, vector<int>& sampelesLeft, vector<int>& samplesRight, Data& data);
	void ChooseBestSplitFeatures(shared_ptr<Node>& node, vector<int>& samplesVec, Data& data);
	shared_ptr<Node> ConstructNode(vector<int>& sampleVec, Data& trainData, int depth);
	void SortByFeatures(vector<pair<int, double>>& samplesFeaturesVec, int featureIndex, Data& data);

public:
	explicit DecisionTree(const string& Criterion = "gini", int maxDepth = -1, int minSamplesSplit = 2, int minSamplesLeaf = 1, int SampleNum = -1, const string& maxFeatures = "auto");
	void fit(Data& trainData);
	double ComputeProbability(int sampleIndex, Data& data);
	void PredictProbability(Data& data, vector<double>& results);
};

#endif // !DECISION_TREE_H

