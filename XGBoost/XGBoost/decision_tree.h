#ifndef DECISION_TREE_H
#define DECISION_TREE_H
#pragma once

#include <string>
#include <vector>
#include <list>
#include <algorithm>
#include <iostream>
#include "config.h"
#include "numpy.h"
#include "pandas.h"
#include "tree.h"

using namespace std;

namespace xgboost
{
	struct BestSplitInfo
	{
		int m_BestSplitFeature = 0;
		double m_BestSplitValue = 0;
		double m_BestSplitGain = -1e10;
		double m_BestInternalValue = 0;

		vector<int> m_BestSubDatasetLeft;
		vector<int> m_BestSubDatasetRight;
	};

	class BaseDecisionTree
	{
	private:
		const Config m_Config;
		Tree* m_DecisionTree;
		vector<vector<double>> m_Features;
		vector<int> m_Labels;
		vector<double> m_Grad;
		vector<double> m_Hess;

		Tree* fit(vector<int>& subDataset, int depth);
		BestSplitInfo ChooseBestSplitFeature(const vector<int>& subDataset);
		BestSplitInfo ChooseBestSplitValue(const vector<int>& subDataset, int featureIndex);
		double CalculateLeafValue(const vector<int>& subDataset);
		double CalculateSplitGain(const double& leftGradSum, const double& leftHessSum, const double& rightGradSum, const double& rightHessSum);

	public:
		BaseDecisionTree(Config config);
		~BaseDecisionTree() = default;
		Tree* fit(const vector<vector<double>>& featuresIn, const vector<int>& labelsIn, const vector<double>& gradIn, const vector<double>& hessIn);

	};
}

#endif // !DECISION_TREE_H