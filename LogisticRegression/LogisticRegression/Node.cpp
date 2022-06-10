#include "Node.h"
#include <iostream>
#include <typeinfo>
#include <algorithm>

Node::Node(const DataFrame& data)
{
	this->pLeftChild = NULL;
	this->pRightChild = NULL;
	const vector<int> outputsBefore = data[data.size() - 1];
	this->trainingData = data;

	this->bestSplit = getBestSplit(data);
}

Node::~Node()
{}

float Node::getGiniImpurity(const vector<int>& outcomes)
{
	if (outcomes.size() == 0)
		return 0.0;

	float giniSum = 0.0;

	const int maxOutcome = *max_element(outcomes.begin(), outcomes.end());

	vector<int> counts(maxOutcome + 1, 0);

	for (int i: outcomes)
		counts[i] += 1;

	for (int i = 0; i < maxOutcome; i++)
	{
		float pClass = (float)counts[i] / (float)outcomes.size();
		giniSum += pClass * (1 - pClass);
	}

	return giniSum;
}

DataFrame Node::getSplitTargets(const DataFrame& pData, int feature, int category)
{
	vector<int> trueSplit;
	vector<int> falseSplit;

	int idx = 0;
	for (int i: pData[feature])
	{
		int val = pData[feature][idx];

		if (val == category)
			trueSplit.push_back(idx);
		else
			falseSplit.push_back(idx);

		idx++;
	}

	vector<int> allOutcomes = pData[pData.size() - 1];
	vector<int> trueOutcomes;
	vector<int> falseOutcomes;

	for (auto& itr : trueSplit)
		trueOutcomes.push_back(allOutcomes[itr]);


	for (auto& itr : falseSplit)
		falseOutcomes.push_back(allOutcomes[itr]);

	DataFrame returnData{
		trueOutcomes,
		falseOutcomes
	};

	return returnData;
}

BestSplitReturn Node::getBestSplit(const DataFrame& pData)
{
	const vector<int> outputBefore = pData[pData.size() - 1];
	const float giniBefore = Node::getGiniImpurity(outputBefore);

	// get number of features
	int features = pData.size() - 1;
	BestSplitReturn bestChoice;
	bestChoice.resultGini = giniBefore;

	bestChoice.feature = 0;
	bestChoice.category = 0;
	
	for (int feature = 0; feature < features; feature++)
	{
		vector<int> featureVec = pData[feature];
		int cats = *max_element(featureVec.begin(), featureVec.end())+1;

		for (int cat = 0; cat < cats; cat++)
		{
			DataFrame split = getSplitTargets(pData, feature, cat);
			float trueGini = getGiniImpurity(split[0]);
			float weightedTrueGini = trueGini * split[0].size() / outputBefore.size();

			float falseGini = getGiniImpurity(split[1]);
			float weightedFalseGini = falseGini * split[1].size() / outputBefore.size();

			float weightedGiniSum = weightedTrueGini + weightedFalseGini;

			if (weightedGiniSum < bestChoice.resultGini)
			{
				bestChoice.resultGini = weightedGiniSum;
				bestChoice.feature = feature;
				bestChoice.category = cat;
			}
		}
	}
	return bestChoice;
}
