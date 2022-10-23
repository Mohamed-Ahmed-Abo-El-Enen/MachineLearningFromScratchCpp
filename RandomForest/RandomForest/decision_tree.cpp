#include "random_forest.h"

set<double> DecisionTree::GetValuesRange(int& featureIndex, vector<int>& samplesVec, Data& data)
{
	set<double> featureRange;
	for (auto sampleIndex : samplesVec) 
		featureRange.insert(data.ReadFeature(sampleIndex, featureIndex));
	
	return featureRange;
}

void DecisionTree::SplitSamplesVec(int& featureIndex, double& threshold, vector<int>& samplesVec, vector<int>& leftSamples, vector<int>& rightSamples, Data& data)
{
	leftSamples.clear();
	rightSamples.clear();
	for (auto samplesIndex : samplesVec) 
	{
		if (data.ReadFeature(samplesIndex, featureIndex) > threshold) 
			rightSamples.push_back(samplesIndex);		
		else 
			leftSamples.push_back(samplesIndex);		
	}
}

void DecisionTree::SortByFeatures(vector<pair<int, double>>& samplesFeaturesVec, int featureIndex, Data& data)
{
	for (int i = 0; i < samplesFeaturesVec.size(); i++) 
		samplesFeaturesVec[i].second = data.ReadFeature(samplesFeaturesVec[i].first, featureIndex);
	
	sort(samplesFeaturesVec.begin(), samplesFeaturesVec.end(), [](pair<int, double>& a, pair<int, double>& b) 
		{
			return a.second < b.second;
		});
}

void DecisionTree::ChooseBestSplitFeatures(shared_ptr<Node>& node, vector<int>& samplesVec, Data& data)
{
	vector<int> featuresVec = data.GenerateFeatures(MaxFeatureFunction);
	int bestFeatureIndex = featuresVec[0];
	int samplesTrueNum = utils::ComputeTrue(samplesVec, data);
	double minValue = DBL_MAX;
	double bestThreshold = 0.0;
	double threshold = 0.0;
	int sampleIndex;

	vector<pair<int, double>> samplesFeaturesVec;
	samplesFeaturesVec.reserve(samplesVec.size());

	for (auto index : samplesVec) 
		samplesFeaturesVec.emplace_back(index, 0);	

	for (auto featureIndex : featuresVec) 
	{
		SortByFeatures(samplesFeaturesVec, featureIndex, data);
		int leftSize = 0;
		int rightSize = (int)samplesVec.size();
		int leftTrue = 0;
		int rightTrue = samplesTrueNum;
		for (int index = 0; index < samplesFeaturesVec.size();) 
		{
			sampleIndex = samplesFeaturesVec[index].first;
			threshold = samplesFeaturesVec[index].second;
			while (index < samplesVec.size() &&
				samplesFeaturesVec[index].second <= threshold)
			{
				leftSize++;
				rightSize--;
				if (data.ReadTarget(sampleIndex) == 1) 
				{
					leftTrue++;
					rightTrue--;
				}
				sampleIndex = samplesFeaturesVec[index].first;
				index++;
			}

			if (index == samplesVec.size())
				continue; 

			double value = CriterionFunction(leftTrue, leftSize, rightTrue, rightSize);
			if (value <= minValue) {
				minValue = value;
				bestThreshold = threshold;
				bestFeatureIndex = featureIndex;
			}
		}
	}
	node->m_FeatureIndex = bestFeatureIndex;
	node->m_Threshold = bestThreshold;
}

shared_ptr<Node> DecisionTree::ConstructNode(vector<int>& samplesVec, Data& trainData, int depth)
{
	double targetProb = utils::ComputeTargetProbability(samplesVec, trainData);
	shared_ptr<Node> node(new Node());
	node->m_Depth = depth;
	node->m_Probability = 0;

	if (targetProb == 0 || targetProb == 1 || samplesVec.size() <= m_MinSamplesSplit || depth == m_MaxDepth) 
	{
		node->m_IsLeaf = true;
		node->m_Probability = targetProb;
	}
	else 
	{
		ChooseBestSplitFeatures(node, samplesVec, trainData);
		vector<int> leftSamples;
		vector<int> rightSamples;
		SplitSamplesVec(node->m_FeatureIndex, node->m_Threshold, samplesVec, leftSamples, rightSamples, trainData);
		if ((leftSamples.size() < m_MinSamplesLeaf) or (rightSamples.size() < m_MinSamplesLeaf))
		{
			node->m_IsLeaf = true;
			node->m_Probability = targetProb;
		}
		else 
		{
			node->m_LeftTree = ConstructNode(leftSamples, trainData, depth + 1);
			node->m_RightTree = ConstructNode(rightSamples, trainData, depth + 1);
		}
	}
	return node;
}

DecisionTree::DecisionTree(const string& criterion, int maxDepth, int minSamplesSplit, int minSamplesLeaf, int sampleNum, const string& maxFeatures)
{	
	CriterionFunction = utils::ComputeGiniIndex;
	
	if (maxFeatures == "auto" || maxFeatures == "sqrt")
		MaxFeatureFunction = utils::_sqrt;
	else if (maxFeatures == "log2")
		MaxFeatureFunction = utils::_log2;
	else
		MaxFeatureFunction = utils::_none;

	m_SampleNum = sampleNum;
	m_MaxDepth = maxDepth;
	m_MinSamplesSplit = minSamplesSplit;
	m_MinSamplesLeaf = minSamplesLeaf;
}

void DecisionTree::fit(Data& trainData)
{
	vector<int> samplesVec = trainData.GenerateSample(m_SampleNum);
	m_Root = ConstructNode(samplesVec, trainData, 0);
}

double DecisionTree::ComputeProbability(int sampleIndex, Data& data)
{
	auto node = m_Root;
	while (!node->m_IsLeaf)
	{
		if (data.ReadFeature(sampleIndex, node->m_FeatureIndex) > node->m_Threshold)
			node = node->m_RightTree;
		else
			node = node->m_LeftTree;
	}
	return node->m_Probability;
}

void DecisionTree::PredictProbability(Data& data, vector<double>& results)
{
	for (int i = 0; i < results.size(); i++)	
		results[i] += ComputeProbability(i, data);
}