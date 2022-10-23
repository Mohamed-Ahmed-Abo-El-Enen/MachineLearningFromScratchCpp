#include "decision_tree.h"

namespace xgboost
{
	BaseDecisionTree::BaseDecisionTree(Config config) :m_Config(config) {};
	
	Tree* BaseDecisionTree::fit(const vector<vector<double>>& featuresIn, const vector<int>& labelsIn, const vector<double>& gradIn, const vector<double>& hessIn)
	{
		m_Features = featuresIn;
		m_Labels = labelsIn;
		m_Grad = gradIn;
		m_Hess = hessIn;

		vector<int>subDataset;
		for (size_t i = 0; i < m_Labels.size(); i++)		
			subDataset.push_back(i);
		
		m_DecisionTree = fit(subDataset, 1);
		return m_DecisionTree;
	}

	Tree* BaseDecisionTree::fit(vector<int>& subDataset, int depth)
	{
		double subHess = 0.0;
		for (int i:subDataset)		
			subHess += m_Hess[i];

		if (subDataset.size() <= m_Config.m_MinSamplesSplit || subHess <= m_Config.m_MinChildWeight)
		{
			Tree* tree = new Tree();
			tree->m_LeafValue = CalculateLeafValue(subDataset);
			return tree;
		}

		if (depth <= m_Config.m_MaxDepth)
		{
			int datasetIndex;
			for (size_t i = 0; i < subDataset.size(); i++)			
				datasetIndex = subDataset[i];

			BestSplitInfo bestSplitInfo = ChooseBestSplitFeature(subDataset);
			Tree* tree = new Tree();

			if (bestSplitInfo.m_BestSubDatasetLeft.size() < m_Config.m_MinDataInLeaf || bestSplitInfo.m_BestSubDatasetRight.size() < m_Config.m_MinDataInLeaf)
			{
				tree->m_LeafValue = CalculateLeafValue(subDataset);
				return tree;
			}
			else
			{
				tree->m_SplitFeature = bestSplitInfo.m_BestSplitFeature;
				tree->m_SplitValue = bestSplitInfo.m_BestSplitValue;
				tree->m_SplitGain = bestSplitInfo.m_BestSplitGain;
				tree->m_InternalValue = bestSplitInfo.m_BestInternalValue;
				tree->m_TreeLeft = fit(bestSplitInfo.m_BestSubDatasetLeft, depth + 1);
				tree->m_TreeRight = fit(bestSplitInfo.m_BestSubDatasetRight, depth + 1);

				return tree;
			}
		}
		else
		{
			Tree* tree = new Tree();
			tree->m_LeafValue = CalculateLeafValue(subDataset);
			return tree;
		}
	}

	BestSplitInfo BaseDecisionTree::ChooseBestSplitFeature(const vector<int>& subDataset)
	{
		BestSplitInfo bestSplitInfo;

		double bestInternalValue = CalculateLeafValue(subDataset);
		bestSplitInfo.m_BestInternalValue = bestInternalValue;

		list<BestSplitInfo> bestSplitInfoList(m_Features[0].size(), BestSplitInfo());

		for (int i = 0; i < m_Features[0].size(); i++)		
			bestSplitInfoList.push_back(ChooseBestSplitValue(subDataset, i));
		
		list<BestSplitInfo>::iterator iter = bestSplitInfoList.begin();
		while (iter != bestSplitInfoList.end())
		{
			if ((*iter).m_BestSplitGain > bestSplitInfo.m_BestSplitGain)
			{
				bestSplitInfo.m_BestSplitGain = (*iter).m_BestSplitGain;
				bestSplitInfo.m_BestSplitFeature = (*iter).m_BestSplitFeature;
				bestSplitInfo.m_BestSplitValue = (*iter).m_BestSplitValue;
				bestSplitInfo.m_BestSubDatasetLeft = (*iter).m_BestSubDatasetLeft;
				bestSplitInfo.m_BestSubDatasetRight = (*iter).m_BestSubDatasetRight;
			}
			iter++;
		}

		return bestSplitInfo;
	}

	BestSplitInfo BaseDecisionTree::ChooseBestSplitValue(const vector<int>& subDataset, int featureIndex)
	{
		vector<double> featureValues;
		vector<double> featureValuesUnique;

		int datasetIndex;
		for (size_t i = 0; i < subDataset.size(); i++)
		{
			datasetIndex = subDataset[i];
			featureValues.push_back(m_Features[datasetIndex][featureIndex]);
			featureValuesUnique.push_back(m_Features[datasetIndex][featureIndex]);
		}

		vector<double> uniqueValues;
		sort(featureValuesUnique.begin(), featureValuesUnique.end());
		featureValuesUnique.erase(unique(featureValuesUnique.begin(), featureValuesUnique.end()), featureValuesUnique.end());

		if (featureValuesUnique.size() <= m_Config.m_MaxBin)
			uniqueValues = featureValuesUnique;

		else
		{
			vector<double> lins = numpy::LinSpace(0, 100, m_Config.m_MaxBin);
			sort(featureValues.begin(), featureValues.end());
			for (size_t i = 0; i < lins.size(); i++)
			{
				double p = lins[i];
				uniqueValues.push_back(numpy::Percentile(featureValues, p));
			}
			uniqueValues.erase(unique(uniqueValues.begin(), uniqueValues.end()), uniqueValues.end());
		}	

		vector<int> subDatasetLeft;
		vector<int> subDatasetRight;

		double leftGradSum;
		double rightGradSum;

		double leftHessSum;
		double rightHessSum;

		double splitGain;

		BestSplitInfo bestSplitInfo;
		bestSplitInfo.m_BestSplitFeature = featureIndex;

		for (double splitValue : uniqueValues)		
		{
			subDatasetLeft.clear();
			subDatasetRight.clear();

			leftGradSum = 0;
			rightGradSum = 0;

			leftHessSum = 0;
			rightHessSum = 0;

			for (int index : subDataset)
			{
				if (m_Features[index][featureIndex] <= splitValue)
				{
					subDatasetLeft.push_back(index);
					leftGradSum += m_Grad[index];
					leftHessSum += m_Hess[index];
				}
				else
				{
					subDatasetRight.push_back(index);
					rightGradSum += m_Grad[index];
					rightHessSum += m_Hess[index];
				}
			}
			splitGain = CalculateSplitGain(leftGradSum, leftHessSum, rightGradSum, rightHessSum);
			if (bestSplitInfo.m_BestSplitGain < splitGain)
			{
				bestSplitInfo.m_BestSplitGain = splitGain;
				bestSplitInfo.m_BestSplitFeature = featureIndex;
				bestSplitInfo.m_BestSplitValue = splitValue;
				bestSplitInfo.m_BestSubDatasetLeft = subDatasetLeft;
				bestSplitInfo.m_BestSubDatasetRight = subDatasetRight;

			}
		}
		return bestSplitInfo;
	}

	double BaseDecisionTree::CalculateSplitGain(const double& leftGradSum, const double& leftHessSum, const double& rightGradSum, const double& rightHessSum)
	{
		double tmp1 = pow(leftGradSum, 2) / (leftHessSum + m_Config.m_RegLambda);
		double tmp2 = pow(rightGradSum, 2) / (rightHessSum + m_Config.m_RegLambda);
		double tmp3 = pow((leftGradSum + rightGradSum), 2) / (leftHessSum + rightHessSum + m_Config.m_RegLambda);
		return (tmp1 + tmp2 - tmp3) / 2 - m_Config.m_RegGamma;
	}

	double BaseDecisionTree::CalculateLeafValue(const vector<int>& subDataset)
	{
		double gradSum = 0;
		double hessSum = 0;
		for (int index : subDataset)
		{
			gradSum += m_Grad[index];
			hessSum += m_Hess[index];
		}

		return -gradSum / (hessSum + m_Config.m_RegLambda);
	}
}