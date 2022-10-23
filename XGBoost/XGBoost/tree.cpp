#include "tree.h"

namespace xgboost
{
	Tree::Tree() :
		m_SplitFeature(0),
		m_SplitValue(0.0),
		m_SplitGain(0.0),
		m_InternalValue(0.0),
		m_LeafValue(0.0),
		m_TreeLeft(nullptr),
		m_TreeRight(nullptr) {};

	double Tree::PredictLeafValue(const vector<double>& datasetOne)
	{
		if (!this->m_TreeLeft && !this->m_TreeRight)
			return this->m_LeafValue;

		else if (datasetOne[this->m_SplitFeature] <= this->m_SplitValue)
			return this->m_TreeLeft->PredictLeafValue(datasetOne);

		else
			return this->m_TreeRight->PredictLeafValue(datasetOne);
	}

	string Tree::DescribeTree()
	{
		if(!this->m_TreeLeft && !this->m_TreeRight)
			return "{\"m_LeafValue\":" + to_string(this->m_LeafValue) + "}";

		string leftInfo = this->m_TreeLeft->DescribeTree();
		string rightInfo = this->m_TreeRight->DescribeTree();

		string treeStructure;
		treeStructure = "{\"m_SplitFeature\":" + to_string(this->m_SplitFeature) + \
			",\"m_SplitValue\":" + to_string(this->m_SplitValue) + \
			",\"m_SplitGain\":" + to_string(this->m_SplitGain) + \
			",\"m_InternalValue\":" + to_string(this->m_InternalValue) + \
			",\"tree_left\":" + leftInfo + \
			",\"tree_right\":" + rightInfo + "}";

		return treeStructure;
	}
}