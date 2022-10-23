#ifndef TREE_H
#define TREE_H
#pragma once

#include <vector>
#include <string>

using namespace std;

namespace xgboost
{
	class Tree
	{
	public: 
		int m_SplitFeature;
		double m_SplitValue;
		double m_SplitGain;
		double m_InternalValue;
		double m_LeafValue;
		Tree* m_TreeLeft;
		Tree* m_TreeRight;

		Tree();
		~Tree() = default;
		double PredictLeafValue(const vector<double>& dataset);	
		string DescribeTree();

	};
}

#endif // !TREE_H