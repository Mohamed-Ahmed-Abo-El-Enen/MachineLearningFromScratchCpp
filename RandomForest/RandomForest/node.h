#ifndef NODE_H
#define NODE_H
#pragma once

#include <iostream>

using namespace std;

struct Node
{
	int m_FeatureIndex;
	shared_ptr<Node> m_LeftTree;
	shared_ptr<Node> m_RightTree;
	double m_Threshold;
	bool m_IsLeaf;
	int m_Depth;
	double m_Probability;

	Node()
	{
		m_LeftTree = nullptr;
		m_RightTree = nullptr;
		m_IsLeaf = false;
		m_Probability = 0;
		m_Depth = 1;
		m_Threshold = 0;
		m_FeatureIndex = 0;
	}
};
#endif // !NODE_H
