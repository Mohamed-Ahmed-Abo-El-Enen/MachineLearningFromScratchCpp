#pragma once

#include "Node.h"

using namespace std;

struct TwoDataFrame
{
	DataFrame pres;
	DataFrame abs;

	TwoDataFrame(DataFrame pres, DataFrame abs)
	{
		this->pres = pres;
		this->abs = abs;
	}
};


class DecisionTree
{
	Node* pRoot;

public:
	DecisionTree(DataFrame);
	~DecisionTree();

	void traverseTree()const;
	static TwoDataFrame splitData(const DataFrame& dataBefore, int feature, int Category);
	static void constructTree(Node* pNode);
	vector<int> predict(DataFrame);
	void deleteChildren(Node* pNode);
	static int recursivelyPredict(Node* pNode, vector<int> observation);

};
