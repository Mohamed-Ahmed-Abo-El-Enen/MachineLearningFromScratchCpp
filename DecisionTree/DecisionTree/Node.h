#pragma once

#include <string>
#include<vector>

using namespace std;

using DataFrame = vector<vector<int>>;

struct BestSplitReturn
{
	float resultGini;
	int feature;
	int category;
};



class Node
{
public:
	Node* pLeftChild;
	Node* pRightChild;
	DataFrame trainingData;
	BestSplitReturn bestSplit;

	Node(const DataFrame& data);
	~Node();

	static float getGiniImpurity(const vector<int>& outcomes);
	static DataFrame getSplitTargets(const DataFrame& pData, int feature, int category);
	static BestSplitReturn getBestSplit(const DataFrame& pData);

};
