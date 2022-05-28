#include "DecisionTree.h"

DecisionTree::DecisionTree(DataFrame trainingData)
{
	Node* pRoot = new Node(trainingData);
	constructTree(pRoot);
	this->pRoot = pRoot;
}

DecisionTree::~DecisionTree()
{
	DecisionTree::deleteChildren(this->pRoot);
}

TwoDataFrame DecisionTree::splitData(const DataFrame& dataBefore, int feature, int category)
{
	vector<int> presSplit;
	vector<int> absSplit;

	int idx = 0;
	for (int i:dataBefore[feature])
	{
		int val = dataBefore[feature][idx];
		if (val == category)
			presSplit.push_back(idx);
		else
			absSplit.push_back(idx);

		idx++;
	}

	int numRows = dataBefore.size();
	DataFrame presData;
	vector<int> rowBuffer;
	for (int row = 0; row < numRows; row++)
	{
		for (int col: presSplit)
			rowBuffer.push_back(dataBefore[row][col]);

		presData.push_back(rowBuffer);
		rowBuffer.clear();
	}

	DataFrame absData;
	for (int row = 0; row < numRows; row++)
	{
		for (int col : absSplit)
			rowBuffer.push_back(dataBefore[row][col]);

		absData.push_back(rowBuffer);
		rowBuffer.clear();
	}

	TwoDataFrame returnObj(presData, absData);

	return returnObj;
}

void DecisionTree::constructTree(Node* pNode)
{
	const vector<int> outputsBefore = (pNode->trainingData)[(pNode->trainingData).size() - 1];
	float giniBefore = Node::getGiniImpurity(outputsBefore);

	if (pNode->bestSplit.resultGini < giniBefore)
	{
		TwoDataFrame childrenData = DecisionTree::splitData(pNode->trainingData, pNode->bestSplit.feature, pNode->bestSplit.category);

		if (childrenData.pres[0].size() > 0)
		{
			Node* newRightChild = new Node(childrenData.pres);
			pNode->pRightChild = newRightChild;
			DecisionTree::constructTree(pNode->pRightChild);
		}

		if (childrenData.abs[0].size() > 0)
		{
			Node* newLeftChild = new Node(childrenData.abs);
			pNode->pLeftChild = newLeftChild;
			DecisionTree::constructTree(pNode->pLeftChild);
		}
	}
}

void DecisionTree::deleteChildren(Node* pNode)
{
	if (pNode->pRightChild != NULL)
		DecisionTree::deleteChildren(pNode->pRightChild);

	if (pNode->pLeftChild != NULL)
		DecisionTree::deleteChildren(pNode->pLeftChild);

	delete pNode;
}

int DecisionTree::recursivelyPredict(Node* pNode, vector<int> observations)
{
	int obsCat = observations[pNode->bestSplit.feature];
	int predictClass = -1;

	if ((pNode->pLeftChild == NULL) && pNode->pRightChild == NULL)
		predictClass = (pNode->trainingData)[(pNode->trainingData).size() - 1][0];

	else if (obsCat == pNode->bestSplit.category)
		predictClass = DecisionTree::recursivelyPredict(pNode->pRightChild, observations);

	else
		predictClass = DecisionTree::recursivelyPredict(pNode->pLeftChild, observations);

	return predictClass;
}

vector<int> DecisionTree::predict(DataFrame testData)
{
	vector<int> predictions;

	int numObs = testData[0].size();
	int numFeatures = testData.size();

	for (int obs = 0; obs < numObs; obs++)
	{
		vector<int> observations;
		for (int feature = 0; feature < numFeatures; feature++)
			observations.push_back(testData[feature][obs]);

		predictions.push_back(DecisionTree::recursivelyPredict(this->pRoot, observations));
	}
	return predictions;
}

