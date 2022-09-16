#ifndef NAIVEBAYES_H
#define NAIVEBAYES_H

#include "Utils.h"

using namespace std;

struct ClassSummary
{
	vector<vector<double>> m_MeanVarianceDistribution;
	double m_ClassProbability;
};

class NaiveBayes
{
private:
	vector<ClassSummary> m_Summary;
	vector<int> m_UniqueLabel;

	double ProbabilityBySummary(vector<double>& testDataset, ClassSummary& summary);
	ClassSummary CalcualteClassSummary(vector<vector<double>>& dataset, int& classLabel, int yIndex);
	vector<vector<double>> SplitByClass(vector<vector<double>>& dataset, int& classLabel, int yIndex);

public:
	void fit(vector<vector<double>>& trainDataset, int yIndex);
	int predict(vector<double>& testDataset);
	void PrintClassesDistribution();
};

#endif // !NAIVEBAYES_H

