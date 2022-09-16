#include "NaiveBayes.h"

vector<vector<double>> NaiveBayes::SplitByClass(vector<vector<double>>& dataset, int& classLabel, int yIndex)
{
	vector<vector<double>> result;
	for (size_t i = 0; i < dataset.size(); i++)
	{
		vector<double> sample = dataset[i];

		if (sample[yIndex] != classLabel)
			continue;

		vector<double> temp;
		for (size_t j = 0; j < sample.size(); j++)
		{
			if (j == yIndex)
				continue;

			temp.push_back(dataset[i][j]);			
		}
		result.push_back(temp);
	}
	return result;
}

ClassSummary NaiveBayes::CalcualteClassSummary(vector<vector<double>>& dataset, int& classLabel, int yIndex)
{
	auto classData = SplitByClass(dataset, classLabel, yIndex);
	ClassSummary summary;
	vector<double> temp;
	vector<double> feature;

	for (int i = 0; i < classData[0].size(); i++)
	{
		temp.clear();
		feature.clear();

		for (size_t j = 0; j < classData.size(); j++)
			feature.push_back(classData[j][i]);
		
		temp.push_back(Probability::Mean(feature));
		temp.push_back(Probability::Variance(feature));
		summary.m_MeanVarianceDistribution.push_back(temp);
	}
		
	summary.m_ClassProbability = double(classData.size()) / dataset.size();
	return summary;
}

void NaiveBayes::fit(vector<vector<double>>& trainDataset, int yIndex)
{
	m_UniqueLabel = DataManipulation::GetColumnValues(trainDataset, yIndex);
	sort(m_UniqueLabel.begin(), m_UniqueLabel.end());
	m_UniqueLabel.erase(unique(m_UniqueLabel.begin(), m_UniqueLabel.end()), m_UniqueLabel.end());
	for (auto row = m_UniqueLabel.begin(); row != m_UniqueLabel.end(); row++)
		m_Summary.push_back(CalcualteClassSummary(trainDataset, *row, yIndex));
}

double NaiveBayes::ProbabilityBySummary(vector<double>& testDataset, ClassSummary& summary)
{
	int index = 0;
	double prob = 1;
	for (auto row = summary.m_MeanVarianceDistribution.begin(); row != summary.m_MeanVarianceDistribution.end() - 1; row++)
	{
		prob *= Probability::CalculateNormalProbability(testDataset[index], (*row)[0], (*row)[1]);
		index++;
	}
	prob *= summary.m_ClassProbability;
	return prob;
}

int NaiveBayes::predict(vector<double>& testDataset)
{
	vector<double> out;
	for (size_t i = 0; i < m_UniqueLabel.size(); i++)	
		out.push_back(ProbabilityBySummary(testDataset, m_Summary[i]));
	
	int maxClassIndex = max_element(out.begin(), out.end()) - out.begin();
	return maxClassIndex;
}

void NaiveBayes::PrintClassesDistribution()
{
	cout << "\n";
	for (size_t i = 0; i < m_Summary.size(); i++)
	{
		cout << "Class # " << i << " Class Probability = "<< m_Summary[i].m_ClassProbability<<endl;
		cout << "***********************************************" << endl;
		for (size_t j = 0; j < m_Summary[i].m_MeanVarianceDistribution.size(); j++)		
			cout << "Feature #" << j << " Mean = " << m_Summary[i].m_MeanVarianceDistribution[j][0] << " Variance = " << m_Summary[i].m_MeanVarianceDistribution[j][1] << endl;		

		cout << "\n\n";
	}
}
