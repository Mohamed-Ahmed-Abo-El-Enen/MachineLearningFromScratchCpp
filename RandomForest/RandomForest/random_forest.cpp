#include "random_forest.h"

RandomForest::RandomForest(int nEstimators, string criterion, string maxFeatures, int maxDepth, int minSamplesSplit, int minSamplesLeaf, int eachTreeSamplesNum, int nJobs):
	m_NumEstimatiors(nEstimators),
	m_Criterion(criterion),
	m_MaxFeatures(maxFeatures),
	m_MaxDepth(maxDepth),
	m_MinSamplesSplit(minSamplesSplit),
	m_MinSamplesLeaf(minSamplesLeaf),
	m_EachTreeSamplesNum(eachTreeSamplesNum),
	m_n_Jobs(nJobs) 
{
	m_DecisionTrees.reserve(nEstimators);
}

RandomForest::RandomForest(Config config):
	m_NumEstimatiors(config.m_NumEstimatiors),
	m_Criterion(config.m_Criterion),
	m_MaxFeatures(config.m_MaxFeatures),
	m_MaxDepth(config.m_MaxDepth),
	m_MinSamplesSplit(config.m_MinSamplesSplit),
	m_MinSamplesLeaf(config.m_MinSamplesLeaf),
	m_EachTreeSamplesNum(config.m_EachTreeSamplesNum),
	m_n_Jobs(config.m_n_Jobs)
{
	m_DecisionTrees.reserve(config.m_NumEstimatiors);
}

void RandomForest::fit(Data& trainData)
{
	ThreadPool pool(m_n_Jobs);
	std::vector<std::future<DecisionTree>> results;
	for (int i = 0; i < m_NumEstimatiors; i++) 
	{
		results.emplace_back(pool.enqueue([&, i] 
			{
				DecisionTree tree(m_Criterion, m_MaxDepth, m_MinSamplesSplit, m_MinSamplesLeaf, m_EachTreeSamplesNum, m_MaxFeatures);
				tree.fit(trainData);
				cout << "Fitted Tree: " << i << endl;
				return tree;
			}));
	}

	for (auto&& each : results) 
		m_DecisionTrees.push_back(each.get());
}

void RandomForest::norm(vector<double>& total)
{
	for (double& i : total)
		i /= m_NumEstimatiors;
}

void VecAdd(vector<double>& total, vector<double>& part)
{
	for (int i = 0; i < total.size(); i++)
		total[i] += part[i];
}

vector<int> RandomForest::PredictClass(const vector<double>& probability)
{
	vector<int> yHat;
	for (size_t i = 0; i < probability.size(); i++)
	{
		if (probability[i] > 0.5)
			yHat.push_back(1);
		else
			yHat.push_back(0);
	}
	return yHat;
}

vector<double> RandomForest::PredictProbability(Data& data)
{
	ThreadPool pool(m_n_Jobs);
	vector<future<vector<double>>> poolResults;
	for (int i = 0; i < m_NumEstimatiors; i++) 
	{
		poolResults.emplace_back(pool.enqueue([&, i] 
			{
				vector<double> results(data.GetSampleSize(), 0);
				m_DecisionTrees[i].PredictProbability(data, results);
				cout << "Predict Tree: " << i << endl;
				return results;
			}));
	}
	vector<double> results(data.GetSampleSize(), 0);
	for (auto&& each : poolResults) 
	{
		auto tmpResults = each.get();
		VecAdd(results, tmpResults);
	}
	norm(results);
	return results;
}