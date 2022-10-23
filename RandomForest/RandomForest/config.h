#ifndef CONFIG_H
#define CONFIG_H
#pragma once

#include <string>

using namespace std;

namespace random_forest
{
	class Config
	{
	public:
		//NumEstimators : Number of boosted trees to fit.
		int m_NumEstimatiors = 10;

		//MaxDepth :Maximum tree depth for base learners, -1 means no limit.
		int m_MaxDepth = -1;

		//MinSamplesSplit : The minimum number of samples required to split an internal node.
		int m_MinSamplesSplit = 150;

		//MinSamplesLeaf :The minimum number of samples required to be at a leaf node.
		int m_MinSamplesLeaf = 1;

		//EachTreeSamplesNum : samples in each tree.
		int m_EachTreeSamplesNum = -1;

		//n_jobs: number of threads
		int m_n_Jobs = 8;		

		//Criterion: Criterion function
		string m_Criterion = "gini";

		// MaxFeatures: MaxFeatures number selection function
		string m_MaxFeatures = "log2";

	};
}
#endif // !CONFIG_H