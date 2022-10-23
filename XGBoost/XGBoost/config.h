#ifndef CONFIG_H
#define CONFIG_H
#pragma once

namespace xgboost
{
	class Config
	{
	public:
		//NumEstimators : Number of boosted trees to fit.
		int m_NumEstimators = 5;

		//MaxDepth :Maximum tree depth for base learners, -1 means no limit.
		int m_MaxDepth = 6;

		//LearningRate : Boosting learning rate.
		double m_LearningRate = 0.1;

		//MinSamplesSplit : The minimum number of samples required to split an internal node.
		int m_MinSamplesSplit = 2;

		//MinDataInLeaf :The minimum number of samples required to be at a leaf node.
		int m_MinDataInLeaf = 1;

		//MinChildWeight : Minimum sum of instance weight(hessian) needed in a child(leaf).
		double m_MinChildWeight = 1e-3;

		//ColSampleByTree : Subsample ratio of columns when constructing each tree.
		double m_ColSampleByTree = 1.0;

		//RegGamma : L1 regularization term on weights.
		double m_RegGamma = 0.0;

		//RegLambda : L2 regularization term on weights.
		double m_RegLambda = 0.0;

		//MaxBin: Max number of discrete bins for features.
		int m_MaxBin = 100;
	};
}
#endif // !CONFIG_H