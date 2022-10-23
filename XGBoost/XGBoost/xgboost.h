#ifndef XGBOOST_H
#define XGBOOST_H
#pragma once

#include <numeric>
#include "decision_tree.h"
#include "tree.h"
#include "config.h"

using namespace std;

namespace xgboost
{
	struct Gradients
	{
		double grad;
		double hess;
	};

	class XGBoost
	{
	private:
		vector<double> grad;
		vector<double> hess;
		Gradients CalculateGradHess(int y, double yPred);

	public:
		vector<Tree*> m_Trees;
		const Config m_Config;
		double m_Prediction;

		XGBoost(Config config);
		~XGBoost();
		void fit(const vector<vector<double>>& features, const vector<int>& labels);
		vector<double> PredictProba(const vector<double>& features);
		string SaveModelToString();
	};
};

#endif // !XGBOOST_H