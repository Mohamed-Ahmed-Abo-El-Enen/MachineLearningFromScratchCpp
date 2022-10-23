#include "xgboost.h"

namespace xgboost
{
	XGBoost::XGBoost(Config config) : m_Config(config) {};
	
	XGBoost::~XGBoost() {};

	void XGBoost::fit(const vector<vector<double>>& features, const vector<int>& labels)
	{
		double mean = accumulate(labels.begin(), labels.end(), 0) / (double)labels.size();
		m_Prediction = 0.5 * log((1 + mean) / (1 - mean));

		Gradients gradients;
		for (size_t i = 0; i < labels.size(); i++)
		{
			gradients = CalculateGradHess(labels[i], m_Prediction);
			grad.push_back(gradients.grad);
			hess.push_back(gradients.hess);
		}

		for (int stage = 1; stage <= m_Config.m_NumEstimators; stage++)
		{
			cout << "=============================== iter: " << stage << " ===============================" << endl;
			Tree* stageTree;
			BaseDecisionTree baseDecisionTree = BaseDecisionTree(m_Config);
			stageTree = baseDecisionTree.fit(features, labels, grad, hess);
			m_Trees.push_back(stageTree);
			cout << stageTree->DescribeTree() << endl;

			for (size_t i = 0; i < labels.size(); i++)
			{
				double yHat = stageTree->PredictLeafValue(features[i]);
				gradients = CalculateGradHess(labels[i], yHat);
				grad[i] += m_Config.m_LearningRate * gradients.grad;
				hess[i] += m_Config.m_LearningRate * gradients.hess;
			}
		}				
	}

	Gradients XGBoost::CalculateGradHess(int y, double yHat)
	{
		
		Gradients gradients;
		double pred = 1.0 / (1.0 + exp(-yHat));

		double grad = (-y + (1 - y) * exp(pred)) / (1.0 + exp(pred));
		double hess = exp(pred) / pow((1 + exp(pred)), 2);

		gradients = { grad, hess };
		return gradients;
	}
	
	vector<double> XGBoost::PredictProba(const vector<double>& features)
	{
		double pred = m_Prediction;
		vector<double> res;
		double p;
		for (Tree *tree : m_Trees)		
			pred += m_Config.m_LearningRate * tree->PredictLeafValue(features);
		
		p = 1.0 / (1 + exp(2 * pred));

		res.push_back(p);
		res.push_back(1 - p);
		return res;
	}

	string XGBoost::SaveModelToString()
	{
		string s;

		//Trees
		s += "{\"Trees\":[";
		for (size_t i = 0; i < m_Trees.size(); i++)
		{
			Tree* tree = m_Trees[i];
			s += m_Trees[i]->DescribeTree();
			s += ",";
		}
		s = s.substr(0, s.length() - 1) + "]";

		// Param
		s += ",\"Param\":{";
		s += "\"m_NumEstimators\":" + to_string(m_Config.m_NumEstimators) + ",";
		s += "\"m_MaxDepth\":" + to_string(m_Config.m_MaxDepth) + ",";
		s += "\"m_LearningRate\":" + to_string(m_Config.m_LearningRate) + ",";
		s += "\"m_MinSamplesSplit\":" + to_string(m_Config.m_MinSamplesSplit) + ",";
		s += "\"m_MinDataInLeaf\":" + to_string(m_Config.m_MinDataInLeaf) + ",";
		s += "\"m_MinChildWeight\":" + to_string(m_Config.m_MinChildWeight) + ",";
		s += "\"m_ColSampleByTree\":" + to_string(m_Config.m_ColSampleByTree) + ",";
		s += "\"m_RegGamma\":" + to_string(m_Config.m_RegGamma) + ",";
		s += "\"m_RegLambda\":" + to_string(m_Config.m_RegLambda) + ",";
		s += "\"m_MaxBin\":" + to_string(m_Config.m_MaxBin);
		s += "}";

		return s + "}";
	}	
}