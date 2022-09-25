#ifndef ESTIMATOR_H
#define ESTIMATOR_H
#pragma once

#include "HiddenMarkovModel.h"
#include "TestData.h"
#include "EvaluationMetrics.h"
#include <numeric>

class Estimator
{
public:
	// Use forward-backward probabilities to get the most probable state at each step
	vector<size_t> GetMostProbableStates(const vector<vector<pair<double, double>>>& forwardBackwardProb);

	// Confusion matrix element[i][j] is the number of elements with the		
	vector<vector<size_t>> CombineConfusionMatrix(const TestData& realData, const vector<size_t>& predictedStates, const HiddenMarkovModel& model);	

	// Use confusion matrix to calculate estimations of the prediction results		
	vector<EvaluationMetrics> GetStatePredicionEstimations(const vector<vector<size_t>>& confusionMatrix);

	void PrintPredictionsEstimations(size_t stateInd, const EvaluationMetrics& estimation, const HiddenMarkovModel& model);

};
#endif //!ESTIMATOR_H