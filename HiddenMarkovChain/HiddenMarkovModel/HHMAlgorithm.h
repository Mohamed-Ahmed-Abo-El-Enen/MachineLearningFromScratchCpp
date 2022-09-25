#ifndef HHMALGORITHMS_H
#define HHMALGORITHMS_H
#pragma once

#include "HiddenMarkovModel.h"
#include "TestData.h"

using namespace std;

class HHMAlgorithm
{
private:
	double CalcualteForwardStepProbability(size_t stepNumber, size_t curState, const HiddenMarkovModel& model, const TestData& data, const vector<vector<double>>& forwardStateProbability);
	double CalcualteBackwardStepProbability(size_t stepNumber, size_t curState, const HiddenMarkovModel& model, const TestData& data, const vector<vector<double>>& backwardStateProbability);
	double CalculateNewStateProbability(size_t stepNumber, size_t prevState, size_t curState, size_t curSymbol, const HiddenMarkovModel& model, const vector<vector<double>>& sequenceProbability);
	size_t FindBestTransitionSource(size_t stepNumber, size_t curState, size_t curSymbol, const HiddenMarkovModel& model, const vector<vector<double>>& sequenceProbability);

public:
	const size_t HMM_UNDEFINED_STATE = -1;

	// Find most probable sequence of hidden states
	vector<size_t> FindMostProbableStateSequence(const HiddenMarkovModel& model, const TestData& data);

	// Calculates alpha-beta value pairs for each time moment		 
	vector<vector<pair<double, double>>> CalcualteForwardBackwardProbabilites(const HiddenMarkovModel& model, const TestData& data);
};
#endif //!HHMALGORITHMS_H