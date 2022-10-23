#ifndef UTILS_H
#define UTILS_H
#pragma once

#include <numeric>
#include "data.h"

namespace utils
{
	double ComputeTargetProbability(vector<int>& samplesVec, Data& data);
	double ComputeGini(int&, int&);
	double ComputeGiniIndex(int&, int&, int&, int&);
	int ComputeTrue(vector<int>& samplesVec, Data& data);
	int _sqrt(int num);
	int _log2(int num);
	int _none(int num);
	void WriteDataToCSV(const vector<int>& results, Data& data, const string& filePath, bool train);
	double CalcualteACC(vector<int>& yTrue, vector<int>& yHat);
}

#endif // !UTILS_H