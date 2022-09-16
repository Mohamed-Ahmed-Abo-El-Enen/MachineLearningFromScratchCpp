#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <vector>
#include <map>

using namespace std;

#define PI 3.14159265358979323846

namespace Probability
{
	double Mean(const vector<double>& data);
	double Variance(const vector<double>& data);
	double CalculateNormalProbability(double x, double mean, double stdev);
}

namespace Evaluation
{
	double CalculateAccuracy(const vector<int>& yTrue, const vector<int>& yHat);
}

namespace DataReader
{

	vector<string> lineSpliter(string line, string delimiter);
	vector<double> slicing(vector<string> arr);
	vector<vector<double>> ReadIrisDataset(string filePath, int yIndex);
}

namespace DataManipulation
{
	void Train_Test_Split(const vector<vector<double>>& dataset, float testSize, vector<vector<double>>& trainDataset, vector<vector<double>>& testDataset);
	vector<int> GetColumnValues(const vector<vector<double>>& dataset, int yIndex);
	vector<vector<double>> RemoveColumnDataset(vector<vector<double>>& dataset, int colIdx);
}

#endif // !UTILS_H