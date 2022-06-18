#pragma once
#ifndef UTILS_HPP
#define UTILS_HPP

#include <string>
#include <vector>
#include <algorithm>
#include "CSVReader.h"

using namespace std;

void MinMaxScaler(vector<vector<double>>& X, vector<double>& minVec, vector<double>& maxVec)
{
	for (size_t i = 0; i < X[0].size(); i++)
	{
		vector<double> feature_value;
		for (size_t j = 0; j < X.size(); j++)		
			feature_value.push_back(X[j][i]);
		
		minVec.push_back(*min_element(feature_value.begin(), feature_value.end()));
		maxVec.push_back(*max_element(feature_value.begin(), feature_value.end()));

		for (size_t j = 0; j < X.size(); j++)		
			X[j][i] = (X[j][i] - minVec[i]) / (maxVec[i] - minVec[i]);		
	}
}

vector<int> ConvertY2SVMY(vector<int> y, int y1Index=1)
{
	for (size_t i = 0; i < y.size(); i++)
		if (y[i] != y1Index)
			y[i] = -1;
	return y;
}

vector<vector<double>> SplitWithClassIndex(const vector<vector<double>> X, const vector<int> y, const int classIndex)
{
	vector<vector<double>> classData;
	for (size_t i = 0; i < y.size(); i++)
		if (y[i] == classIndex)
			classData.push_back(X[i]);
	
	return classData;
}

void CombineDataWithClassIndex(const vector<vector<double>> X1, const vector<int> y1, const vector<vector<double>> X2, const vector<int> y2, vector<vector<double>>& X, vector<int>& y)
{
	for (size_t i = 0; i < y1.size(); i++)
	{
		X.push_back(X1[i]);
		y.push_back(y1[i]);
	}

	for (size_t i = 0; i < y2.size(); i++)
	{
		X.push_back(X2[i]);
		y.push_back(y2[i]);
	}
}

#endif