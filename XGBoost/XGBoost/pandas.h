#ifndef PANDAS_H
#define PANDAS_H
#pragma once

#include <vector>
#include <string>
#include <fstream>
#include <sstream>

using namespace std;

namespace pandas
{
	struct Dataset
	{
		vector<vector<double>> features;
		vector<int> labels;
	};

	Dataset ReadCSV(string filePath, char sep, double fillNA, int numRows = INT_MAX);
	void SaveCSV(const vector<double>& datasetVect, const string filePath);
}

#endif // !PANDAS_H

