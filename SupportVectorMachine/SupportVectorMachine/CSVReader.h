#pragma once
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include<map>

using namespace std;

class CSVSample
{
public:
	vector<string> values;

	CSVSample(vector<string> _values);
	static vector<string> lineSpliter(string line, string delimiter);
	static vector<double> slicing(vector<string> arr, int X, int Y);
	static vector<CSVSample> readCSV(string csv_file_path);
	static void removeCol(vector<CSVSample>& Samples, int colIdx);
};

class Sample
{
private:
public:
	vector<double> features;
	int label;

	Sample();
	Sample(vector<double> _features, int _label);

	static vector<Sample> ConvertCSVSamples(vector<CSVSample> csvArr, vector<int>  categorical_cols, int yIndex, map<string, int>& class_map);
	static void ConvertCSVSamplesXY(vector<CSVSample> csvArr, vector<int>  categorical_cols, int yIndex, map<string, int>& class_map, vector<vector<double>>& X, vector<int>& y);
};

