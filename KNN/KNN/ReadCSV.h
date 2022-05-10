#pragma once
#include <ctime>
#include <iostream>
#include <string>
#include <vector>
#include <fstream>


using namespace std;

struct CSVSample
{
	vector<string> values;

	CSVSample(vector<string> _values) :
		values(_values){}
};

vector<string> lineSpliter(string line, string delimiter)
{
	vector<string> values;
	size_t pos = 0;
	std::string token;
	while ((pos = line.find(delimiter)) != std::string::npos)
	{
		token = line.substr(0, pos);
		values.push_back(token);
		line.erase(0, pos + delimiter.length());
	}
	values.push_back(line);
	return values;
}

vector<double> slicing(vector<string> arr, int X, int Y)
{
	vector<double> res;
	for (size_t i = X; i < Y; i++)
		res.push_back(stod(arr[i]));
	return res;
}

vector<CSVSample> readcsv(string csv_file_path)
{
	vector<CSVSample> Samples;
	string line;
	ifstream file(csv_file_path);
	getline(file, line);
	string delimiter = ",";
	while (getline(file, line))
	{
		CSVSample sam(lineSpliter(line, delimiter));
		Samples.push_back(sam);
	}
	return Samples;
}
