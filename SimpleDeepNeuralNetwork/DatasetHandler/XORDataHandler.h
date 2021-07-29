#pragma once
#include<iostream>
#include<string>
#include<vector>
#include <fstream>
#include <sstream> 

using namespace std;

class XORDataHandler
{
private:
	ifstream m_XORDataHandlerFile;

public:
	XORDataHandler(const string filename);
	unsigned GetNextInputs(vector<double>& inputVals);
	unsigned GetTargetOutputs(vector<double>& targetOutputVals);
	bool isEof();
	void GenerateXORData();
	void ReadXORData(vector<vector<double>>& datasetFeatures, vector<vector<double>>& datasetTargets);
};
