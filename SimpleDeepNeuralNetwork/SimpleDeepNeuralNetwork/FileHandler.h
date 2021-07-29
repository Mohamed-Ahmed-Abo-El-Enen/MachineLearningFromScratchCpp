#pragma once
#include<iostream>
#include<string>
#include<vector>
#include <fstream>
#include <sstream> 
#include<time.h>

using namespace std;

class FileHandler
{
private:
	ifstream m_FileHandler;
	unsigned GetSampleFeatures(vector<double>& sampleFeature);
	unsigned GetSampleLabel(vector<double>& sampleLabel);
	bool isEof();

public:
	FileHandler();
	void ReadDatasetFile(string filePath, vector<vector<double>>& datasetFeatures,  vector<vector<double>>& datasetTargets);
};
