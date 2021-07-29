#pragma once
#include<iostream>
#include<fstream>
#include<algorithm>
#include<vector>
#include<cassert>
#include<time.h>

using namespace std;

class MNIST
{
private:
	unsigned char* LoadMnistData(const string filename, int& row, int& col, int& num);
	unsigned char* LoadMnistLabel(const string filename, int& num);
public:
	MNIST();
	void GetMNISTDataset(string imagesFilePath, string lableFilePath, vector<vector<double>>& samplesFeature, vector<vector<double>>& samplesLabel);
	void GenerateMNISTDataFormat(vector<vector<double>>& samplesFeature, vector<vector<double>>& samplesLabel);
};
