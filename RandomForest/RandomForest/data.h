#ifndef DATA_H
#define DATA_H
#pragma once

#include <iostream>
#include <vector>
#include <functional>
#include <string>
#include <sstream>
#include <fstream>
#include <iterator>
#include <algorithm>
#include <iostream>

using namespace std;

class Data
{
private:
	vector<vector<double>> m_Features;
	vector<int> m_Target;
	int m_FeatureSize = 0;
	int m_SamplesSize = 0;
	bool m_IsTrain;
	vector<int> m_FeaturesVec;
	vector<int> m_SamplesVec;

public:
	Data(bool isTrain = true);
	void PrepareData(vector<vector<double>>& features, vector<int>& labels);
	double ReadFeature(int sampleIndex, int featureIndex);
	int ReadTarget(int sampleIndex);
	int GetSampleSize();
	int GetFeatureSize();
	vector<int> GenerateSample(int& num);
	vector<int> GenerateFeatures(function<int(int)>& func);
	void SortByFeature(vector<int>& samplesVec, int featureIndex);
};

#endif // !DATA_H

