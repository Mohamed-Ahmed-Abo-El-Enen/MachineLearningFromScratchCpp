#include "data.h"

Data::Data(bool isTrain)
{
	m_IsTrain = isTrain;
}

void Data::PrepareData(vector<vector<double>>& features, vector<int>& labels)
{
	m_Features = features;	
	m_Target = labels;
	m_FeatureSize = features[0].size();
	m_SamplesSize = features.size();

	m_FeaturesVec.reserve(m_FeatureSize);
	for (int i = 0; i < m_FeatureSize; i++)
		m_FeaturesVec.push_back(i);

	m_SamplesVec.reserve(m_SamplesSize);
	for (int i = 0; i < m_SamplesSize; i++)	
		m_SamplesVec.push_back(i);		
}

double Data::ReadFeature(int sampleIndex, int featureIndex)
{
	return m_Features[sampleIndex][featureIndex];
}

int Data::ReadTarget(int sampleIndex)
{
	return m_Target[sampleIndex];
}

int Data::GetSampleSize()
{
	return (int)m_Features.size();
}

int Data::GetFeatureSize()
{
	return m_FeatureSize;
}

vector<int> Data::GenerateSample(int& num)
{
	if (num == -1) 
		return m_SamplesVec;
	else 
	{
		random_shuffle(m_SamplesVec.begin(), m_SamplesVec.end());
		return vector<int>(m_SamplesVec.begin(), m_SamplesVec.begin() + num);
	}
}

vector<int> Data::GenerateFeatures(function<int(int)>& func)
{
	int m = func(GetFeatureSize());
	random_shuffle(m_FeaturesVec.begin(), m_FeaturesVec.end());
	return vector<int>(m_FeaturesVec.begin(), m_FeaturesVec.begin() + m);
}

void Data::SortByFeature(vector<int>& samplesVec, int featureIndex)
{
	sort(samplesVec.begin(), samplesVec.end(), [&](int a, int b) 
		{
			return ReadFeature(a, featureIndex) < ReadFeature(b, featureIndex);
		});
}