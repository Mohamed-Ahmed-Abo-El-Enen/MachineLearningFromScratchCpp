#include "utils.h"

double CalcualteAUC(vector<int>& yTrue, vector<double>& yHat)
{
	int numBins = yTrue.size() <= 10000 ? 1000 : 100;

	int countPositive = accumulate(yTrue.begin(), yTrue.end(), 0);
	int countNegative = yTrue.size() - countPositive;
	int totalCase = countPositive * countNegative;

	vector<int> posHistogram(numBins, 0);
	vector<int> negHistogram(numBins, 0);

	double binWidth = 1.0 / numBins;
	for (size_t i = 0; i < yTrue.size(); i++)
	{
		int ix = int(yHat[i] / binWidth);
		if (yTrue[i] == 1)
			posHistogram[ix] += 1;
		else
			negHistogram[ix] += 1;
	}

	int accumulatedNeg = 0;
	double satisfiedPair = 0;
	for (size_t i = 0; i < numBins; i++)
	{
		satisfiedPair += (posHistogram[i] * accumulatedNeg + posHistogram[i] * negHistogram[i] * 0.5);
		accumulatedNeg += negHistogram[i];
	}

	return satisfiedPair / double(totalCase);
}

double CalcualteKS(vector<int>& yTrue, vector<double>& yHat)
{
	vector<double> uniqueValues = yHat;
	sort(uniqueValues.begin(), uniqueValues.end());
	uniqueValues.erase(unique(uniqueValues.begin(), uniqueValues.end()), uniqueValues.end());

	double tpr, fpr, distance;
	double maxDistance = 0;
	for (double curPoint : uniqueValues)
	{
		double tp = 0.0001;
		double fp = 0.0001;
		double tn = 0.0001;
		double fn = 0.0001;

		for (size_t i = 0; i < yTrue.size(); i++)
		{
			if ((yHat[i] >= curPoint) && (yTrue[i] == 1))
				tp += 1;
			else if ((yHat[i] >= curPoint) && (yTrue[i] != 1))
				fp += 1;
			else if ((yHat[i] < curPoint) && (yTrue[i] == 1))
				fn += 1;
			else
				tn += 1;
		}

		tpr = tp / (tp + fn);
		fpr = fp / (fp + tn);
		distance = tpr - fpr;
		if (distance > maxDistance)
			maxDistance = distance;
	}
	return maxDistance;
}

double CalcualteACC(vector<int>& yTrue, vector<double>& yHat)
{
	int countRight = 0;
	for (size_t i = 0; i < yTrue.size(); i++)	
		if ((yTrue[i] == 0 && yHat[i] < 0.5) || (yTrue[i] == 1 && yHat[i] >= 0.5))		
			countRight += 1;		
	
	return (double)countRight / yTrue.size();
}