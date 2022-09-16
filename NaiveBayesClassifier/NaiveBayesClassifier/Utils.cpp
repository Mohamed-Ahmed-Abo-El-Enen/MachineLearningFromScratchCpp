#include "Utils.h"

namespace Probability
{
	double Mean(const vector<double>& data)
	{
		double mean = accumulate(begin(data), end(data), 0.0) / data.size();
		return mean;
	}

	double Variance(const vector<double>& data)
	{
		double mean = Mean(data);
		double accum = 0.0;
		for_each(begin(data), end(data), [&](const double d) 
			{
			accum += ((d - mean) * (d - mean));
			});

		double stdev = accum / data.size();
		return stdev;
	}

	double CalculateNormalProbability(double x, double mean, double stdev)
	{
		double inv_2_Sqrt_Pi_Stdev = 1 / sqrt(stdev * 2 * PI);
		double a = - ((x - mean) * (x - mean));
		a /= (2 * stdev);
		return inv_2_Sqrt_Pi_Stdev * exp(a);
	}
}

namespace Evaluation
{
	double CalculateAccuracy(const vector<int>& yTrue, const vector<int>& yHat)
	{
		double result = 0;
		int index = 0;
		for_each(yTrue.begin(), yTrue.end(), [&](double y)
			{
				if (y == yHat[index])
					result++;
				index++;
			});
		return double(100 * result / yTrue.size());
	}
}

namespace DataReader
{
	vector<string> lineSpliter(string line, string delimiter)
	{
		vector<string> values;
		size_t pos = 0;
		string token;
		while ((pos = line.find(delimiter)) && pos != string::npos)
		{
			token = line.substr(0, pos);
			values.push_back(token);
			line.erase(0, pos + delimiter.length());
		}
		values.push_back(line);
		return values;
	}

	vector<double> slicing(vector<string> arr)
	{
		vector<double> res;
		for (size_t i = 0; i < arr.size(); i++)
			res.push_back(stod(arr[i]));
		return res;
	}

	vector<vector<double>> ReadIrisDataset(string filePath, int yIndex)
	{
		map<string, int> classIndex;
		classIndex.insert(make_pair("Iris-setosa", 0));
		classIndex.insert(make_pair("Iris-versicolor", 1));
		classIndex.insert(make_pair("Iris-virginica", 2));

		ifstream file(filePath);
		string line;
		vector<vector<double>> dataset;
		int count = 0;
		if (file.is_open())
		{
			while(getline(file, line))
			{
				vector<string> vec = lineSpliter(line, ",");
				if (classIndex.find(vec[yIndex]) != classIndex.end())
					vec[yIndex] = to_string(classIndex[vec[yIndex]]);
				
				else
					vec[yIndex] = "0";

				dataset.push_back(slicing(vec));
			}
		}
		return dataset;
	}
}

namespace DataManipulation
{
	void Train_Test_Split(const vector<vector<double>>& dataset, float testSize, vector<vector<double>>& trainDataset, vector<vector<double>>& testDataset)
	{
		int size = dataset.size() * (1 - testSize);
		for (auto temp = dataset.begin(); temp != dataset.begin() + size; temp++)
			trainDataset.push_back(*temp);

		for (auto temp = dataset.begin() + size; temp != dataset.end(); temp++)
			testDataset.push_back(*temp);
	}

	vector<int> GetColumnValues(const vector<vector<double>>& dataset, int yIndex)
	{
		vector<int> y;
		for (size_t i = 0; i < dataset.size(); i++)
			y.push_back(dataset[i][yIndex]);

		return y;
	}

	vector<vector<double>> RemoveColumnDataset(vector<vector<double>>& dataset, int colIdx)
	{
		vector<vector<double>> result;
		for (int i = 0; i < dataset.size(); i++)
		{
			vector<double> temp;
			for (size_t j = 0; j < dataset[0].size(); j++)
			{
				if (colIdx == j)
					continue;
				temp.push_back(dataset[i][j]);
			}
			result.push_back(temp);
		}
		return result;
	}
}
