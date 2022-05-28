#include "CSVReader.h"

CSVSample::CSVSample(vector<string> _values)
{
	values = _values;
}

vector<string> CSVSample::lineSpliter(string line, string delimiter)
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

vector<double> CSVSample::slicing(vector<string> arr, int X, int Y)
{
	vector<double> res;
	for (size_t i = X; i < Y; i++)
		res.push_back(stod(arr[i]));
	return res;
}

vector<CSVSample> CSVSample::readCSV(string csv_file_path)
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

void CSVSample::removeCol(vector<CSVSample>& Samples, int colIdx)
{
	for (int i = 0; i < Samples.size(); i++)
		Samples[i].values.erase(Samples[i].values.begin() + colIdx);
}


Sample::Sample()
{
	label = -1;
}

Sample::Sample(vector<double> _features, int _label)
{
	features = _features;
	label = _label;
}

vector<Sample> Sample::ConvertCSVSamples(vector<CSVSample> csvArr, vector<int> categorical_cols, int yIndex, map<string, int>& class_map)
{
	vector<Sample> points;
	vector< map<string, double>> feature_map(categorical_cols.size(), map<string, double>());

	for (size_t i = 0; i < csvArr.size(); i++)
	{
		Sample pt;
		int cat_col_index = 0;

		for (int c = 1; c < csvArr[0].values.size(); c++)
		{
			if (find(categorical_cols.begin(), categorical_cols.end(), c) == categorical_cols.end() || categorical_cols.size() <= 0)
				pt.features.push_back(stod(csvArr[i].values[c]));
			else
			{
				feature_map[cat_col_index].insert({ csvArr[i].values[c], feature_map[cat_col_index].size() });
				pt.features.push_back(feature_map[cat_col_index].at(csvArr[i].values[c]));
				cat_col_index++;
			}
		}	
		pt.label = -1;
		points.push_back(pt);
	}

	for (int i = 0; i < csvArr.size(); i++)
		class_map.insert({ csvArr[i].values[yIndex], class_map.size() });

	for (int i = 0; i < csvArr.size(); i++)
		points[i].label = class_map.at(csvArr[i].values[yIndex]);

	return points;
}

vector<vector<int>> Sample::getConcatedDataframe(vector<Sample> points)
{
	vector<vector<int>> features;

	for (int i = 0; i < points[0].features.size(); i++)
	{
		vector<int> sample_fet;
		for (int j = 0; j < points.size(); j++)
		{
			sample_fet.push_back((int)points[j].features[i]);
		}
		features.push_back(sample_fet);
	}

	vector<int> labels;
	for (int i = 0; i < points.size(); i++)
		labels.push_back((int)points[i].label);

	features.push_back(labels);

	return features;
}