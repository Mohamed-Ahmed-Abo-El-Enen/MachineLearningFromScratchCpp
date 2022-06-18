#include"WriteCSV.h"

void WriteCSV::writeCSVFile(vector<CSVSample> points, vector<int> predictions, string csv_file_path)
{
	ofstream file;
	file.open(csv_file_path);
	string line = "";
	int numFeat = points[0].values.size();

	for (size_t k = 0; k < numFeat-1; k++)
		line += "Feat_" + to_string(k + 1) + ',';

	line += "y_true,";
	line += "y_hat";
	file << line << endl;
	for (int i = 0; i < points.size(); i++)
	{
		line = "";
		for (size_t k = 0; k < numFeat; k++)
			line += points[i].values[k] + ',';
		line += to_string(predictions[i]);
		file << line << endl;
	}

	file.close();
}