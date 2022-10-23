#include "pandas.h"

namespace pandas
{
	Dataset ReadCSV(string filePath, char sep, double fillNA, int numRows)
	{
		Dataset dataset;
		vector<vector<double>> features;
		vector<int> labels;

		ifstream ifs(filePath);
		string line;
		int countRows = 0;
		while (getline(ifs, line) && (countRows < numRows))
		{
			if (!line.empty())
			{
				stringstream ss(line);
				vector<double> vectLine;
				string tmp;
				while (getline(ss, tmp, sep))
				{
					if (tmp == "")
						vectLine.push_back(fillNA);
					else
						vectLine.push_back(stod(tmp));
				}

				labels.push_back(int(vectLine.back()));
				vectLine.pop_back();
				features.push_back(vectLine);
			}
		}

		dataset = { features, labels };
		return dataset;
	}

	void SaveCSV(const vector<double>& datasetVect, const string filePath)
	{
		ofstream outFile;
		outFile.open(filePath, ios::out);
		for (double value : datasetVect)
			outFile << value << endl;

		outFile.close();
	}
}