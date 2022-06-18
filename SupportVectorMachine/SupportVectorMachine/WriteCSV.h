#pragma once
#include "CSVReader.h"

class WriteCSV
{
public:
	static void writeCSVFile(vector<CSVSample> points, vector<int> predictions, string csv_file_path);
};
