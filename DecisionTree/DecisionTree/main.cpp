#include "CSVReader.h"
#include "DecisionTree.h"
#include "WriteCSV.h"

int main()
{
    string csv_file_path = "../Dataset/Mall_data.csv";
    vector<CSVSample> csvArr = CSVSample::readCSV(csv_file_path);

    int yIndex = 0;
    map<string, int> class_map;
    vector<int> categorical_cols = { };

    CSVSample::removeCol(csvArr, 0);

    vector<Sample> points = Sample::ConvertCSVSamples(csvArr, categorical_cols, yIndex, class_map);

    DecisionTree* pTree = new DecisionTree(Sample::getConcatedDataframe(points));

    vector<int> predictions = pTree->predict(Sample::getConcatedDataframe(points));

    WriteCSV::writeCSVFile(csvArr, predictions, "../Dataset/Results.csv");

    return 0;
}