#include "Utils.h"
#include "OneClassSVM.h"
#include "WriteCSV.h"

int main() 
{
    string csv_file_path = "../Dataset/AnomalyDataset.csv";
    vector<CSVSample> csvArr = CSVSample::readCSV(csv_file_path);

    int yIndex = 2;
    map<string, int> class_map;
    vector<int> categorical_cols = { };

    vector<vector<double>> X;
    vector<int> y;
    Sample::ConvertCSVSamplesXY(csvArr, categorical_cols, yIndex, class_map, X, y);

   
    vector<double> minVec;
    vector<double> maxVec;
    MinMaxScaler(X, minVec, maxVec);

    int D = X[0].size();
    bool verbose = true;
    double nu  = 0.003;
    double lr = 0.001;

    kernelFunction kernel = kernel::rbf;
    double gamma = 5.0;
    vector<double> params = vector<double>();
    params.push_back(gamma);

    OneClassSVM oneClassSVM(kernel, params, verbose);
    oneClassSVM.train(X, D, nu, lr);

    vector<vector<double>> normalDataset = SplitWithClassIndex(X, y, 1);
    vector<vector<double>> AnomalyDataset = SplitWithClassIndex(X, y, -1);

    oneClassSVM.Evaluation(normalDataset, AnomalyDataset);

    vector<int> yHat = oneClassSVM.predict(X);
    WriteCSV::writeCSVFile(csvArr, yHat, "../Dataset/AnomalyResults.csv");

    return 0;
}