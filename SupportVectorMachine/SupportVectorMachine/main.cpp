#include "Utils.h"
#include "SVM.h"
#include "WriteCSV.h"

int main() 
{
    string csv_file_path = "../Dataset/Cryotherapy.csv";
    vector<CSVSample> csvArr = CSVSample::readCSV(csv_file_path);

    int yIndex = 6;
    map<string, int> class_map;
    vector<int> categorical_cols = { };

    vector<vector<double>> X;
    vector<int> y;
    Sample::ConvertCSVSamplesXY(csvArr, categorical_cols, yIndex, class_map, X, y);

   
    vector<double> minVec;
    vector<double> maxVec;
    MinMaxScaler(X, minVec, maxVec);

    y = ConvertY2SVMY(y);

    int D = X[0].size();
    bool verbose = true;
    double lr = 0.005;
    double C = 10;

    kernelFunction kernel = kernel::rbf;
    double gamma = 5.0;
    vector<double> params = vector<double>();
    params.push_back(gamma);

    SVM svm(kernel, params, verbose);
    //SVM svm(verbose);
    svm.train(X, y, D, C, lr);

    vector<vector<double>> train_class1_data = SplitWithClassIndex(X, y, 1);
    vector<vector<double>> train_class2_data = SplitWithClassIndex(X, y, -1);

    svm.Evaluation(train_class1_data, train_class2_data);
    cout << "///////////////////////// Test /////////////////////////" << endl;
    cout << "accuracy-all: " << svm.accuaracy << " (" << svm.correctC1 + svm.correctC2 << "/" << train_class1_data.size() + train_class2_data.size() << ")" << endl;
    cout << "accuracy-class1: " << svm.accuracyC1 << " (" << svm.correctC1 << "/" << train_class1_data.size() << ")" << endl;
    cout << "accuracy-class2: " << svm.accuracyC2 << " (" << svm.correctC2 << "/" << train_class2_data.size() << ")" << endl;
    cout << "////////////////////////////////////////////////////////" << endl;

    vector<int> yHat = svm.predict(X);
    WriteCSV::writeCSVFile(csvArr, yHat, "../Dataset/Results.csv");

    return 0;
}