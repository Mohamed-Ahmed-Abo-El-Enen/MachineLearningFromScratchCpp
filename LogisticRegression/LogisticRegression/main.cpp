#include "LogisticRegression.h"
#include "LinearRegression.h"
#include "CSVReader.h"
#include "Node.h"
#include "WriteCSV.h"
#include "Utilites.h"

void regTest() 
{
    mat X = mat("3 0; 0 4;2 5;0 0; 5 2; 7 0; 0 8");
    vec y = vec("3 8 12 0 9 7 16");

    rowvec minVec;
    rowvec maxVec;
    X = MinMaxScaler(X, minVec, maxVec);

    LinearRegression* reg = new LinearRegression(X, y);
    reg->Train(GRADIENTDESCENT, 0.1, 1000);

    vec prediction = reg->Predict(X);
    for (size_t i = 0; i < reg->numberSamples(); i++)
        cout << "Regressor : " << y[i] << " ---> " << reg->Predict(conv_to<mat>::from(X.row(i)))[0] << endl;

}

void clfTest() 
{
    mat X = mat("2 2; 3 3; 4 4; -1 2; -10 8; 8 9; -4 -4;1 -3; 8 1; 3 -4; 1 -1; "
        "-1 1; 4 -6");
    vec y = vec("1 1 1 0 0 1 0 0 1 0 0 0 0");

    rowvec minVec;
    rowvec maxVec;
    X = MinMaxScaler(X, minVec, maxVec);

    LogisticRegression* clf = new LogisticRegression(X, y);
    clf->Train(GRADIENTDESCENT, 0.1, 1000);

    vec prediction = clf->Predict(X);
    for (size_t i = 0; i < clf->numberSamples(); i++)    
        cout << "Classifier : " << y[i] << " ---> " << clf->Predict(conv_to<mat>::from(X.row(i)))[0] << endl;
    
}

int main() 
{
    regTest();
    clfTest();
 
    string csv_file_path = "../Dataset/diabetes.csv";
    vector<CSVSample> csvArr = CSVSample::readCSV(csv_file_path);

    int yIndex = 8;
    map<string, int> class_map;
    vector<int> categorical_cols = { };

    vector<Sample> points = Sample::ConvertCSVSamples(csvArr, categorical_cols, yIndex, class_map);

    mat X;
    vec y;
    convrtVectorSample2Mat(points, X, y);

    rowvec minVec;
    rowvec maxVec;
    X = MinMaxScaler(X, minVec, maxVec);

    LogisticRegression* clf = new LogisticRegression(X, y);
    clf->Train(GRADIENTDESCENT, 0.5, 1000);
    cout <<"Cost : "<< clf->getCost() << endl;

    vec yhat = clf->Predict(X);
    cout << "Accuracy : " << calcuateAccuracy(y, yhat) << endl;
    
    vector<int> prediction = conv_to<vector<int>>::from(yhat);

    //for (size_t i = 0; i < clf->numberSamples(); i++)
    //    cout << "Classifier : " << y[i] << " ---> " << prediction[i] << endl;

    WriteCSV::writeCSVFile(csvArr, prediction, "../Dataset/Results.csv");

    return 0;
}