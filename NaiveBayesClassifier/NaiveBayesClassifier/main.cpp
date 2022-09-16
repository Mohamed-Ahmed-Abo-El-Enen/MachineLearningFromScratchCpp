#include <vector>
#include "NaiveBayes.h"

int main()
{
    int yIndex = 4;

	string filePath = "../Dataset/iris.data";
	vector<vector<double>> dataset = DataReader::ReadIrisDataset(filePath, yIndex);

	random_shuffle(dataset.begin(), dataset.end());
    float testSize = 0.2;
    vector<vector<double>> trainDataset;
    vector<vector<double>> testDataset;
    DataManipulation::Train_Test_Split(dataset, testSize, trainDataset, testDataset);

    cout<<"Iris train data size = " << trainDataset.size() << endl;

    cout<<"Iris test data size = " << testDataset.size() << endl;
    cout << "\n";

    NaiveBayes naiveBayes = NaiveBayes();
    naiveBayes.fit(trainDataset, yIndex);

    vector<int> predicitions;
    vector<vector<double>> xTest = DataManipulation::RemoveColumnDataset(testDataset, 4);
    for (int i = 0; i < xTest.size(); i++)
    {
        auto index = naiveBayes.predict(xTest[i]);
        predicitions.push_back(index);
    }

    cout << "Test Dataset Accuracy = " << Evaluation::CalculateAccuracy(DataManipulation::GetColumnValues(testDataset, 4), predicitions) << endl;
    naiveBayes.PrintClassesDistribution();
    return 0;
}