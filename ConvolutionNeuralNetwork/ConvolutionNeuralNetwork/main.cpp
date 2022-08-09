#include "CNN.h"

using namespace std;

int Test1()
{
	Shape inputDim = { 28, 28 };
	Shape kernelDim = { 6,6 };
	Shape poolSize = { 4,4 };
	
	unique_ptr<CNN> cnn(new CNN(inputDim, kernelDim, poolSize, 30, 10));

	const char* path = "../Dataset/mnist_png/training/";
	vector <unique_ptr<Matrix>> xTrain;
	vector <unique_ptr<vector<double>>> yTrain;

	Prerocess::ProcessMNISTImage(path, xTrain, yTrain, 100);

	cnn->train(xTrain, yTrain, 0.01, 30);
	
	return 0;
}

int main()
{
	Test1();
}