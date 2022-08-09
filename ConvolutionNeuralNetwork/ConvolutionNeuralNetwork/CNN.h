#ifndef CNN_H
#define CNN_H

#include "LinearAlgebra.h"

using namespace std;

class CNN
{
private:
	vector<unique_ptr<Matrix>> weights;
	unique_ptr<Matrix> kernel;

	Shape poolWindow;

	void BackPropagate(unique_ptr<vector<double>>& deltaL, vector<unique_ptr<Matrix>>& convActivations, vector<unique_ptr<vector<double>>>& activations,
		unique_ptr<Matrix>& input, double (*activateFuncDeriv)(double), double learningRate);

	unique_ptr<vector<double>> MaxPooling(unique_ptr<Matrix> &conv, vector<unique_ptr<Matrix>>& convActivations);

	void ForwardPropagate(unique_ptr<Matrix>& input, vector<unique_ptr<Matrix>>& convActivations, vector<unique_ptr<vector<double>>>& activations);

	double CrossEntropy(unique_ptr<vector<double>>& yHat, unique_ptr<vector<double>>& yTrue);

public:
	CNN(Shape inputDim, Shape kernelSize, Shape poolSize, size_t hiddenLayerNodes, size_t outputDim);

	void train(vector<unique_ptr<Matrix>>& xTrain, vector<unique_ptr<vector<double>>>& yTrain, double learningRate, size_t epochs);

	double validate(vector<unique_ptr<Matrix>>& xVal, vector<unique_ptr<vector<double>>>& yVal);

	void info();
};

#endif // !CNN_H
