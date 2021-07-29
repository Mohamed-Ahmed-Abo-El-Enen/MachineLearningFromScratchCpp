#pragma once
#include "XORDataHandler.h"
#include "MNIST.h"

using namespace std;

int main()
{
	//string datasetFilePath = "..\\Debug\\XORData.txt";
	//XORDataHandler xorDataHandler(datasetFilePath);
	//xorDataHandler.GenerateXORData();

	MNIST mnist;
    string imagesFilePath = "..\\Debug\\train-images.idx3-ubyte";
    string lableFilePath = "..\\Debug\\train-labels.idx1-ubyte";
    vector<vector<double>> samplesFeature;
    vector<vector<double>> samplesLabel;
    mnist.GetMNISTDataset(imagesFilePath, lableFilePath, samplesFeature, samplesLabel);
	mnist.GenerateMNISTDataFormat(samplesFeature, samplesLabel);

	return 0;
}