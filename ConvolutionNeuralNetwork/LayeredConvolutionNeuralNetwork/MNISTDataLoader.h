#ifndef MNISTDATALOADER_H
#define MNISTDATALOADER_H

#include <stdio.h>
#include <string>
#include <iostream>
#include <fstream>
#include <errno.h>
#include <string.h>
#include "Tensor.h"

using namespace std;

class MNISTDataLoader
{
private:
	vector<vector<vector<double>>> xSample;
	vector<int> ySample;

	unsigned int batchIdx = 0;
	unsigned int batchSize;
	unsigned int rows;
	unsigned int cols;
	unsigned int numLabels;
	unsigned int numImages = 0;

	/*
		Load MNIST labels
	*/
	void LoadLabels(string const& path);

	/*
		Converts an array of 4bytes to an unsigned int
	*/
	unsigned int bytesToUInt(const char* bytes);

	/*
		Load MNIST bytes image set
	*/
	void LoadImages(const string& imagesPath);

	void LoadMNISTBytes(const string& imagesPath, const string& labelsPath);

public:
	MNISTDataLoader(unsigned int batchSize=1, unsigned int rows=28, unsigned int cols=28);
	virtual ~MNISTDataLoader() = default;

	/*
		Get the number of batches in the dataset
	*/
	int GetNumBatches();

	/*
		Get next batch, lst batch of the dataset may not have the same size og the others
		Is cyclical, so it can be used indefinitely 
	*/
	pair<unique_ptr<Tensor<double>>, vector<int>> NextBatch();

	unique_ptr<Tensor<double>> GetxTensorShape();
	vector<vector<vector<double>>> GetxSamples();
	vector<int> GetySamples();
	void LoadMNISTImage(string filePath, int nImages=-1);
	void LoadMNISTCSV(string filePath);
	void LoadImage(string filePath);
};

#endif // MNISTDATALOADER_H
