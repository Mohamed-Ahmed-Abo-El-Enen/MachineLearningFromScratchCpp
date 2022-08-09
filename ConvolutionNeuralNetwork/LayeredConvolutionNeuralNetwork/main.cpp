#include <stdio.h>
#include <stdarg.h>
#include "MNISTDataLoader.h"
#include "Utils.h"
#include "Module.h"
#include "Conv2D.h"
#include "MaxPool.h"
#include "Dropout.h"
#include "Sigmoid.h"
#include "Tanh.h"
#include "LeakyRelu.h"
#include "Relu.h"
#include "Dense.h"
#include "LinearLRScheduler.h"
#include "NetworkModel.h"
#include "Softmax.h"

using namespace std;

int main()
{
	printf("Loading training set ...\n");

	const char* path = "../Dataset/mnist_png/training/";
	unsigned int batchSize = 32;
	unsigned int rows = 28;
	unsigned int cols = 28;
	MNISTDataLoader trainLoader(batchSize, rows, cols);
	trainLoader.LoadMNISTImage(path, 100);
	printf("DatasetLoaded\n");

	int seed = 0;

	vector<Module*> modules = 
	{
		new Conv2D(FilterShape(1,8), MatShape(3,3), 1, 0, seed),
		new Conv2D(FilterShape(8,4), MatShape(3,3), 1, 1, seed),
		new Conv2D(FilterShape(4,2), MatShape(3,3), 1, 0, seed),
		new MaxPool(2, 2),
		new Relu(),
		new Dropout(),
		new Dense(288, 128, seed),
		new Relu(),
		new Dense(128, 10, seed),
	};
	
	auto lrSchedule = new LinearLRScheduler(0.2, -0.000005);
	NetworkModel model = NetworkModel(modules, new Softmax(), lrSchedule);

	model.compile(trainLoader.GetxTensorShape(), true);

	int epochs = 10;
	model.train(epochs, trainLoader);
	model.eval();

	model.save("network.txt");
	
	model.load("network.txt");

	printf("Loading testing set... ");
	fflush(stdout);
	MNISTDataLoader testLoader(batchSize, rows, cols);
	testLoader.LoadMNISTImage(path, 100);
	printf("DatasetLoaded");

	Evaluation::CalculateAccuracy(model, testLoader);

	return 0;
}