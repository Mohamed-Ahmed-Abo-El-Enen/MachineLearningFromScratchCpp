#pragma once
#include <armadillo>
#include <assert.h>
#include <iostream>
#include "TrainingType.h"

using namespace arma;
using namespace std;

class LinearRegression
{
private:
	// features
	mat X;
	// target vector elments between 0 or 1
	vec y;

	// vector for prediction
	vec theta;

	// regularization rate
	double regularizationParm;

	// model trained or not
	bool trained;

	double costDiffrenceThreshold = 5e-6;

	// Initialize theta
	void InitalizeTheta();

	// Compute cost function from derivative
	vec calculateCostDerivative(mat _X, vec h, vec y_true);

	//Normalize Equation
	void getNormalizeEquation();

	// Calcualte Cost function
	double calcualteCost(vec h, vec y_true);

	// forward path
	vec forward(mat x);

	// train with gradient descent
	void trainGradientDescent(mat _X, vec y_true, double alpha, unsigned int epoch);

	// Calculate GradientDescent
	void updateTheta(vec thetaDrivative, double alpha);

public:
	// Create a new instance from the given data set.
	LinearRegression(mat x, vec y, double regularizationParm = 0);

	// Destructor
	~LinearRegression();

	// Train the model
	void Train(TrainingType Type, double alpha = 0, unsigned int epoch = 0);

	// Return number of sampels
	uword numberSamples();

	// Predict y according to given x
	vec Predict(mat x);

	// Cost function using the own data;
	double getCost();
};