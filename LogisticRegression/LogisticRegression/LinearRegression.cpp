#include "LinearRegression.h"

LinearRegression::LinearRegression(mat x, vec y, double regularizationParm)
	: X{ x }, y{ y }, regularizationParm{ regularizationParm }, trained{ false }
{
	assert(X.n_rows == y.n_rows);

	// Create bias column and append at the end of  x
	mat bias = ones<mat>(this->numberSamples(), 1);
	this->X.insert_cols(0, bias);
	this->InitalizeTheta();
}

LinearRegression::~LinearRegression() {}

void LinearRegression::getNormalizeEquation()
{
	mat X2 = this->X.t() * this->X;
	mat L = eye(X2.n_rows, X2.n_cols);
	L[0] = 0;
	// Check if X^2 is full-rank matrix
	if (arma::rank(X2) == X2.n_rows)
	{
		if (this->regularizationParm > 0)
			this->theta = pinv(X2 - (this->regularizationParm * L)) * this->X.t() * this->y;
		else
			this->theta = pinv(X2) * this->X.t() * this->y;

		this->trained = true;
	}
}

void LinearRegression::Train(TrainingType type, double alpha, unsigned int epoch)
{
	if (type == TrainingType::NORMALEQUATION)
		this->getNormalizeEquation();
	else if (type == TrainingType::GRADIENTDESCENT)
		this->trainGradientDescent(this->X, this->y, alpha, epoch);
	else
		cout << "Invalid training type" << endl;
}

uword LinearRegression::numberSamples()
{
	return this->X.n_rows;
}

vec LinearRegression::forward(mat _X)
{
	if (!this->trained)
	{
		cout << "This model hasn't been trained" << endl;
		return 0;
	}

	mat inputX = _X;
	if (this->theta.n_rows > _X.n_cols)
	{
		mat bias = ones<mat>(_X.n_rows, 1);
		inputX.insert_cols(0, bias);
	}

	return inputX * this->theta;
}

vec LinearRegression::Predict(mat _X)
{
	if (!this->trained)
	{
		cout << "Model does not trained yet" << endl;
		return 0;
	}

	vec prob = this->forward(_X);
	return prob;
}

double LinearRegression::getCost()
{
	vec h = forward(this->X);
	return this->calcualteCost(h, this->y);
}

double LinearRegression::calcualteCost(vec h, vec y_true)
{
	// J(Theta) = 1/2m * (X Theta - y)^T (X Theta - y) + lambda theta^2
	vec ve = h - y_true;
	vec thetaWithoutFirst = this->theta;
	thetaWithoutFirst[0] = 0;
	double res = (((double)1 / 2) * y_true.n_rows * dot(ve, ve) +
		this->regularizationParm * dot(thetaWithoutFirst, thetaWithoutFirst));
	return res;
}

vec LinearRegression::calculateCostDerivative(mat _X, vec h, vec y_true)
{
	vec deriv = ((h - y_true).t() * _X).t();
	vec thetaWithoutFirst = this->theta;
	thetaWithoutFirst[0] = 0;
	vec res = 1 / (double)y_true.n_rows * deriv + this->regularizationParm / (double)y_true.n_rows * thetaWithoutFirst;
	return res;
}

void LinearRegression::trainGradientDescent(mat _X, vec y_true, double alpha, unsigned int epoch)
{
	double prev_cost = INFINITY;
	this->trained = true;

	for (unsigned int i = 0; i < epoch; i++)
	{
		vec h = forward(_X);
		double cost = calcualteCost(h, y_true);
		if ((prev_cost - cost) < costDiffrenceThreshold)
			break;

		vec thetaDrivative = calculateCostDerivative(_X, h, y_true);
		updateTheta(thetaDrivative, alpha);
		prev_cost = cost;
	}
}

void LinearRegression::updateTheta(vec thetaDrivative, double alpha)
{
	this->theta = this->theta - (alpha * thetaDrivative);
}

void LinearRegression::InitalizeTheta()
{
	if (this->theta.n_rows != this->X.n_cols)
		this->theta = randu<vec>(this->X.n_cols);
}
