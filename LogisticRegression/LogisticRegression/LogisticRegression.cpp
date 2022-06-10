#include "LogisticRegression.h"

LogisticRegression::LogisticRegression(mat x, vec y, double regularizationParm)
	: X{ x }, y{ y }, regularizationParm{ regularizationParm }, trained{ false } 
{
	assert(X.n_rows == y.n_rows);

	// Create bias column and append at the end of  x
	vec bias = ones<vec>(this->numberSamples());
	this->X.insert_cols(0, bias);
	this->InitalizeTheta();
}

LogisticRegression::~LogisticRegression(){}

void LogisticRegression::getNormalizeEquation()
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

void LogisticRegression::Train(TrainingType type, double alpha, unsigned int epoch)
{
	if (type == TrainingType::NORMALEQUATION)
		this->getNormalizeEquation();
	else if (type == TrainingType::GRADIENTDESCENT)
		this->trainGradientDescent(this->X, this->y, alpha, epoch);
	else
		cout << "Invalid training type" << endl;
}

uword LogisticRegression::numberSamples()
{
	return this->X.n_rows;
}

vec LogisticRegression::forward(mat _X)
{
	if (!this->trained)
	{
		cout << "This model hasn't been trained" << endl;
		return 0;
	}

	mat inputX = _X;
	if (this->theta.n_rows > _X.n_cols)
	{
		vec bias = ones<vec>(_X.n_rows);
		inputX.insert_cols(0, bias);
	}
	
	return inputX * this->theta;
}

vec LogisticRegression::Predict(mat _X)
{
	if (!this->trained)
	{
		cout << "Model does not trained yet"<<endl;
		return 0;
	}

	vec prob = this->forward(_X);
	prob = getSigmoid(prob);

	for (size_t i = 0; i < prob.n_rows; i++)
	{
		if (prob[i] >= probabilityThreshold)
			prob[i] = 1;
		else
			prob[i] = 0;
	}

	return prob;
}

mat LogisticRegression::getSigmoid(mat Z)
{
	return 1 / (1 + exp(-Z));
}

double LogisticRegression::getCost()
{
	vec h = forward(this->X);
	h = getSigmoid(h);
	return this->calcualteCost(h, this->y);
}

double LogisticRegression::calcualteCost(vec h, vec y_true)
{
	// h=g(x theta)
	// J(theta) = 1/m * (-y^T log(h) - (1-y)^T log(1-h)) + lambda/2m theta^2
	vec ve = (-y_true.t() * log(h)) - ((1 - y_true).t() * log(1 - h));
	vec thetaWithoutFirst = this->theta;
	thetaWithoutFirst[0] = 0;
	double res = (1 / (double)y_true.n_rows * ve + this->regularizationParm / (double)y_true.n_rows * 2 * thetaWithoutFirst.t() * thetaWithoutFirst).eval()(0, 0);
	return res;
}	

vec LogisticRegression::calculateCostDerivative(mat _X, vec h, vec y_true)
{
	vec deriv = _X.t() * (h - y_true);
	vec thetaWithoutFirst = this->theta;
	thetaWithoutFirst[0] = 0;
	vec res = 1 / (double)y_true.n_rows * deriv + this->regularizationParm / (double)y_true.n_rows * thetaWithoutFirst;
	return res;
}

void LogisticRegression::trainGradientDescent(mat _X, vec y_true, double alpha, unsigned int epoch)
{
	double prev_cost = INFINITY;
	this->trained = true;

	int numberEpochSameCost = 0;
	for (unsigned int i = 0; i < epoch; i++)
	{
		if (numberEpochSameCost >= numberMaxEpochSameCost)
			break;

		vec h = forward(_X);
		h = getSigmoid(h);
		double cost = calcualteCost(h, y_true);
		
		
		if ((prev_cost - cost) < costDiffrenceThreshold)
			numberEpochSameCost++;

		vec thetaDrivative = calculateCostDerivative(_X, h, y_true);
		updateTheta(thetaDrivative, alpha);
		prev_cost = cost;
	}
}

void LogisticRegression::updateTheta(vec thetaDrivative, double alpha)
{
	this->theta = this->theta - (alpha * thetaDrivative);
}

void LogisticRegression::InitalizeTheta()
{
	if (this->theta.n_rows != this->X.n_cols)
		this->theta = randu<vec>(this->X.n_cols);
}
