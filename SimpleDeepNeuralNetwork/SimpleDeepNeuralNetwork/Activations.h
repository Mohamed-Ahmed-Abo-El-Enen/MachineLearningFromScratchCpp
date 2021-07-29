#pragma once
#include<math.h>

enum ActivationFunctions
{
	SigmoidTag = 0,
	TanhTag = 1
};

inline double Sigmoid(double x)
{
	return 1 / (1 + exp(-1*x));
}

inline double SigmoidDrv(double x)
{
	return Sigmoid(x) * (1 - Sigmoid(x));
}

inline double Tanh(double x)
{
	return tanhf(x);
}

inline double TanhDrv(double x)
{
	return (1 - tanhf(x) * tanhf(x));
}
