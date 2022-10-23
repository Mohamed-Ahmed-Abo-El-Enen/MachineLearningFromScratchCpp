#ifndef UTILS_H
#define UTILS_H
#pragma once

#include <vector>
#include <numeric>
#include <algorithm>

using namespace std;

double CalcualteAUC(vector<int>& yTrue, vector<double>& yHat);
double CalcualteKS(vector<int>& yTrue, vector<double>& yHat);
double CalcualteACC(vector<int>& yTrue, vector<double>& yHat);

#endif // !UTILS_H