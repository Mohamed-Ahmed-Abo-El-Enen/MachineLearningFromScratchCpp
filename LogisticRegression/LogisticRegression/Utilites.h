#pragma once
#include <armadillo>

using namespace arma;

mat MinMaxScaler(mat X, rowvec& minVec, rowvec& maxVec)
{
	maxVec = max(X, 0); 
	minVec = min(X, 0);

	mat scaledX = X;
	for (size_t rowIndex = 0; rowIndex < scaledX.n_rows; rowIndex++)
		scaledX.row(rowIndex) = (scaledX.row(rowIndex) - minVec) / (maxVec - minVec);
	return scaledX;
}


void convrtVectorSample2Mat(vector<Sample> points, mat& X, vec& y)
{
    string xString = "";
    string yString = "";
    for (Sample s : points)
    {
        for (double f : s.features)
        {
            xString += to_string(f) + ' ';
        }
        xString += ';';

        yString += to_string(s.label) + ' ';
    }
    X = mat(xString);
    y = vec(yString);
}

float calcuateAccuracy(vec ytrue, vec yhat)
{
    float correct = 0.0;
    for (size_t i = 0; i < ytrue.n_rows; i++)
    {
        if (ytrue[i] == yhat[i])
            correct++;
    }
    return (correct / ytrue.n_rows);
}