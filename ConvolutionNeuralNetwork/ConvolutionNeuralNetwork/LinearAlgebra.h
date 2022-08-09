#ifndef LINEARALGEBRA_H
#define LINEARALGEBRA_H

#include "Util.h"

namespace np
{
	// Scaler Multiplication 
	unique_ptr<Matrix> multiply(unique_ptr<Matrix>& matrix, double value);

	// Vector elements wise product 
	unique_ptr<vector<double>> multiply(unique_ptr<vector<double>>& vec1, unique_ptr<vector<double>>& vec2);

	// Matrix elements wise product 
	unique_ptr<Matrix> multiply(unique_ptr<Matrix>& matrix1, unique_ptr<Matrix>& matrix2, size_t slice=0);

	// Dot product and sum all
	double multiply(unique_ptr<Matrix>& matrix1, unique_ptr<Matrix>& matrix2, size_t xSlice, size_t ySlice);

	// Dot product of 2 matrix
	unique_ptr<Matrix> dot(unique_ptr<Matrix>& matrix1, unique_ptr<Matrix>& matrix2, size_t slice = 0);

	// Dot product of Matrix and a vector
	unique_ptr<vector<double>> dot(unique_ptr<Matrix>& matrix, unique_ptr<vector<double>>& vec, size_t vSlice = 0);

	// Dot product of 2 vector
	unique_ptr<Matrix> dot(unique_ptr<vector<double>>& vec1, unique_ptr<vector<double>>& vec2, size_t v2Slice = 0);

	// Addation
	unique_ptr<Matrix> add(unique_ptr<Matrix> &matrix1, unique_ptr<Matrix>& matrix2);

	// subtraction (Vectors)
	unique_ptr<vector<double>> subtract(unique_ptr<vector<double>>& vec1, unique_ptr<vector<double>>& vec2);

	// subtraction (Matrices)
	unique_ptr<Matrix> subtract(unique_ptr<Matrix>& matrix1, unique_ptr<Matrix>& matrix2);

	// transpose
	unique_ptr<Matrix> transpose(unique_ptr<Matrix>& matrix);

	// Apply a function to every elemnt of the vector
	unique_ptr<vector<double>> apply(unique_ptr<vector<double>>& vec, double(*function)(double));

	// Apply a function to every elemnt of the matrix
	unique_ptr<Matrix> apply(unique_ptr<Matrix>& matrix, double(*function)(double));

	// Concatenate with another matrix
	unique_ptr<Matrix > concatenate(unique_ptr<Matrix>& matrix, vector<double>& vec);

	// Concatenate with another matrix
	unique_ptr<Matrix > concatenate(unique_ptr<Matrix>& matrix1, unique_ptr<Matrix>& matrix2);

	// normalize a vector such that each row is 1
	unique_ptr<vector<double>> normalize(unique_ptr<vector<double>> &vec);

	// normalize a matrix such that each row is 1
	unique_ptr<Matrix> normalize(unique_ptr<Matrix>& matrix);

	// Sum all element in vector
	double sum(unique_ptr<vector<double>> &vec);

	// Sum all element in matrix
	double sum(unique_ptr<Matrix>& matrix);

	// flatten the matrix
	unique_ptr<vector<double>> flatten(unique_ptr<Matrix>& matrix);

	// maximum of matrix with boundries
	double maximum(unique_ptr<Matrix>& matrix, size_t xPtr, size_t yPtr, Shape window, unique_ptr<Shape>& index);

	// reshape a vector to matrix
	unique_ptr<Matrix> reshape(unique_ptr<vector<double>>& vec, Shape shape);
}

#endif // !LINEARALGEBRA_H
