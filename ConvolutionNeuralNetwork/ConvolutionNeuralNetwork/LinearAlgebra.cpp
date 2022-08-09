#include <vector>
#include <assert.h>
#include "LinearAlgebra.h"

using namespace std;

namespace np
{
	// Scaler Multiplication 
	unique_ptr<Matrix> multiply(unique_ptr<Matrix>& matrix, double value)
	{
		unique_ptr<Matrix> resultMat(new Matrix(matrix->GetRows(), matrix->GetColumns(), false));
		for (size_t i = 0; i < matrix->GetRows(); i++)		
			for (size_t j = 0; j < matrix->GetColumns(); j++)			
				resultMat->set(i, j, matrix->get(i, j) * value);
		
		return resultMat;
	}

	// Vector elements wise product 
	unique_ptr<vector<double>> multiply(unique_ptr<vector<double>>& vec1, unique_ptr<vector<double>>& vec2)
	{
		assert(vec1->size() == vec2->size());

		unique_ptr<vector<double>> resultVec = make_unique<vector<double>>();
		for (size_t i = 0; i < vec1->size(); i++)
			resultVec->push_back(vec1->at(i) * vec2->at(i));

		return resultVec;
	}


	// Matrix elements wise product 
	unique_ptr<Matrix> multiply(unique_ptr<Matrix>& matrix1, unique_ptr<Matrix>& matrix2, size_t slice)
	{
		assert(matrix1->GetRows() == matrix2->GetColumns() && matrix1->GetColumns() == matrix2->GetRows());

		unique_ptr<Matrix> resultMat(new Matrix(matrix2->GetRows(), matrix2->GetColumns(), true));
		for (size_t i = 0; i < resultMat->GetRows(); i++)
			for (size_t j = 0; j < resultMat->GetColumns(); j++)
				resultMat->set(i, j, matrix1->get(i, j) * matrix2->get(i, j));

		return resultMat;
	}

	// Dot product and sum all
	double multiply(unique_ptr<Matrix>& matrix1, unique_ptr<Matrix>& matrix2, size_t xSlice, size_t ySlice)
	{
		assert(matrix1->GetRows() <= matrix2->GetRows() && matrix1->GetColumns() <= matrix2->GetColumns());
		
		double result = 0;
		for (size_t i = 0; i < matrix1->GetRows(); i++)
			for (size_t j = 0; j < matrix1->GetColumns(); j++)
				result += matrix1->get(i, j) * matrix2->get(i + xSlice, j + ySlice);

		return result;
	}

	// Dot product of 2 matrix
	unique_ptr<Matrix> dot(unique_ptr<Matrix>& matrix1, unique_ptr<Matrix>& matrix2, size_t slice)
	{
		assert(matrix1->GetColumns() == matrix2->GetRows());

		double res = 0;
		unique_ptr<Matrix> resultMat(new Matrix(matrix1->GetRows(), matrix2->GetColumns() - slice, false));
		for (size_t i = 0; i < resultMat->GetRows(); i++)
		{
			for (size_t j = 0; j < resultMat->GetColumns(); j++)
			{
				res = 0;
				for (size_t k = 0; k < matrix1->GetColumns(); k++)
					res += (matrix1->get(i, k) * matrix2->get(k, j));

				resultMat->set(i, j, res);
			}
		}
		return resultMat;
	}

	// Dot product of Matrix and a vector
	unique_ptr<vector<double>> dot(unique_ptr<Matrix>& matrix, unique_ptr<vector<double>>& vec, size_t vSlice)
	{
		assert(matrix->GetColumns() == vec->size()-vSlice);

		double res = 0;
		unique_ptr<vector<double>> resultVec = make_unique<vector<double>>();
		for (size_t i = 0; i < matrix->GetRows(); i++)
		{
			res = 0;
			for (size_t j = 0; j < matrix->GetColumns() - vSlice; j++)
				res += (matrix->get(i, j) * vec->at(j));
		
			resultVec->push_back(res);
		}
		return resultVec;
	}

	// Dot product of 2 vector
	unique_ptr<Matrix> dot(unique_ptr<vector<double>>& vec1, unique_ptr<vector<double>>& vec2, size_t v2Slice)
	{
		unique_ptr<Matrix> resultMat(new Matrix(vec1->size(), vec2->size() - v2Slice, true));
		for (size_t i = 0; i < vec1->size(); i++)		
			for (size_t j = 0; j < vec2->size() - v2Slice; j++)
				resultMat->set(i, j, (vec1->at(i) * vec2->at(j)));

		return resultMat;
	}

	// Addation
	unique_ptr<Matrix> add(unique_ptr<Matrix>& matrix1, unique_ptr<Matrix>& matrix2)
	{
		assert(matrix1->GetRows() == matrix2->GetRows() && matrix1->GetRows() == matrix2->GetColumns());

		unique_ptr<Matrix> resultMat(new Matrix(matrix1->GetRows(), matrix1->GetColumns(), true));
		for (size_t i = 0; i < matrix1->GetRows(); i++)		
			for (int j = 0; j < matrix1->GetColumns(); j++)			
				resultMat->set(i, j, matrix1->get(i, j) + matrix2->get(i, j));					

		return resultMat;
	}

	// subtraction (Vectors)
	unique_ptr<vector<double>> subtract(unique_ptr<vector<double>>& vec1, unique_ptr<vector<double>>& vec2)
	{
		assert(vec1->size() == vec2->size());

		unique_ptr<vector<double>> resultVec = make_unique<vector<double>>();
		for (size_t i = 0; i < vec1->size(); i++)
			resultVec->push_back(vec1->at(i) - vec2->at(i));

		return resultVec;
	}

	// subtraction (Matrices)
	unique_ptr<Matrix> subtract(unique_ptr<Matrix>& matrix1, unique_ptr<Matrix>& matrix2)
	{
		assert(matrix1->GetRows() == matrix2->GetRows() && matrix1->GetColumns() == matrix2->GetColumns());

		unique_ptr<Matrix> resultMat(new Matrix(matrix1->GetRows(), matrix1->GetColumns(), true));
		for (size_t i = 0; i < matrix1->GetRows(); i++)
			for (int j = 0; j < matrix1->GetColumns(); j++)
				resultMat->set(i, j, matrix1->get(i, j) - matrix2->get(i, j));

		return resultMat;
	}

	// transpose
	unique_ptr<Matrix> transpose(unique_ptr<Matrix>& matrix)
	{
		unique_ptr<Matrix> resultMat(new Matrix(matrix->GetColumns(), matrix->GetRows(), false));
		for (size_t i = 0; i < matrix->GetRows(); i++)
			for (size_t j = 0; j < matrix->GetColumns(); j++)
				resultMat->set(j, i, matrix->get(i, j));
					
		return resultMat;
	}

	// Apply a function to every elemnt of the vector
	unique_ptr<vector<double>> apply(unique_ptr<vector<double>>& vec, double(*function)(double))
	{
		unique_ptr<vector<double>> resultVec = make_unique<vector<double>>();
		for (size_t i = 0; i < vec->size(); i++)
		{
			double res = (*function)(vec->at(i));
			if (isnan(res))
				resultVec->push_back(0);
			else 
				resultVec->push_back(res);						
		}

		return resultVec;
	}

	// Apply a function to every elemnt of the matrix
	unique_ptr<Matrix> apply(unique_ptr<Matrix>& matrix, double(*function)(double))
	{
		unique_ptr<Matrix> resultMat(new Matrix(matrix->GetRows(), matrix->GetColumns(), false));
		for (size_t i = 0; i < matrix->GetRows(); i++)
		{
			for (size_t  j = 0; j < matrix->GetColumns(); j++)
			{
				double res = (*function)(matrix->get(i, j));
				if (isnan(res))
					resultMat->set(i, j, 0);
				else
					resultMat->set(i, j, res);
			}
		}
		return resultMat;
	}

	// Concatenate with another matrix
	unique_ptr<Matrix > concatenate(unique_ptr<Matrix>& matrix, vector<double>& vec)
	{
		assert(matrix->GetRows() == vec.size());

		unique_ptr<Matrix> resultMat(new Matrix(matrix->GetRows(), matrix->GetColumns() + 1, true));
		for (size_t i = 0; i < matrix->GetRows(); i++)
		{
			for (size_t j = 0; j < matrix->GetColumns(); j++)
				resultMat->set(i, j, matrix->get(i, j));
			
			resultMat->set(i, matrix->GetColumns(), vec[i]);
		}
		return resultMat;
	}

	// Concatenate with another matrix
	unique_ptr<Matrix > concatenate(unique_ptr<Matrix>& matrix1, unique_ptr<Matrix>& matrix2)
	{
		assert(matrix1->GetRows() == matrix2->GetRows());
		
		unique_ptr<Matrix> resultMat(new Matrix(matrix1->GetRows(), matrix1->GetRows() + matrix2->GetColumns()));
		for (size_t i = 0; i < matrix1->GetRows(); i++)
		{
			for (size_t j = 0; j < matrix1->GetColumns(); j++)			
				resultMat->set(i, j, matrix1->get(i, j));

			for (size_t j = 0; j < matrix2->GetColumns(); j++)
				resultMat->set(i, j, matrix2->get(i, j));			
		}
		return resultMat;
	}

	// normalize a vector such that each row is 1
	unique_ptr<vector<double>> normalize(unique_ptr<vector<double>>& vec)
	{
		unique_ptr <vector<double>> resultVec = make_unique<vector<double>>();

		double sum = 0;
		for (size_t i = 0; i < vec->size(); i++)		
			sum += vec->at(i);
		
		assert(sum != 0);
		for (size_t i = 0; i < vec->size(); i++)
			resultVec->push_back(vec->at(i) / sum);
		
		return resultVec;
	}

	// normalize a matrix such that each row is 1
	unique_ptr<Matrix> normalize(unique_ptr<Matrix>& matrix)
	{
		unique_ptr<Matrix> resultMat(new Matrix(matrix->GetRows(), matrix->GetColumns(), true));

		for (size_t i = 0; i < matrix->GetRows(); i++)
		{
			double sum = 0;
			for (size_t j= 0; j < matrix->GetColumns(); j++)			
				sum += matrix->get(i, j);
			
			for (size_t j = 0; j < matrix->GetColumns(); j++)
				resultMat->set(i, j, matrix->get(i, j) / sum);
		}
		return resultMat;
	}

	// Sum all element in vector
	double sum(unique_ptr<vector<double>>& vec)
	{
		double sum = 0;
		for (size_t i = 0; i < vec->size(); i++)		
			sum += vec->at(i);
		return sum;
	}

	// Sum all element in matrix
	double sum(unique_ptr<Matrix>& matrix)
	{
		double sum = 0;
		for (size_t i = 0; i < matrix->GetRows(); i++)
			for (size_t j = 0; j < matrix->GetColumns(); j++)
				sum += matrix->get(i, j);
		return sum;
	}

	// flatten the matrix
	unique_ptr<vector<double>> flatten(unique_ptr<Matrix>& matrix)
	{
		unique_ptr<vector<double>> resultVec(new vector<double>(matrix->GetRows() * matrix->GetColumns()));

		for (size_t i = 0; i < matrix->GetRows(); i++)		
			for (size_t j = 0; j < matrix->GetColumns(); j++)			
				resultVec->at(i * matrix->GetColumns() + j) = matrix->get(i, j);
		return resultVec;		
	}

	// maximum of matrix with boundries
	double maximum(unique_ptr<Matrix>& matrix, size_t xPtr, size_t yPtr, Shape window, unique_ptr<Shape>& index)
	{
		assert(xPtr + window.rows <= matrix->GetRows() && yPtr + window.columns <= matrix->GetColumns());

		double max = -DBL_MAX;
		size_t i = xPtr;
		while (i-xPtr < window.rows && i<matrix->GetRows())
		{
			size_t j = yPtr;
			while (j - yPtr < window.columns && i < matrix->GetColumns())
			{
				if (matrix->get(i, j) > max)
				{
					max = matrix->get(i, j);
					index->rows = i;
					index->columns = j;
				}
				j++;
			}
			i++;
		}
		return max;
	}

	// reshape a vector to matrix
	unique_ptr<Matrix> reshape(unique_ptr<vector<double>>& vec, Shape shape)
	{
		assert((shape.rows * shape.columns) == vec->size());
		unique_ptr<Matrix> resultMat(new Matrix(shape.rows, shape.columns, true));
		for (size_t i = 0; i < shape.rows; i++)
			for (size_t j = 0; j < shape.columns; j++)
				resultMat->set(i, j, vec->at(i * shape.rows) + j);
		
		return resultMat;
	}
}