#include "Matrix.h"

using namespace std;

time_t seed = time(0);
default_random_engine random_engine((unsigned int)seed);

Matrix::Matrix() {}

//Matrix::~Matrix() {}

Matrix::Matrix(size_t rows, size_t columns, bool init)
{
	random_engine.seed(1);
	this->rows = rows;
	this->columns = columns;
	this->array = vector<vector<double>>(rows, vector<double>(columns));

	int factor = 0;
	if (init)
		factor = 1;

	uniform_real_distribution<double> unif(-1, 1);
	for (unsigned int i = 0; i < rows; i++)
	{
		for (unsigned int j = 0; j < columns; j++)
		{
			double x = unif(random_engine);
			this->array[i][j] = (x * factor);
		}
	}
}

Matrix::Matrix(vector<vector<double>> const& array)
{
	assert(array.size() != 0);
	random_engine.seed(1);
	this->rows = array.size();
	this->columns = array[0].size();
	this->array = array;
}

void Matrix::set(size_t row, size_t column, double value)
{
	if (row >= rows || column >= columns)
		assert("ERROR Row index Or Column index out of bound");
	else
		this->array[row][column] = value;
}

double Matrix::get(size_t row, size_t column)
{
	if (row >= rows || column >= columns)
	{
		assert("ERROR Row index Or Column index out of bound");
		return double(0);
	}
	else
		return this->array[row][column];	 
}

void Matrix::pretty_print()
{
	for (unsigned int i = 0; i < rows; i++)
	{
		for (unsigned int j = 0; j < columns; j++)		
			cout << array[i][j] << " ";		
		cout << endl;
	}
}