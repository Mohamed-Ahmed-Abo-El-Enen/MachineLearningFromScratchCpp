#ifndef MATRIX_H
#define MATRIX_H

#include <iostream>
#include <assert.h>
#include <vector>
#include <random>
#include <time.h>

using namespace std;
extern default_random_engine random_engine;

class Matrix
{
private:
	vector<vector<double>> array;
	size_t rows;
	size_t columns;

public:
	Matrix();
	virtual ~Matrix() = default;
	Matrix(size_t rows, size_t columns, bool init = false);
	Matrix(vector<vector<double>> const& array);

	void set(size_t row, size_t colum, double value);
	double get(size_t row, size_t column);

	size_t GetRows() const { return rows; }
	size_t GetColumns() const { return columns; }

	void pretty_print();
};

#endif // !MATRIX_H
