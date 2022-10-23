#ifndef NUMPY_H
#define NUMPY_H

#pragma once

#include <vector>
#include <iostream>

using namespace std;

namespace numpy
{
		template<typename T, typename VECT_T = vector<T>>
	double Percentile(const VECT_T& vect, T p)
	{
		if (!p)
			return (double)vect[0];

		else if (100 - p < 1e-5)
			return (double)vect[vect.size() - 1];

		else
		{
			double temp = (vect.size() - 1) * p / 100.0 + 1;
			int posInteger = floor(temp);
			double posDecimal = temp - posInteger;
			double res = vect[posInteger - 1] + (vect[posInteger] - vect[posInteger - 1]) * posDecimal;
			return (double)res;
		}
	}

	template<typename T>
	vector<double> LinSpace(T start, T end, int n)
	{
		double step = (end - start) * 1.0 / (n - 1);
		vector<double> res;
		for (int i = 0; i < n; i++)
			res.push_back(start + i * step);

		return res;
	}
}

#endif // !NUMPY_H