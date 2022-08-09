#include "Tensor.h"

int main()
{
	vector<vector<vector<vector<int>>>> a;
	for (int i = 0; i < 32; i++)
	{
		int count = 0;
		vector<vector<vector<int>>> b;
		for (int j = 0; j < 1; j++)
		{
			vector<vector<int>> c;
			for (int k = 0; k < 28; k++)
			{
				vector<int> d;
				for (int m = 0; m < 28; m++)
				{
					d.push_back(count);
					count++;
				}	
				c.push_back(d);
			}
			b.push_back(c);
		}
		a.push_back(b);
	}

	vector<vector<int>> c;
	int count = 0;
	for (int k = 0; k < 28; k++)
	{
		vector<int> d;
		for (int m = 0; m < 28; m++)
		{
			d.push_back(count);
			count++;
		}
		c.push_back(d);
	}

	int xDims[] = { 28,28 };
	Tensor<double> x{ 2, xDims };

	//int yDims[] = { 1,2,3,4 };
	//Tensor<double> y{ 4, yDims };

	for (int i = 0; i < (28* 28); i++)
	{
		x.SetListIndex(i, i);
	}

	x.transpose();
	unique_ptr<Tensor<double>> xPtr= make_unique<Tensor<double>>(x);

	x.transpose();

	xPtr->transpose();
	xPtr = move(xPtr);

	return 0;
}