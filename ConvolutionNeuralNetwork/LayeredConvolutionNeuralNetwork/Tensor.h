#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <random>
#include <assert.h>
#include <memory>
#include <stdarg.h>
#include <iostream>

using namespace std;
/*
	Tensor class - support form of 1 to 4 dimensions
*/
template<typename T>
class Tensor
{
private:
	T* m_Data; 
	int m_Size = -1; // -1 means the size is undefined

public:
	int numDims = 0;
	int m_Dims[4]{}; // Max tensor dimensions is 4 (could be unlimited, but this makes the implementation simpler)

	Tensor() = default;
	Tensor(int numDims, int const* dims);
	Tensor(vector<int> m_Dims);
	Tensor(const Tensor<T>& other);
	Tensor(const unique_ptr<Tensor<T>>& other);

	void view(int newNumDims, int* newDims);
	void zero();
	int GetDimLen(int dimIndex, int start, int end);
	int GetSize();
	T GetListIndex(int i);

	T get(int i); //1D tensor
	T get(int i, int j); //2D tensor
	T get(int i, int j, int k); //3D tensor
	T get(int i, int j, int k, int l); //4D tensor
	//T get(const unsigned int args, ...); //nD tensor

	void set(T value, int i); //1D tensor
	void set(T value, int i, int j); //2D tensor
	void set(T value, int i, int j, int k); //3D tensor
	void set(T value, int i, int j, int k, int l); //4D tensor
	void SetListIndex(T value, int i);
	//void set(T value, const unsigned int args, ...); //nD tensor

	void add(T value, int i); // add value to 1D
	void add(T value, int i, int j, int k, int l); // add value to 4D
	//void add(T value, const unsigned int args, ...); // add value to nD

	/*
		Matrix Multiplication
	*/
	Tensor<T> matmul(Tensor<T> other);
	
	/*
		Return the transpose
	*/
	Tensor<T> transpose();	

	/*
		Sum every element
	*/
	T sum();

	/*
		Sum of 2d Tensor
	*/	
	Tensor<T> operator+(Tensor<T>& other);
	Tensor<T> operator+(unique_ptr<Tensor<T>>& other);

	/*
		Element wise multiplication of two 2D tensors
	*/
	Tensor<T> operator*(Tensor<T> other2D);
	Tensor<T> operator*(unique_ptr<Tensor<T>> other2D);

	/*
		Multiplies every element of the tensor by a value
	*/
	Tensor<T> operator*(T multiplier);

	/*
		Divide every elemtn of the tensor by a value
	*/
	Tensor<T> operator/(T divisor);

	/*
		assign tensor form another
	*/
	Tensor<T>& operator=(const Tensor<T>& other);
	Tensor<T>& operator=(const unique_ptr<Tensor<T>>& other);

	/*
		Subtracts Two 2D tensor
	*/
	Tensor<T> operator-=(Tensor<T> difference);
	Tensor<T> operator-=(unique_ptr<Tensor<T>> difference);

	/*
		Calcualtes the mean across of each row
	*/
	Tensor<T> ColumnWiseSum();

	/*
		Initializes a tensor\s values from a distribtution
	*/

	void randn(default_random_engine generator, normal_distribution<double> distribution, double multiplier);

	/*
		prints the tensor data
	*/
	void print();

	/*
		tensor deconstructor
	*/
	virtual ~Tensor();
};


#endif // !TENSOR_H
