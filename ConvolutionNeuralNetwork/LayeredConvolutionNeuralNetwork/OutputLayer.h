#ifndef OUTPUTLAYER_H
#define OUTPUTLAYER_H

#include<string>  
#include "Tensor.h"

class OutputLayer
{
protected:
	vector<int> standardOutputTensorDims;
	unique_ptr<Tensor<double>> m_Output;
	string costFunctionName;
public:
	
	virtual unique_ptr<Tensor<double>> predict(unique_ptr<Tensor<double>>& input) = 0;
	virtual pair<double, unique_ptr<Tensor<double>>> BackwardPropagate(unique_ptr<vector<int>>& yTrue) = 0;
	virtual void Compile(unique_ptr<Tensor<double>>& input) = 0;
	string GetcostFunctionName() { return costFunctionName; }
	string GetcostFunctionInfo();
	vector<int> GetStandardOutputTensorDims() { return standardOutputTensorDims; }
	virtual ~OutputLayer() = default;
};

inline string OutputLayer::GetcostFunctionInfo()
{
	string outputShape = "";
	for (int i = 0; i < standardOutputTensorDims.size() - 1; i++)
	{
		outputShape += to_string(standardOutputTensorDims[i]);
		outputShape += "x";
	}
	outputShape += to_string(standardOutputTensorDims[standardOutputTensorDims.size() - 1]);

	return costFunctionName + " with output shape : " + outputShape;
}
#endif // !OUTPUTLAYER_H
