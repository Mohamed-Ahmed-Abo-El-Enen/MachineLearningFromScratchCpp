#ifndef MODULE_H
#define MODULE_H

#include<string>  
#include "Tensor.h"

/*
	Interface to used for modules block
*/

class Module
{
protected:
	vector<int> standardInputTensorDims;
	vector<int> standardOutputTensorDims;
	unique_ptr<Tensor<double>> m_Input;
	unique_ptr<Tensor<double>> m_Output;
	string layerName;
	bool isEval = false;
public:
	virtual unique_ptr<Tensor<double>> ForwardPropagate(unique_ptr<Tensor<double>>& input) = 0;
	virtual unique_ptr<Tensor<double>> BackwardPropagate(unique_ptr<Tensor<double>>& chainGradient, double learningRate) = 0;
	virtual void Compile(unique_ptr<Tensor<double>>& input) = 0;
	virtual void load(FILE* fileModel) = 0;
	virtual void save(FILE* fileModel) = 0;
	virtual ~Module() = default;

	string GetLayerName() { return layerName; }
	string GetLayerInfo(unique_ptr<Tensor<double>>& output);
	vector<int>& GetStandardInputTensorDims() { return standardInputTensorDims; }
	vector<int>& GetStandardOutputTensorDims() { return standardOutputTensorDims; }
	void train();
	void eval();
};

inline string Module::GetLayerInfo(unique_ptr<Tensor<double>>& output)
{ 
	string inputShape = "";
	for (int i = 0; i < standardInputTensorDims.size()-1; i++)
	{		
		inputShape += to_string(standardInputTensorDims[i]);
		inputShape += "x";
	}
	inputShape += to_string(standardInputTensorDims[standardInputTensorDims.size() - 1]);

	string outputShape = "";
	for (int i = 0; i < output->numDims - 1; i++)
	{		
		outputShape += to_string(output->m_Dims[i]);
		outputShape += "x";
	}
	outputShape += to_string(output->m_Dims[output->numDims - 1]);

	return layerName + " with input shape : " + inputShape + " and output shape : " + outputShape;
}

inline void Module::eval()
{
	this->isEval = true;
}

inline void Module::train()
{
	this->isEval = false;
}

#endif // !MODULE_H
