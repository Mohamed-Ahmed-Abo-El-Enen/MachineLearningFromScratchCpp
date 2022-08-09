#include "CNN.h"

CNN::CNN(Shape inputDim, Shape kernelSize, Shape poolSize, size_t hiddenLayerNodes, size_t outputDim)
{
	/*	Initialize the weight matrices and kernel matrix
	 *	Since there is only one hidden_layer, there are two weight matrices -
	 *	One between Pool layer (flattened output of pool) and one that maps
	 *	Hidden layer to outputs.
	 *	Also there is only one convolution layer, so only one kernel.
	 *	There is no learning with pool layer, therefore no weights associated
	 *	with them.
	 */

	assert(inputDim.rows > kernelSize.rows&& inputDim.columns > kernelSize.columns);
	assert(inputDim.rows - kernelSize.rows + 1> poolSize.rows && inputDim.columns - kernelSize.columns + 1> poolSize.columns);

	kernel = make_unique<Matrix>(kernelSize.rows, kernelSize.columns, true);

	poolWindow = poolSize;

	size_t x = ((inputDim.rows - kernelSize.rows + 1) / poolSize.rows);
	size_t y = ((inputDim.columns - kernelSize.columns + 1) / poolSize.columns);

	unique_ptr<Matrix> W0(new Matrix((x * y) + 1, hiddenLayerNodes, true));
	this->weights.emplace_back(move(W0));

	unique_ptr<Matrix> W1(new Matrix(hiddenLayerNodes+1, outputDim, true));
	this->weights.emplace_back(move(W1));

} 

unique_ptr<vector<double>> CNN::MaxPooling(unique_ptr<Matrix> &conv, vector<unique_ptr<Matrix>>& convLayersMatrix)
{
	size_t x = (conv->GetRows() / poolWindow.rows);
	size_t y = (conv->GetColumns() / poolWindow.columns);

	unique_ptr<Matrix> pool = make_unique<Matrix>(conv->GetRows(), conv->GetColumns(), false);
	unique_ptr<vector<double>> poolFlatten = make_unique<vector<double>>();

	size_t xPtr = 0;
	size_t yPtr = 0;

	auto maxIndex = make_unique<Shape>(Shape{ 0,0 });
	for (size_t i = 0; i < x; i++)
	{
		xPtr = (i * poolWindow.rows);
		for (size_t j = 0; j < y; j++)
		{
			yPtr = (j * poolWindow.columns);
			double max = np::maximum(conv, xPtr, yPtr, poolWindow, maxIndex);
			poolFlatten->push_back(max);
			pool->set(maxIndex->rows, maxIndex->columns, 1);
		}
	}
	convLayersMatrix[0] = move(pool);
	// append 1s to inputs and to output of every layer (for bias)
	poolFlatten->push_back(1);

	return poolFlatten;
}

void CNN::ForwardPropagate(unique_ptr<Matrix>& input, vector<unique_ptr<Matrix>>& convLayersMatrix, vector<unique_ptr<vector<double>>>& activations)
{
	/*Forward propagate the provided inputs through the Convolution Neural Network
	 *	and the outputs of each Dense layer is appended as vector to activations.
	 *	Output of convolution layer(matrix) is appended to conv_activations
	*/
	assert(weights.size() == 2); // conv->maxpolling, flatten->hidden, hidden->output
	unique_ptr<Matrix> conv = make_unique<Matrix>(input->GetRows() - kernel->GetRows() + 1, input->GetColumns() - kernel->GetColumns() + 1, true);

	for (size_t i = 0; i < conv->GetRows(); i++)
		for (size_t j = 0; j < conv->GetColumns(); j++)
			conv->set(i, j, np::multiply(kernel, input, i, j));	
	conv = np::apply(conv, fns::relu);

	unique_ptr<vector<double>> poolFlatten = MaxPooling(conv, convLayersMatrix);

	// Hidden Layer
	unique_ptr<Matrix> W0 = np::transpose(weights[0]);
	unique_ptr<vector<double>> hidden = np::dot(W0, poolFlatten);
	hidden = np::apply(hidden, fns::relu);
	hidden->push_back(1);
	activations[0] = move(poolFlatten);

	//output layer
	unique_ptr<Matrix> W1 = np::transpose(weights[1]);
	unique_ptr<vector<double>> output = np::dot(W1, hidden);
	output = np::apply(output, fns::softmax);
	output = np::normalize(output);

	activations[1] = move(hidden);
	activations[2] = move(output);
}

double CNN::CrossEntropy(unique_ptr<vector<double>>& yHat, unique_ptr<vector<double>>& yTrue)
{
	assert(yHat->size() == yTrue->size());
	unique_ptr<vector<double>> z = np::apply(yHat, log);
	z = np::multiply(z, yTrue);
	double error = np::sum(z);
	return (-error);
}

void CNN::BackPropagate(unique_ptr<vector<double>>& deltaL, vector<unique_ptr<Matrix>>& convLayersMatrix, vector<unique_ptr<vector<double>>>& activations,
	unique_ptr<Matrix>& input, double (*activateFuncDeriv)(double), double learningRate)
{	
	//Compute deltas of each layer and return the same.
	//*	delta_L: delta of the final layer, computed and passed as argument
	//*	activations: Output of each layer after applying activation function, assume that all layers have same activation function except that of final layer.
	//*	active_fn_der: function pointer for the derivative of activation function, which takes activation of the layer as input 
	unique_ptr<vector<double> > deltaH = np::dot(weights[1], deltaL);
	unique_ptr<vector<double> > active = np::apply(activations[1], activateFuncDeriv);
	deltaH = np::multiply(deltaH, active);

	unique_ptr<vector<double> > delta_x = np::dot(weights[0], deltaH, 1); // don't compute last layer
	active = np::apply(activations[0], activateFuncDeriv);
	delta_x = np::multiply(delta_x, active);

	unique_ptr<Matrix> delta_conv =
		make_unique<Matrix>(convLayersMatrix[0]->GetRows(), convLayersMatrix[0]->GetColumns(), false);

	unsigned int counter = 0;
	for (unsigned int r = 0; r < convLayersMatrix[0]->GetRows(); r++) 
	{
		for (unsigned int c = 0; c < convLayersMatrix[0]->GetColumns(); c++)
		{
			if (convLayersMatrix[0]->get(r, c) == 1.0) 
			{
				delta_conv->set(r, c, delta_x->at(counter));
				counter++;
			}
		}
	}

	// update weights
	unique_ptr<Matrix> dW0 = np::dot(activations[0], deltaH, 1); // last column has to be sliced off	
	unique_ptr<Matrix> dW1 = np::dot(activations[1], deltaL);
	dW0 = np::multiply(dW0, (learningRate));
	dW1 = np::multiply(dW1, (learningRate));

	weights[0] = np::subtract(weights[0], dW0);
	weights[1] = np::subtract(weights[1], dW1);

	for (unsigned int i = 0; i < kernel->GetRows(); i++) 
		for (unsigned int j = 0; j < kernel->GetColumns(); j++) 
			kernel->set(i, j, np::multiply(delta_conv, input, i, j));		
}

void CNN::train(vector<unique_ptr<Matrix>>& xTrain, vector<unique_ptr<vector<double>>>& yTrain, double learningRate, size_t epochs)
{
	/*	Train the Neural Network aka change weights such that error between
	* the forward propagated Xtrain and Ytrain is reduced.
	* Break (Xtrain, ytrain) into batches of size batch_size and run 3 following
	* methods for all batches:
	*		1) forward_propagate
	*		2) error calculation: cross_entropy
	*		3) back_propagate
	* 		4) update_weights
	* Do this procedure 'epochs' number of times
	*/

	assert(xTrain.size() == yTrain.size());
	size_t e = 1;
	while (e <= epochs)
	{
		// Split (Xtrain, Ytrain) into batches	
		// This is expensive because it is making sub vectors by copying
		// Have to think of a better way

		size_t it = 0;
		double error = 0;
		while (it < xTrain.size())
		{
			vector<unique_ptr<Matrix>> convLayersMatrix(2);
			vector<unique_ptr<vector<double>>> activations(3);

			ForwardPropagate(xTrain[it], convLayersMatrix, activations);
			error += CrossEntropy(activations.back(), yTrain[it]);

			unique_ptr<vector<double>> deltaL = np::subtract(activations.back(), yTrain[it]);
			BackPropagate(deltaL, convLayersMatrix, activations, xTrain[it], fns::relu_gradient, learningRate);
			it += 1;
		}
		cout << "Epoch: " << e << " Error: " << (error / xTrain.size()) << endl;
		e += 1;
	}
}

double CNN::validate(vector<unique_ptr<Matrix>>& xVal, vector<unique_ptr<vector<double>>>& yVal)
{
	/*	Calculate the Validation error over the validation set.
	 *	So only do forward_propagate for each batch without updating weights
	 *	each iteration
	*/

	assert(xVal.size() == yVal.size());
	size_t it = 1;
	double error = 0;
	while (it <= xVal.size())
	{
		vector<unique_ptr<Matrix>> convLayerMatrix(2);
		vector<unique_ptr<vector<double>>> activations(3);
		ForwardPropagate(xVal[it], convLayerMatrix, activations);
		error += CrossEntropy(activations.back(), yVal[it]);
		it += 1;
 	}
	cout << "Error: " << (error / xVal.size()) << endl;
	return (error / xVal.size());
}

void CNN::info()
{
	cout << "Kernel size: (" << kernel->GetRows() << ", " << kernel->GetColumns() << ")" << endl;
	for (size_t i = 0; i < weights.size(); i++)
	{
		cout << "Weight" << i << " size: (" << weights[i]->GetRows() << ", " << weights[i]->GetColumns() << ")" << endl;
	}
}
