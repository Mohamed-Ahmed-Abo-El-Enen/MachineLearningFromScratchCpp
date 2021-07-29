#include"Net.h"

Net::Net(const vector<LayerCharacteristic>& layers)
{
	unsigned numLayers = layers.size();
	for (unsigned layersNum = 0; layersNum < numLayers; layersNum++)
	{
		m_layers.push_back(Layer());
		cout << "Made a A Layer of size:"<< layers[layersNum].GetLayerNumNeurons() << endl;

		// We have made a new layer, 
		// Now fill it with neurons and add a bias neuron to the layer

		unsigned numOutputs = layersNum == layers.size() - 1 ? 0 : layers[layersNum + 1].GetLayerNumNeurons();
		for (unsigned neuroNum = 0; neuroNum <= layers[layersNum].GetLayerNumNeurons(); neuroNum++)
		{
			m_layers.back().push_back(Neuron(numOutputs, neuroNum, layers[layersNum].GetActivationFunction(), layers[layersNum].GetLearningRate(), layers[layersNum].GetAlphaMomentum()));
			cout << "Made a Neuro "<< neuroNum+1 << endl;
		}
		// force the bias node's at output layer to be 1 
		m_layers.back().back().SetOutputVal(1);
	}
	m_error = 0;
	m_ReccentAverageSmothingFactor = 100;
}

void Net::FeedFirstLayer(const vector<double>& inputVals)
{
	assert(inputVals.size() == m_layers[0].size() - 1);
	//Assign the input values into the input neurons
	for (unsigned i = 0; i < inputVals.size(); i++)
	{
		m_layers[0][i].SetOutputVal(inputVals[i]);
	}
}

// Forward propagate
void Net::ForwardPropagate()
{
	for (unsigned layerNum = 1; layerNum < m_layers.size(); layerNum++)
	{
		for (unsigned n = 0; n < m_layers[layerNum].size() - 1; n++)
		{
			m_layers[layerNum][n].FeedForward(m_layers[layerNum - 1]);
		}
	}
}

void Net::FeedForward(const vector<double>& inputVals)
{
	FeedFirstLayer(inputVals);
	ForwardPropagate();	
}

void Net::GradientFirstLayer(Layer& outputLayer, const vector<double>& targetVals)
{
	// Calculate output layer gradients
	for (unsigned n = 0; n < outputLayer.size() - 1; n++)
	{
		outputLayer[n].CalcualteOutputGradient(targetVals[n]);
	}
}

void Net::GradientHiddenLayer()
{
	// Calculate gradients of hidden layers
	for (unsigned layerNum = m_layers.size() - 2; layerNum > 0; layerNum--)
	{
		Layer& hiddenLayer = m_layers[layerNum];
		Layer& nextLayer = m_layers[layerNum + 1];

		for (unsigned n = 0; n < hiddenLayer.size(); n++)
		{
			hiddenLayer[n].CalculateHiddenLayerGradient(nextLayer);
		}
	}
}

void Net::CalculateGradient(Layer& outputLayer, const vector<double>& targetVals)
{
	GradientFirstLayer(outputLayer, targetVals);
	GradientHiddenLayer();
}

void Net::UpdateWeight()
{
	// for all layers from outputs to first hidden layer
	// update connection weights
	for (unsigned layerNum = m_layers.size() - 1; layerNum > 0; layerNum--)
	{
		Layer& currLayer = m_layers[layerNum];
		Layer& prevLayer = m_layers[layerNum - 1];

		for (unsigned n = 0; n < currLayer.size() - 1; n++)
		{
			currLayer[n].UpdateInputWeights(prevLayer);
		}
	}
}

void Net::GetLossFunctionError(const vector<double>& targetVals, vector<Neuron>& outputLayer)
{
	switch (m_LossfuntionTag)
	{
	case LossFunction::MSETag:
		m_error = LossFunction::MSE(targetVals, outputLayer);
		return;
	default:
		m_error = LossFunction::RMS(targetVals, outputLayer);
		return;
	}
}

void Net::BackPropagate(const vector<double>& targetVals)
{
	// Calculate overall net error(RMS of output neuron errors)
	vector<Neuron>& outputLayer = m_layers.back();
	GetLossFunctionError(targetVals, outputLayer);

	m_RecentAverageError = (m_RecentAverageError * m_ReccentAverageSmothingFactor + m_error) / (m_ReccentAverageSmothingFactor + 1);

	CalculateGradient(outputLayer, targetVals);

	UpdateWeight();
}

void Net::GetResults(vector<double>& resultVals) const
{
	resultVals.clear();
	for (unsigned n = 0; n < m_layers.back().size()-1; n++)
	{
		resultVals.push_back(m_layers.back()[n].GetOutputVal());
	}
}

void Net::ShowVectorVals(string label, vector<double>& v)
{
	cout << label << " ";
	for (unsigned i = 0; i < v.size(); ++i) cout << v[i] << " ";
	cout << endl;
}

void Net::Compile(LossFunction::LossFunctionTags lossFunctionTag)
{
	m_LossfuntionTag = lossFunctionTag;
}

unsigned Net::Fit(vector<vector<double>>& samplesFeatures, vector<vector<double>>& samplesTarget, int epochs, bool verbose)
{
	assert(samplesFeatures.size() == samplesTarget.size());

	for (int i = 1; i <= epochs; i++)
	{
		for (unsigned j = 0; j < samplesFeatures.size(); j++)
		{
			assert(samplesFeatures[j].size() == m_layers[0].size() - 1);
			FeedForward(samplesFeatures[j]);
			BackPropagate(samplesTarget[j]);
			if (verbose)
			{
				vector<double> predictedVals;
				GetResults(predictedVals);
				cout << "Sample #Num: " << j + 1 << endl;
				ShowVectorVals("Inputs:", samplesFeatures[j]);
				ShowVectorVals("Outputs:", predictedVals);
				ShowVectorVals("Targets:", samplesTarget[j]);
				cout << "Net recent average error: " << GetRecentAverageError() << endl;
			}
		}
		cout << "Epoch #Num: "<< i <<" With Error: "<< GetRecentAverageError() << endl;
	}
	return 0;
}

unsigned Net::Predict(vector<vector<double>>& samplesFeatures, vector<double>& predictedVals, bool printWhilePredict)
{
	for (unsigned i = 0; i < samplesFeatures.size(); i++)
	{
		assert(samplesFeatures[i].size() == m_layers[0].size() - 1);
		FeedForward(samplesFeatures[i]);
		GetResults(predictedVals);
		if (printWhilePredict)
		{
			vector<double> predictedVals;
			GetResults(predictedVals);
			ShowVectorVals("Inputs:", samplesFeatures[i]);
			ShowVectorVals("Outputs:", predictedVals);
		}
	}
	return 0;
}