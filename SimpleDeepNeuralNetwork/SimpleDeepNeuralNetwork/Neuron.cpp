#include"Neuron.h"

Neuron::Neuron(unsigned numOutputs, unsigned neuronIndex, ActivationFunctions activationFunctions, double learningRate, double alphaMomentum)
{
	for (unsigned i = 0; i < numOutputs; i++)
	{
		m_OutputWeights.push_back(Connection());
		m_OutputWeights.back().weight = RandomWeight();
		//m_outputtVal = RandomWeight();
	}
	m_neuronIndex = neuronIndex;
	m_ActivationFunction = activationFunctions;
	m_LearningRate = learningRate;
	m_AlphaMomentum = alphaMomentum;
}

Neuron::~Neuron()
{
}

double Neuron::RandomWeight()
{
	return rand() / double(RAND_MAX);
}

void Neuron::SetOutputVal(double val)
{
	m_outputtVal = val;
}

double Neuron::GetOutputVal() const
{
	return m_outputtVal;
}

void Neuron::FeedForward(const vector<Neuron>& prevLayer)
{
	double sum = 0.0;

	// Sum the prevoius layer's output (which our inputs in neuron)
	// Include the bias node from the prevoius layer

	for (unsigned n = 0; n < prevLayer.size(); n++)
	{
		sum += prevLayer[n].GetOutputVal() * prevLayer[n].m_OutputWeights[m_neuronIndex].weight;
	}
	m_outputtVal = ActivationFunction(sum);
}

void Neuron::SetActivationFunction(ActivationFunctions ActivationFunction)
{ 
	m_ActivationFunction = ActivationFunction; 
}

double Neuron::ActivationFunction(double x)
{
	switch (m_ActivationFunction)
	{
	case TanhTag:
		return Tanh(x);
	default:
		return Sigmoid(x);
	}
}

double Neuron::ActivationFunctionDerivative(double x)
{
	switch (m_ActivationFunction)
	{
	case TanhTag:
		return TanhDrv(x);
	default:
		return SigmoidDrv(x);
	}
}

void Neuron::CalcualteOutputGradient(double targetVal)
{
	double delta = targetVal - m_outputtVal;
	m_Gradient = delta * ActivationFunctionDerivative(m_outputtVal);
}

double Neuron::SumDOW(const vector<Neuron>& nextLayer) const
{
	double sum = 0;
	// Sum our contributions of the errors at the nodes we feed
	for (unsigned n = 0; n < nextLayer.size()-1; n++)
	{
		sum += m_OutputWeights[n].weight * nextLayer[n].m_Gradient;
	}
	return sum;
}


void Neuron::CalculateHiddenLayerGradient(const vector<Neuron>& nextLayer)
{
	double dow = SumDOW(nextLayer);
	m_Gradient = dow * ActivationFunctionDerivative(m_outputtVal);
}

void Neuron::UpdateInputWeights(vector<Neuron>& prevLayer)
{
	// The weight to be updated are in the connection container
	// in the neurons in the preceding layer

	for (unsigned n = 0; n < prevLayer.size(); n++)
	{
		Neuron& neuron = prevLayer[n];
		double oldDeltaWeight = neuron.m_OutputWeights[m_neuronIndex].deltaWeight;

		//Individual input, magnigfied by the gradient and train rate
		// Also add momentum fration of the previous delta weight
		double newDeltaWeight = m_LearningRate * neuron.GetOutputVal() * m_Gradient + m_AlphaMomentum * oldDeltaWeight;

		neuron.m_OutputWeights[m_neuronIndex].deltaWeight = newDeltaWeight;
		neuron.m_OutputWeights[m_neuronIndex].weight += newDeltaWeight;
	}	
}