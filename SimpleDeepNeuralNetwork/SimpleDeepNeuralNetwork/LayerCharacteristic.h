#pragma once
#include "Activations.h"

class LayerCharacteristic
{
private:
	unsigned m_layerNumNeurons;
	ActivationFunctions m_activationFunction;
	double m_LearningRate;
	double m_AlphaMomentum;

public:
	LayerCharacteristic(int layerNumNeurons, ActivationFunctions activationFunction=ActivationFunctions::SigmoidTag, double learningRate=0.1, double alphaMomentum=0.9)
	{
		m_layerNumNeurons = layerNumNeurons;
		m_activationFunction = activationFunction;
		m_LearningRate = learningRate;
		m_AlphaMomentum = alphaMomentum;
	}

	unsigned GetLayerNumNeurons() const { return m_layerNumNeurons; }
	ActivationFunctions GetActivationFunction() const { return m_activationFunction; }
	double GetLearningRate() const { return m_LearningRate; }
	double GetAlphaMomentum() const { return m_AlphaMomentum; }
};

