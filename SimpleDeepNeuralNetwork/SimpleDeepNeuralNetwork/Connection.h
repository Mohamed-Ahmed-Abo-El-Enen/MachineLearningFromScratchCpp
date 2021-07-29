#pragma once

struct Connection
{
	double weight;
	double deltaWeight;

	Connection()
	{
		weight = 0;
		deltaWeight = 0;
	}

	Connection(double m_weight, double m_deltaWeight)
	{
		weight = m_weight;
		deltaWeight = m_deltaWeight;
	}
};

