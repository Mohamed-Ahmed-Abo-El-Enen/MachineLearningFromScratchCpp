#ifndef EVALUATIONMETRICS_H
#define EVALUATIONMETRICS_H
#pragma once

// State prediction estimation results for different hidden markov models algorithms
struct EvaluationMetrics
{
	size_t m_TruePositives;
	size_t m_FalsePositives;
	size_t m_TrueNegative;
	size_t m_FalseNegative;

	double m_Precision;
	double m_Recall;
	double m_F1_Score;
};
#endif // !EVALUATIONMETRICS_H