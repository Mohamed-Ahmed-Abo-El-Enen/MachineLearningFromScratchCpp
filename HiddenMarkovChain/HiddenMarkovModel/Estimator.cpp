#include "Estimator.h"

vector<size_t> Estimator::GetMostProbableStates(const vector<vector<pair<double, double> > >& forwardBackwardProb)
{
    size_t maxtime = forwardBackwardProb.size();
    vector<size_t> mostProbableStates;

    for (size_t t = 0; t < maxtime; ++t) {
        size_t mostProbableState =
            distance(std::begin(forwardBackwardProb[t]),
                max_element(std::begin(forwardBackwardProb[t]),
                    std::end(forwardBackwardProb[t]),
                    [](const pair<double, double>& prev,
                        const pair<double, double>& next)
                    {return (prev.first * prev.second <
                        next.first * next.second); }));
        mostProbableStates.push_back(mostProbableState);
    }

    return std::move(mostProbableStates);
}

vector<vector<size_t>> Estimator::CombineConfusionMatrix(const TestData& realData, const vector<size_t>& predictedStates, const HiddenMarkovModel& model)
{
    size_t maxtime = predictedStates.size();
    size_t nStates = model.m_TransitionProb.size();
    vector<vector<size_t>> confusionMatrix(nStates, vector<size_t>(nStates, 0));

    for (size_t t = 0; t < maxtime; t++)
    {
        size_t predictedInd = predictedStates[t];
        size_t realInd = get<1>(realData.m_TimeStateSymbol[t]);

        confusionMatrix[predictedInd][realInd]++;
    }

    return move(confusionMatrix);
}

vector<EvaluationMetrics> Estimator::GetStatePredicionEstimations(const vector<vector<size_t>>& confusionMatrix)
{
    size_t nStates = confusionMatrix.size();
    vector<EvaluationMetrics> estimations(nStates);
    vector<size_t> colSums(nStates, 0);
    vector<size_t> rowSums(nStates, 0);

    // section: prepare auxiliary columns sums and row sums for further usage
    for (size_t i = 0; i < nStates; i++)
    {
        for (size_t j = 0; j < nStates; j++)
        {
            rowSums[i] += confusionMatrix[i][j];
            colSums[i] += confusionMatrix[j][i];
        }
    }

    size_t totalObesrvations = accumulate(begin(rowSums), end(rowSums), 0UL);

    // section: calculate prediction estimations for each state
    for (size_t state = 0; state < nStates; state++)
    {
        estimations[state].m_TruePositives = confusionMatrix[state][state];
        estimations[state].m_FalsePositives = rowSums[state] - confusionMatrix[state][state];

        // neither predicted to be current state nor its real state is the current one
        estimations[state].m_TrueNegative = totalObesrvations - rowSums[state] - colSums[state] + confusionMatrix[state][state];
        estimations[state].m_FalseNegative = colSums[state] - confusionMatrix[state][state];

        // calculate F1-Scoure
        if (rowSums[state] != 0)
            estimations[state].m_Precision = static_cast<double> (confusionMatrix[state][state]) / static_cast<double>(rowSums[state]);

        if (colSums[state] != 0)
            estimations[state].m_Recall = static_cast<double> (confusionMatrix[state][state]) / static_cast<double> (colSums[state]);

        if (rowSums[state] == 0 && colSums[state])
            estimations[state].m_F1_Score = 0;
        else
            estimations[state].m_F1_Score = 2.0 * (estimations[state].m_Precision * estimations[state].m_Recall) / (estimations[state].m_Precision + estimations[state].m_Recall);
    }
    return move(estimations);
}

void Estimator::PrintPredictionsEstimations(size_t stateInd, const EvaluationMetrics& estimation, const HiddenMarkovModel& model)
{
    cout << "Estimations Report for State: " << model.m_StateIndexName[stateInd] << endl;
    cout << "\tTrue Class/PredictedClass\t\n";
    cout << "True Positives= " << estimation.m_TruePositives << '\t' << "False Positives= " << estimation.m_FalsePositives << endl;
    cout << "True Negatives= " << estimation.m_TrueNegative << '\t' << "False Negatives= " << estimation.m_FalseNegative << endl;
    cout << "Precision= " << estimation.m_Precision << endl;
    cout << "Recall= " << estimation.m_Recall << endl;
	cout << "F1-Score= " << estimation.m_F1_Score << endl;
    cout << "\n";
}
