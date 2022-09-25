#include "HHMAlgorithm.h"

// calculate new state probability for the Viterbi algorithm step
double HHMAlgorithm::CalculateNewStateProbability(size_t stepNumber, size_t prevState, size_t curState, size_t curSymbol, const HiddenMarkovModel& model, const vector<vector<double>>& sequenceProbability)
{
    double prevProbability = 1.;
    if (stepNumber == 0 && prevState == 0)
        prevProbability = 1.0;
    else if (stepNumber == 0 && prevState != 0)
        prevProbability = 0.0;
    else
        prevProbability = sequenceProbability[stepNumber - 1][prevState];

    return (prevProbability * model.m_TransitionProb[prevState][curState] * model.m_StateSymbolProb[curState][curSymbol]);
}

// find the best previous state during the Viterbi algorithm step
size_t HHMAlgorithm::FindBestTransitionSource(size_t stepNumber, size_t curState, size_t curSymbol, const HiddenMarkovModel& model, const vector<vector<double>>& sequenceProbability)
{
    if (stepNumber == 0)
        return 0;

    size_t nStates = model.m_TransitionProb.size();
    double bestProbValue = -1;
    size_t bestPrevState = HMM_UNDEFINED_STATE;

    for (size_t prevState = 0; prevState < nStates; prevState++)
    {
        double curProb = CalculateNewStateProbability(stepNumber, prevState, curState, curSymbol, model, sequenceProbability);

        if (curProb > bestProbValue)
        {
            bestProbValue = curProb;
            bestPrevState = prevState;
        }
    }

    // there must be at least two states => the result won't be undefined
    return bestPrevState;
}


vector<size_t> HHMAlgorithm::FindMostProbableStateSequence(const HiddenMarkovModel& model, const TestData& data)
{
    // section: prepare and initialize data structures for calculations
    size_t nStates = model.m_TransitionProb.size();
    size_t maxtime = data.m_TimeStateSymbol.size();

    // sequenceProbability[i][j] is the probability of the most probable sequence of states    
    vector<vector<double>> sequenceProbability(maxtime, vector<double>(nStates, 0));

    // prevSeqState[i][j] is the previous state from which the most probable
    vector<vector<size_t>> prevSeqState(maxtime, vector<size_t>(nStates, HMM_UNDEFINED_STATE));

    // section: calculate probabilities for Viterbi algorithm using dynamic programming approach
    for (size_t t = 0; t < maxtime; t++)
    {
        for (size_t curState = 0; curState < nStates; curState++)
        {
            size_t curSymbol = get<2>(data.m_TimeStateSymbol[t]);
            size_t bestPrevState = FindBestTransitionSource(t, curState, curSymbol, model, sequenceProbability);
            double bestProbValue = CalculateNewStateProbability(t, bestPrevState, curState, curSymbol, model, sequenceProbability);

            sequenceProbability[t][curState] = bestProbValue;
            prevSeqState[t][curState] = bestPrevState;
        }
    }

    // section: collect most probable sequence in the reverse order
    vector<size_t> mostProbableSeq;
    ptrdiff_t curStep = maxtime - 1;

    // find the last state of the most probable sequence to start recovery from it
    size_t curState = distance(begin(sequenceProbability[curStep]), max_element(begin(sequenceProbability[curStep]), end(sequenceProbability[curStep])));

    for (; curStep > 0; --curStep)
    {
        curState = prevSeqState[curStep][curState];
        mostProbableSeq.push_back(curState);
    }

    mostProbableSeq.push_back(curState);

    // section: restore correct order and return results
    reverse(begin(mostProbableSeq), end(mostProbableSeq));

    return move(mostProbableSeq);
}

// get the cumulative forward step transition probability
double HHMAlgorithm::CalcualteForwardStepProbability(size_t stepNumber, size_t curState, const HiddenMarkovModel& model, const TestData& data, const vector<vector<double>>& forwardStateProbability)
{
    size_t nStates = model.m_TransitionProb.size();
    size_t curSymbol = get<2>(data.m_TimeStateSymbol[stepNumber]);

    if (stepNumber == 0)
        return model.m_TransitionProb[0][curState] * model.m_StateSymbolProb[curState][curSymbol];
    else
    {
        double prevCumlativeProb = 0;
        for (size_t prevState = 0; prevState < nStates; prevState++)
            prevCumlativeProb += (forwardStateProbability[stepNumber - 1][prevState] * model.m_TransitionProb[prevState][curState]);

        return prevCumlativeProb * model.m_StateSymbolProb[curState][curSymbol];
    }
}

// get the cumulative probability for the backward step
double HHMAlgorithm::CalcualteBackwardStepProbability(size_t stepNumber, size_t curState, const HiddenMarkovModel& model, const TestData& data, const vector<vector<double>>& backwardStateProbability)
{
    size_t nStates = model.m_TransitionProb.size();
    size_t maxtime = data.m_TimeStateSymbol.size();

    if (stepNumber + 1 == maxtime)
        return 1.0; // probability to describe empty sequence is 1.
    else
    {
        size_t nextSymbol = get<2>(data.m_TimeStateSymbol[stepNumber + 1]);
        double nextCumulativeProb = 0;

        for (size_t nextState = 0; nextState < nStates; nextState++)
            nextCumulativeProb += (model.m_TransitionProb[curState][nextState] * model.m_StateSymbolProb[nextState][nextSymbol] * backwardStateProbability[stepNumber + 1][nextState]);

        return nextCumulativeProb;
    }
}

vector<vector<pair<double, double>>> HHMAlgorithm::CalcualteForwardBackwardProbabilites(const HiddenMarkovModel& model, const TestData& data)
{
    size_t nStates = model.m_TransitionProb.size();
    size_t maxtime = data.m_TimeStateSymbol.size();


    // forwardStateProbability[i][j] is the probability that any hidden sequence (with
    // the hidden state at i-th step equal to j) describes first 1..i observations.    
    vector<vector<double>> forwardStateProbability(maxtime, vector<double>(nStates, 0));

    // section: calculate forward probabilities of the forward-backward algorithm
    for (size_t i = 0; i < maxtime; i++)
    {
        for (size_t curState = 0; curState < nStates; curState++)
        {
            double cumulativePrevProbability = CalcualteForwardStepProbability(i, curState, model, data, forwardStateProbability);
            forwardStateProbability[i][curState] = cumulativePrevProbability;
        }
    }

    // backwardStateProbability[i][j] is the probability that any hidden sequence (with
    // the hidden state at i+1 step equal to j) describes last i+1..T observations.
    vector<vector<double>> backwardStateProbability(maxtime, vector<double>(nStates));

    // section: calculate backward probabilities of the forward-backward algorithm
    for (ptrdiff_t t = maxtime - 1; t >= 0; t--)
    {
        for (size_t curState = 0; curState < nStates; curState++)
        {
            double cumulativeNextProbability = CalcualteBackwardStepProbability(t, curState, model, data, backwardStateProbability);
            backwardStateProbability[t][curState] = cumulativeNextProbability;
        }
    }

    // section: return joined results
    vector<vector<pair<double, double>>> forwardBackwardProbability(maxtime, vector<pair<double, double>>(nStates, pair<double, double>()));

    for (size_t i = 0; i < maxtime; i++)
    {
        for (size_t curState = 0; curState < nStates; curState++)
        {
            forwardBackwardProbability[i][curState] = pair<double, double>(forwardStateProbability[i][curState], backwardStateProbability[i][curState]);
        }
    }
    return move(forwardBackwardProbability);
}