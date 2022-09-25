#include "HiddenMarkovModel.h"

void HiddenMarkovModel::ReadModel(istream& modelSource)
{
    // section: states reading
    size_t nStates;
    string stateName;

    modelSource >> nStates;

    if (nStates < 2)
        throw domain_error("There must be at least two states: begin and end");

    for (size_t i = 0; i < nStates; i++)
    {
        modelSource >> stateName;
        m_StateNameIndex[stateName] = i;
        m_StateIndexName.push_back(stateName);
    }

    //section alphabet reading
    modelSource >> m_AlphabetSize;

    // section: transitions reading
    size_t nTransitions;
    string targetStateName;

    m_TransitionProb.assign(nStates, vector<double>(nStates, 0));
    modelSource >> nTransitions;

    for (size_t i = 0; i < nTransitions; i++)
    {
        double prob;
        modelSource >> stateName >> targetStateName >> prob;

        size_t fromInd = m_StateNameIndex[stateName];
        size_t toInd = m_StateNameIndex[targetStateName];

        if (fromInd + 1 == nStates)
            throw domain_error("Transition from the ending state is forbidden");

        if (toInd == 0)
            throw domain_error("Transition to the starting state is forbidden");

        m_TransitionProb[fromInd][toInd] = prob;
    }

    // section: state-symbol emission probabilities reading
    size_t neMissions;
    string symbol; // supposed to be single character, string is used for simpler reading code

    m_StateSymbolProb.assign(nStates, vector<double>(m_AlphabetSize, 0));
    modelSource >> neMissions;

    for (size_t i = 0; i < neMissions; i++)
    {
        double prob;
        modelSource >> stateName >> symbol >> prob;

        size_t stateInd = m_StateNameIndex[stateName];
        size_t symbolInd = Utils::SymbolToInd(m_SymbolIndex, symbol);

        if (stateInd == 0 || stateInd + 1 == nStates)
            throw domain_error("Symbol emission from the begining or the ending states is forbidden");

        m_StateSymbolProb[stateInd][symbolInd] = prob;

    }
}