#ifndef TESTDATA_H
#define TESTDATA_H
#pragma once

#include "Utils.h"
#include "HiddenMarkovModel.h"
#include <tuple>
#include <fstream>

using namespace std;

class TestData
{
public:
    // Data triples as (time, state, symbol_emitted)
    vector<tuple<size_t, size_t, size_t>> m_TimeStateSymbol;

    // brief Read experiment data from the stream						
    void ReadTestData(const HiddenMarkovModel& model, istream& dataSource);
    void ReadTestData(const HiddenMarkovModel& model, const string& dataPath, const string& delimiter=" ");
};
#endif //!TESTDATA_H