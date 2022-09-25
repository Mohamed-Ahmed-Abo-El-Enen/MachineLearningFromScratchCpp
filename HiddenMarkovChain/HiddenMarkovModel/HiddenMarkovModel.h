#ifndef HIDDENMARKOVMODEL_H
#define HIDDENMARKOVMODEL_H
#pragma once

#include <string>
#include <iostream>
#include "Utils.h"
#include <fstream>

using namespace std;

class HiddenMarkovModel
{
private:
	// number of different emission symbols
	size_t m_AlphabetSize;
public:
	// inverse conversion
	vector<string> m_StateIndexName;

	// element[i][j] here is the probability to emit symbol j from state i
	vector<vector<double>> m_StateSymbolProb;

	// element[i][j] here is the probability of transition from state i to j
	vector<vector<double>> m_TransitionProb;

	map<string, size_t> m_SymbolIndex;

	// conversion of state name string to state index
	map<string, size_t> m_StateNameIndex;

	// brief Read model description from the stream			
	void ReadModel(istream& modelSource);
};
#endif // !HIDDENMARKOVMODELN_H