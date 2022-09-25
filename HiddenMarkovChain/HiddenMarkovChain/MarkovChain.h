#ifndef MARKOVCHAIN_H
#define MARKOVCHAIN_H
#pragma once

#include "MarkovUtils.h"
#include "Utils.h"
#include <random>

struct State
{
	map<string, int> m_SympolFrequancy;
	int m_Frequancy;
};

class MarkovChain
{
private:
	map<Pair, int> m_TransitionMatrix;
	map<int, Pair> m_IndexPairMap; // index pair map
	bool m_IsComputed;
	int m_Order; // the order of the markiv chain 1-order, 2order etc
	vector<double> m_Propabilities;
	int m_Increment;

public:	
	MarkovChain(int order);

	/**
	* @brief Return the transition probability between two states
	**/
	double TransitionProbability(const NextState& nextState, const Ngram& currentState);
	/**
	* @brief add a new string to the chain
	**/
	void Add(string& s);
	void AddAll(vector<string>& lines);

	/**
	* @brief create a new pair according to the order
	**/
	vector<Pair> MakePairs(vector<string>& v, int order);

	/**
	* @brief generates new text based on an intial seed
	**/
	string GenerateWord(int length);

	/**
	* @brief generates new text based on an intial seed
	**/
	void StoreProbabilites();

	vector<string> BuildStateSequance(string filePath, string delimiter, map<string, State>& sequenceSymbol, map<string, size_t>& symbolIndex);

	void SaveStateSequance(string modelPath, const map<string, State>& sequenceSymbol, const map<string, size_t> symbolIndex);
};

#endif // !MARKOVCHAIN_H
