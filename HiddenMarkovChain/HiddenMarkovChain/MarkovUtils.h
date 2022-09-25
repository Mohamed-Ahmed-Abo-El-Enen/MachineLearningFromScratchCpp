#ifndef MARKOV_UTILS_H
#define MARKOV_UTILS_H
#pragma once

#include <string>
#include <map>
#include <vector>

using namespace std;

#define START_TOKEN "<start>"
#define END_TOKEN "<eos>"

using Ngram = vector<string>;
using NextState = string;
using Pair = pair<Ngram, NextState>;

struct Occurence
{
	int occurence;
	Occurence()
	{
		occurence = 1;
	}

	void operator ++()
	{
		++occurence;
	}

	void operator=(int i)
	{
		occurence = i;
	}

	operator int()
	{
		return occurence;
	}
};

namespace MarkovUtils
{
	// take two ngram and tell whether they are equal or not
	bool compare(const Ngram& leftSide, const Ngram& rightSide);

	// join the ngram vector into string
	string join(const Ngram& n);
};
#endif // !MARKOVUTILS_H