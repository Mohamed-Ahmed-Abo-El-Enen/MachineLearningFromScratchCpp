#ifndef UTILS_H
#define UTILS_H
#pragma once

#include <map>
#include <string>
#include <vector>
#include <algorithm>

using namespace std;

namespace Utils
{
	size_t SymbolToInd(map<string, size_t>& m_SymbolIndex, const string& symbol);
	size_t GetSymbolInd(const map<string, size_t>& m_SymbolIndex, const string& symbol);
	string Lower(string& str);
	string RemoveNumbers(string& str);
	string RemovePunctuations(string& str);
	string trim(const string& str);
	vector<string> split(string line, string delimiter, bool removePunctuations = false);
}
#endif // !UTILS_H