#ifndef UTILS_H
#define UTILS_H
#pragma once

#include <vector>
#include <iostream>
#include <string>
#include <fstream>
#include <algorithm>

using namespace std;

namespace Utils
{
    vector<string> ReadFile(string filePath, string delimiter="+++$+++");
    string Lower(string& str);
    string RemoveNumbers(string& str);
    string trim(const string& str);
    string RemovePunctuations(string& str);
    vector<string> split(string line, string delimiter, bool removePunctuations = false);
}
#endif // !UTILS_H

