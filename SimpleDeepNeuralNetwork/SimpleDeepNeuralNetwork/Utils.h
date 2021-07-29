#pragma once
#include<string>
#include<iostream>
#include <vector>

using namespace std;

void ShowVectorVals(string label, vector<double>& v)
{
	cout << label << " ";
	for (unsigned i = 0; i < v.size(); ++i) cout << v[i] << " ";
	cout << endl;
}
