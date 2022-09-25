#include "MarkovUtils.h"

bool MarkovUtils::compare(const Ngram& leftSide, const Ngram& rightSide)
{
	bool areEqual = true;
	if (leftSide.size() != rightSide.size())
		return false;

	for (size_t i = 0; i < rightSide.size(); i++)
	{
		if (leftSide.at(i) != rightSide.at(i))
		{
			areEqual = false;
			break;
		}
	}
	return areEqual;
}

string MarkovUtils::join(const Ngram& ngram)
{
	string res;
	for (const string& str:ngram)	
		res += str + " ";
	
	return res;
}