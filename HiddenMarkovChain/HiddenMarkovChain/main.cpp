#include "Utils.h"
#include "MarkovChain.h"

void Test0()
{
	vector<string> lines;
	lines.push_back("This is a test case");
	MarkovChain chain(1);

	chain.AddAll(lines);

	Ngram currentState = { "test" };
	NextState nextState("case");
	cout << chain.TransitionProbability(nextState, currentState) << "\n";
}

void Test1()
{
	string filePath = "../Dataset/CornellMovieDialogCorpus.txt";
	vector<string> lines = Utils::ReadFile(filePath);
	MarkovChain chain(2);

	chain.AddAll(lines);

	Ngram currentState = { "listen", "to" };
	NextState nextState("this");
	cout << chain.TransitionProbability(nextState, currentState) << "\n";
}

void Test2()
{
	string filePath = "../Dataset/CornellMovieDialogCorpus.txt";
	vector<string> lines = Utils::ReadFile(filePath);
	MarkovChain chain(2);

	chain.AddAll(lines);

	cout << chain.GenerateWord(10) << "\n";
}

void Test3()
{
	map<string, State> sequenceSymbol;
	map<string, size_t> symbolIndex;
	MarkovChain chain(1);
	vector<string> lines = chain.BuildStateSequance("../Dataset/conll2000_test.txt", " ", sequenceSymbol, symbolIndex);

	chain.AddAll(lines);

	chain.SaveStateSequance("../Model/conll2000_test.model", sequenceSymbol, symbolIndex);
}

int main()
{
	//Test0();
	//Test1();
	//Test2();
	Test3();
	return 0;
}
