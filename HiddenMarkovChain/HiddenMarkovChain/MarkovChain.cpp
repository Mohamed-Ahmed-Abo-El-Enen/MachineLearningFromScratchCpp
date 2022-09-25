#include "MarkovChain.h"

MarkovChain::MarkovChain(int order): m_Order(order), m_IsComputed(false), m_Increment(1){}

void MarkovChain::Add(string& str)
{
	string startTokens = string(START_TOKEN);
	string endTokens = string(END_TOKEN);

	str = Utils::RemovePunctuations(str);

	str = startTokens + " " + str + " " + endTokens;

	// Extract N-Gram from the given string
	Ngram strVec = Utils::split(str, " ");

	vector<Pair> pairs = MakePairs(strVec, m_Order);

	for (const Pair& nPair : pairs)
	{
		if (m_TransitionMatrix.count(nPair))
			m_TransitionMatrix[nPair]++;
		else
		{
			m_TransitionMatrix.insert({ nPair, 1 });
			m_IndexPairMap.insert({ m_Increment++, nPair });
		}
	}
}

void MarkovChain::AddAll(vector<string>& lines)
{
	for (string& line:lines)
		Add(line);	
}

vector<Pair> MarkovChain::MakePairs(vector<string>& strVec, int order)
{
	vector<Pair> pairList;

	for (size_t i = 0; i < strVec.size()-order; i++)
	{
		Ngram currState;
		NextState nextState;
		nextState = strVec.at(i + order);
		for (size_t j = i; j < i + order; j++)
			currState.push_back(strVec.at(j));		
		
		pairList.push_back({ currState, nextState });
	}
	return pairList;
}

double MarkovChain::TransitionProbability(const NextState& nextState, const Ngram& currentState)
{
	if (currentState.size() != m_Order)
		throw "State size not equal order";

	int frequencyOfNg = 0;
	int sumOther = 0;

	Pair pairToLookFor{ currentState, nextState };
	if (m_TransitionMatrix.count(pairToLookFor))
	{
		frequencyOfNg = m_TransitionMatrix[pairToLookFor];
		// Divide by the currentState to any other

		for (const auto& kv : m_TransitionMatrix) 
		{
			if (MarkovUtils::compare(kv.first.first, currentState))
				sumOther += kv.second;
		}
	}
	else
	{
		cout << "The tranisition was not found " << pairToLookFor.first.back() << " " << pairToLookFor.second << "\n";
		return -1;
	}

	return ((double)frequencyOfNg / (double)sumOther);
}

void MarkovChain::StoreProbabilites()
{
	if (!m_IsComputed)
	{
		for (const auto& kv : m_TransitionMatrix)
		{
			auto prob = TransitionProbability(kv.first.second, kv.first.first);
			m_Propabilities.push_back(prob);
		}

		m_IsComputed = true;
	}
}

string MarkovChain::GenerateWord(int length)
{
	this->StoreProbabilites();
	vector<double> vec;
	random_device rand;
	mt19937 generator(rand());

	for (double& p : m_Propabilities)
		vec.push_back(p * 100);

	discrete_distribution<> dist(vec.begin(), vec.end());
	string res;
	for (size_t i = 0; i < length; i++)
	{
		auto idxPair = m_IndexPairMap[dist(generator)];
		string str = string(idxPair.second);
		if (END_TOKEN != str)
			res += str;
		if (END_TOKEN != idxPair.first.front())
			res += " " + idxPair.first.front() + " ";
	}
	return res;
}

vector<string> MarkovChain::BuildStateSequance(string filePath, string delimiter, map<string, State>& sequenceSymbol, map<string, size_t>& symbolIndex)
{
	vector<string> datasetSequence;
	fstream file;
	file.open(filePath, ios::in);
	if (file.is_open())
	{
		string line;
		string samleSequence = "";
		while (getline(file, line))
		{
			if (line == "")
			{
				datasetSequence.push_back(samleSequence.substr(0, samleSequence.size() - 1));
				samleSequence = "";
				continue;
			}

			vector<string> lineSample = Utils::split(line, delimiter);

			if (lineSample[0] == "")
				continue;

			if (lineSample[1] == "")
				continue;

			if (ispunct(lineSample[1][0]))
				continue;

			samleSequence += lineSample[1] + " ";

			if (sequenceSymbol.find(lineSample[1]) == sequenceSymbol.end())
			{
				sequenceSymbol.insert({ lineSample[1] , State() });
				sequenceSymbol[lineSample[1]].m_SympolFrequancy.insert({ lineSample[0], 1 });
				sequenceSymbol[lineSample[1]].m_Frequancy = 1;
			}
			else
			{
				if (sequenceSymbol[lineSample[1]].m_SympolFrequancy.find(lineSample[0]) == sequenceSymbol[lineSample[1]].m_SympolFrequancy.end())
					sequenceSymbol[lineSample[1]].m_SympolFrequancy.insert({ lineSample[0], 1 });

				else
					sequenceSymbol[lineSample[1]].m_SympolFrequancy[lineSample[0]]++;

				sequenceSymbol[lineSample[1]].m_Frequancy++;
			}

			if (symbolIndex.find(lineSample[0]) == symbolIndex.end())
				symbolIndex.insert({ lineSample[0], symbolIndex.size() });

		}
		file.close();
	}
	return datasetSequence;
}

void MarkovChain::SaveStateSequance(string modelPath, const map<string, State>& sequenceSymbol, const map<string, size_t> symbolIndex)
{
	ofstream modelFile(modelPath);

	modelFile << sequenceSymbol.size() + 2 << '\n';

	modelFile << START_TOKEN << ' ';
	for (auto kv = sequenceSymbol.begin(); kv != sequenceSymbol.end(); kv++)
		modelFile << kv->first << ' ';
	modelFile << END_TOKEN<< '\n';

	modelFile << symbolIndex.size() << '\n';

	modelFile << m_TransitionMatrix.size() << '\n';

	StoreProbabilites();
	int index = 0;
	for (const auto& kv : m_TransitionMatrix)
	{		
		modelFile << kv.first.first[kv.first.first.size()-1] << " " << kv.first.second << " " << m_Propabilities[index] << '\n';
		index++;
	}

	int numTransitionSymbols = 0;
	for (auto seqKV = sequenceSymbol.begin(); seqKV != sequenceSymbol.end(); seqKV++)	
		numTransitionSymbols += seqKV->second.m_SympolFrequancy.size();	

	modelFile << numTransitionSymbols << '\n';
	for (auto seqKV = sequenceSymbol.begin(); seqKV != sequenceSymbol.end(); seqKV++)	
		for (auto symKV = seqKV->second.m_SympolFrequancy.begin(); symKV != seqKV->second.m_SympolFrequancy.end(); symKV++)
			modelFile << seqKV->first << " " << symKV->first << " " << (double)symKV->second / (double)seqKV->second.m_Frequancy << '\n';

	modelFile.close();
}