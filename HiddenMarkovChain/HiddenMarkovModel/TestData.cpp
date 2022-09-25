#include "TestData.h"

void TestData::ReadTestData(const HiddenMarkovModel& model, istream& dataSource)
{
    size_t nSteps;
    size_t stepNumber;
    string stateName;
    string symbol; // supposed to be single character, string is used for simpler reading code

    dataSource >> nSteps;
    if (nSteps == 0)
        throw domain_error("Empty experiment data");

    for (size_t i = 0; i < nSteps; i++)
    {
        dataSource >> stepNumber >> stateName >> symbol;

        stateName = Utils::Lower(stateName);
        symbol = Utils::Lower(symbol);

        stateName = Utils::RemoveNumbers(stateName);
        symbol = Utils::RemoveNumbers(symbol);

        stateName = Utils::trim(stateName);
        symbol = Utils::trim(symbol);

        if (stateName == "")
            continue;

        if (symbol == "")
            continue;

        if (ispunct(stateName[0]))
            continue;

        size_t stateInd = model.m_StateNameIndex.at(stateName);
        size_t symbolInd = Utils::GetSymbolInd(model.m_SymbolIndex, symbol);

        m_TimeStateSymbol.emplace_back(stepNumber, stateInd, symbolInd);
    }
}

void TestData::ReadTestData(const HiddenMarkovModel& model, const string& dataPath, const string& delimiter)
{
	vector<vector<tuple<size_t, size_t, size_t>>> m_VecTimeStateSymbol;
	fstream file;
	file.open(dataPath, ios::in);
    size_t stepNumber = 0;
    string stateName;
    string symbol;
	if (file.is_open())
	{
		string line;		

        vector<tuple<size_t, size_t, size_t>> tmpTimeStateSymbol;
		while (getline(file, line))
		{
			if (line == "")
			{
                m_VecTimeStateSymbol.push_back(tmpTimeStateSymbol);
                stepNumber = 0;
				continue;
			}

			vector<string> lineSample = Utils::split(line, delimiter);
            symbol = lineSample[0];
            stateName = lineSample[1];

            if (stateName == "")
                continue;

            if (symbol == "")
                continue;

            if (ispunct(stateName[0]))
                continue;

            size_t stateInd = model.m_StateNameIndex.at(stateName);
            size_t symbolInd = Utils::GetSymbolInd(model.m_SymbolIndex, symbol);
            tmpTimeStateSymbol.emplace_back(stepNumber, stateInd, symbolInd);
            stepNumber++;
		}
		file.close();
	}
}