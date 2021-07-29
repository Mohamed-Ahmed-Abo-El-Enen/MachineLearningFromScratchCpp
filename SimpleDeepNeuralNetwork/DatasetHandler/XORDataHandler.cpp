#include<time.h>
#include "XORDataHandler.h"

XORDataHandler::XORDataHandler(const string filename)
{
    m_XORDataHandlerFile.open(filename.c_str());
}

bool XORDataHandler::isEof()
{
    return m_XORDataHandlerFile.eof();
}

unsigned XORDataHandler::GetNextInputs(vector<double>& inputVals)
{
    inputVals.clear();
    string line;
    getline(m_XORDataHandlerFile, line);
    stringstream stringStream(line);
    string label;
    stringStream >> label;
    if (label.compare("in:") == 0) 
    {
        double oneValue;
        while (stringStream >> oneValue) 
        {
            inputVals.push_back(oneValue);
        }
    }
    return inputVals.size();
}

unsigned XORDataHandler::GetTargetOutputs(vector<double>& targetOutputVals)
{
    targetOutputVals.clear();
    string line;
    getline(m_XORDataHandlerFile, line);
    stringstream stringStream(line);
    string label;
    stringStream >> label;
    if (label.compare("out:") == 0) 
    {
        double oneValue;
        while (stringStream >> oneValue) 
        {
            targetOutputVals.push_back(oneValue);
        }
    }
    return targetOutputVals.size();
}

void XORDataHandler::GenerateXORData()
{
    srand(time(NULL));
    for (int i = 2000; i >= 0; --i)
    {
        int n1 = (int)(2.0 * rand() / double(RAND_MAX));
        int n2 = (int)(2.0 * rand() / double(RAND_MAX));
        long long t = n1 ^ n2; // 0 or 1
        cout << "in: " << n1 << ".0 " << n2 << ".0 " << 1.0 << endl;
        cout << "out: " << t << ".0 " << 1.0 <<endl;
    }
}

void XORDataHandler::ReadXORData(vector<vector<double>>& datasetFeatures, vector<vector<double>>& datasetTargets)
{
    int trainingPass = 0;

    while (!isEof())
    {
        vector<double> inputVals, targetVals;
        trainingPass++;
        GetNextInputs(inputVals);
        datasetFeatures.push_back(inputVals);
        GetTargetOutputs(targetVals);
        datasetTargets.push_back(targetVals);
    }
    cout << "Done" << endl;
}
