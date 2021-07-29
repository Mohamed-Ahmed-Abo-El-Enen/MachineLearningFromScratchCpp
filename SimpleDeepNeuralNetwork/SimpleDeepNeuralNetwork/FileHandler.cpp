#include "FileHandler.h"

FileHandler::FileHandler()
{
}

bool FileHandler::isEof()
{
    return m_FileHandler.eof();
}

unsigned FileHandler::GetSampleFeatures(vector<double>& sampleFeature)
{
    sampleFeature.clear();
    string line;
    getline(m_FileHandler, line);
    stringstream stringStream(line);
    string label;
    stringStream >> label;
    if (label.compare("in:") == 0)
    {
        double oneValue;
        while (stringStream >> oneValue)
        {
            sampleFeature.push_back(oneValue);
        }
    }
    return sampleFeature.size();
}

unsigned FileHandler::GetSampleLabel(vector<double>& sampleLabel)
{
    sampleLabel.clear();
    string line;
    getline(m_FileHandler, line);
    stringstream stringStream(line);
    string label;
    stringStream >> label;
    if (label.compare("out:") == 0)
    {
        double oneValue;
        while (stringStream >> oneValue)
        {
            sampleLabel.push_back(oneValue);
        }
    }
    return sampleLabel.size();
}

void FileHandler::ReadDatasetFile(string filePath, vector<vector<double>>& datasetFeatures, vector<vector<double>>& datasetTargets)
{
    m_FileHandler.open(filePath.c_str());

    int trainingPass = 0;

    while (!isEof())
    {
        vector<double> inputVals, targetVals;
        trainingPass++;
        GetSampleFeatures(inputVals);
        datasetFeatures.push_back(inputVals);
        GetSampleLabel(targetVals);
        datasetTargets.push_back(targetVals);
    }
    cout << "Done" << endl;
}