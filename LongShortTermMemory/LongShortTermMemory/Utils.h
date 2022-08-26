#ifndef UTILS_H
#define UTILS_H

#include <string>
#include <sstream>
#include <set>
#include <map>
#include "Tensor.h"

using namespace std;

vector<vector<vector<vector<int>>>> GenerateDatasetBatch(const vector<vector<vector<int>>>& seqEmbedDataset, int batchSize);
Tensor<double> GenerateTensorDataset(const vector<vector<vector<vector<int>>>>& batchDataset);
Tensor<double> GenerateTensorDatasetLast1D(const vector<vector<vector<vector<int>>>>& batchDataset);
void SplitDataSeq2SeqInputOutput(const vector<vector<vector<vector<int>>>>& batchDataset, int shiftNum, 
    vector<vector<vector<vector<int>>>>& X, vector<vector<vector<vector<int>>>>& y);
void SplitDataSeq2SeqInputOutput(const vector<vector<string>>& dataset, int shiftNum, vector<vector<string>>& X, vector<vector<string>>& y);
void PrintIdxString(vector<vector<Tensor<double>>> yHat, Tensor<double> yTrue, map<int, string>& idxString);
Tensor<double> IncreaseShape(Tensor<double> X_tensor);
Tensor<double> DecreaseShape(Tensor<double> X_tensor);
double GetEcludianDistance(Tensor<double> A, Tensor<double> B);

namespace SequancePreprocessing
{
    vector<string> Tokenize(string text, char delimiter);
    vector<vector<string>> TokenizeDataset(const vector<string>& Dataset, char delimiter);
    int GetMaxSequanceLength(const vector<vector<string>>& tokenizeDataset);
    void GetVocabulary(const vector<vector<string>>& tokenizeDataset, set<string>& vocabularySet, int& vocabularySize, string endToken="EOS", string paddingToken="PAD", string unknownToken="UNK");
    void PaddingSequance(vector<vector<string>>& tokenizeDataset, int maxLen, string endToekn = "EOS", string paddingToken = "PAD");
    void GetTokenIdx(const set<string>& vocabularySet, map<string, int>& stringIdx, map<int, string>& idxString);
    vector<vector<vector<int>>> GetSequanceDatasetEmbedding(const vector<vector<string>>& tokenizeDataset, const map<string, int>& stringIdx, int vocabularySize, string unknownToken = "UNK");
    vector<vector<vector<vector<int>>>> ResahpeSequanceDatasetEmbedding(const vector<vector<vector<int>>>& seqEmbedDataset);
    vector<vector<int>> TextEmbeding(const vector<string>& samples, const map<string, int>& stringIdx, int vocabularySize, string unknownToken = "UNK");
}

namespace ActivationFunctions
{
    Tensor<double> Sigmoid(Tensor<double>& m_Input);
    Tensor<double> SigmoidDerivative(Tensor<double>& m_Input);
    Tensor<double> Tanh(Tensor<double>& m_Input);
    Tensor<double> TanhDerivative(Tensor<double>& m_Input);
    Tensor<double> Softmax(Tensor<double>& m_Input);
}

#endif // !UTILS_H
