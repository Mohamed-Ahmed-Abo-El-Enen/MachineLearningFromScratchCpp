#include "Utils.h"
#include "LSTM.h"
#include "RNN.h"
#include "Adam.h"
#include "Embedding.h"
#include "GradientDescent.h"

// Text Preprocessing 
void Test1(vector<string> seqDataset)
{
    char delimiter = ' ';
    vector<vector<string>> tokenizeDataset = SequancePreprocessing::TokenizeDataset(seqDataset, delimiter);

    int maxLen = SequancePreprocessing::GetMaxSequanceLength(tokenizeDataset);

    set<string> vocabularySet;
    int vocabularySize;
    SequancePreprocessing::GetVocabulary(tokenizeDataset, vocabularySet, vocabularySize);

    map<string, int> stringIdx;
    map<int, string> idxString;
    SequancePreprocessing::GetTokenIdx(vocabularySet, stringIdx, idxString);

    SequancePreprocessing::PaddingSequance(tokenizeDataset, maxLen);

    vector<vector<vector<int>>> seqEmbedDataset;
    seqEmbedDataset = SequancePreprocessing::GetSequanceDatasetEmbedding(tokenizeDataset, stringIdx, vocabularySize);

    vector<vector<vector<vector<int>>>> reshapedSeqEmbedDataset = SequancePreprocessing::ResahpeSequanceDatasetEmbedding(seqEmbedDataset);
}

// Convert To Tensor datatype
void Test2(vector<string> seqDataset)
{
    char delimiter = ' ';
    vector<vector<string>> tokenizeDataset = SequancePreprocessing::TokenizeDataset(seqDataset, delimiter);

    int maxLen = SequancePreprocessing::GetMaxSequanceLength(tokenizeDataset);

    set<string> vocabularySet;
    int vocabularySize;
    SequancePreprocessing::GetVocabulary(tokenizeDataset, vocabularySet, vocabularySize);

    map<string, int> stringIdx;
    map<int, string> idxString;
    SequancePreprocessing::GetTokenIdx(vocabularySet, stringIdx, idxString);

    SequancePreprocessing::PaddingSequance(tokenizeDataset, maxLen);

    vector<vector<vector<int>>> seqEmbedDataset;
    seqEmbedDataset = SequancePreprocessing::GetSequanceDatasetEmbedding(tokenizeDataset, stringIdx, vocabularySize);

    vector<vector<vector<vector<int>>>> reshapedSeqEmbedDataset = SequancePreprocessing::ResahpeSequanceDatasetEmbedding(seqEmbedDataset);

    int shiftNum = 1;

    vector<vector<vector<vector<int>>>> X;
    vector<vector<vector<vector<int>>>> y;
    SplitDataSeq2SeqInputOutput(reshapedSeqEmbedDataset, shiftNum, X, y);
    Tensor<double> X_tensor = GenerateTensorDatasetLast1D(X);
    Tensor<double> y_tesor = GenerateTensorDatasetLast1D(y);
}

// RNN
void Test3(vector<string> seqDataset)
{
    char delimiter = ' ';
    vector<vector<string>> tokenizeDataset = SequancePreprocessing::TokenizeDataset(seqDataset, delimiter);

    int maxLen = SequancePreprocessing::GetMaxSequanceLength(tokenizeDataset);

    set<string> vocabularySet;
    int vocabularySize;
    SequancePreprocessing::GetVocabulary(tokenizeDataset, vocabularySet, vocabularySize);

    map<string, int> stringIdx;
    map<int, string> idxString;
    SequancePreprocessing::GetTokenIdx(vocabularySet, stringIdx, idxString);

    SequancePreprocessing::PaddingSequance(tokenizeDataset, maxLen);

    vector<vector<vector<int>>> seqEmbedDataset;
    seqEmbedDataset = SequancePreprocessing::GetSequanceDatasetEmbedding(tokenizeDataset, stringIdx, vocabularySize);

    vector<vector<vector<vector<int>>>> reshapedSeqEmbedDataset = SequancePreprocessing::ResahpeSequanceDatasetEmbedding(seqEmbedDataset);

    int shiftNum = 1;

    vector<vector<vector<vector<int>>>> X;
    vector<vector<vector<vector<int>>>> y;
    SplitDataSeq2SeqInputOutput(reshapedSeqEmbedDataset, shiftNum, X, y);
    Tensor<double> X_tensor = GenerateTensorDatasetLast1D(X);
    Tensor<double> y_tesor = GenerateTensorDatasetLast1D(y);

    //number of input units or embedding size
    int intputUnits = 100;

    //number of hidden neurons
    int hiddenUnits = 50;

    //number of output units i.e vocab size
    int outputUnits = vocabularySize;

    // learning rate
    double learningRate = 0.05;

    // beta1 for V parameters used in adam optimizer
    double beta1 = 0.90;

    // beta2 for S parameters used in adam optimizer
    double beta2 = 0.99;

    int epochs = 500;

    Optimizer* optimizer = new Adam(beta1, beta2, learningRate);

    RNN rnn(hiddenUnits, outputUnits, optimizer);
    rnn.Train(X_tensor, y_tesor, epochs);
    vector<vector<Tensor<double>>> yHat = rnn.Predict(X_tensor);

    PrintIdxString(yHat, y_tesor, idxString);
}

// LSTM
void Test4(vector<string> seqDataset)
{
    char delimiter = ' ';
    vector<vector<string>> tokenizeDataset = SequancePreprocessing::TokenizeDataset(seqDataset, delimiter);

    int maxLen = SequancePreprocessing::GetMaxSequanceLength(tokenizeDataset);

    set<string> vocabularySet;
    int vocabularySize;
    SequancePreprocessing::GetVocabulary(tokenizeDataset, vocabularySet, vocabularySize);

    map<string, int> stringIdx;
    map<int, string> idxString;
    SequancePreprocessing::GetTokenIdx(vocabularySet, stringIdx, idxString);

    SequancePreprocessing::PaddingSequance(tokenizeDataset, maxLen);

    vector<vector<vector<int>>> seqEmbedDataset;
    seqEmbedDataset = SequancePreprocessing::GetSequanceDatasetEmbedding(tokenizeDataset, stringIdx, vocabularySize);

    vector<vector<vector<vector<int>>>> reshapedSeqEmbedDataset = SequancePreprocessing::ResahpeSequanceDatasetEmbedding(seqEmbedDataset);

    int shiftNum = 1;

    vector<vector<vector<vector<int>>>> X;
    vector<vector<vector<vector<int>>>> y;
    SplitDataSeq2SeqInputOutput(reshapedSeqEmbedDataset, shiftNum, X, y);
    Tensor<double> X_tensor = GenerateTensorDatasetLast1D(X);
    Tensor<double> y_tesor = GenerateTensorDatasetLast1D(y);

    //number of input units or embedding size
    int intputUnits = 100;

    //number of hidden neurons
    int hiddenUnits = 50;

    //number of output units i.e vocab size
    int outputUnits = vocabularySize;

    // learning rate
    double learningRate = 0.05;

    // beta1 for V parameters used in adam optimizer
    double beta1 = 0.90;

    // beta2 for S parameters used in adam optimizer
    double beta2 = 0.99;

    int epochs = 500;

    Optimizer* optimizer = new Adam(beta1, beta2, learningRate);


    LSTM lstm(intputUnits, hiddenUnits, outputUnits, optimizer);

    lstm.Train(X_tensor, y_tesor, epochs);
    vector<vector<Tensor<double>>> yHat = lstm.Predict(X_tensor);
    PrintIdxString(yHat, y_tesor, idxString);
}


// Embedding Layer
void Test5(vector<string> seqDataset)
{
    char delimiter = ' ';
    vector<vector<string>> tokenizeDataset = SequancePreprocessing::TokenizeDataset(seqDataset, delimiter);

    int maxLen = SequancePreprocessing::GetMaxSequanceLength(tokenizeDataset);

    set<string> vocabularySet;
    int vocabularySize;
    SequancePreprocessing::GetVocabulary(tokenizeDataset, vocabularySet, vocabularySize);

    map<string, int> stringIdx;
    map<int, string> idxString;
    SequancePreprocessing::GetTokenIdx(vocabularySet, stringIdx, idxString);

    SequancePreprocessing::PaddingSequance(tokenizeDataset, maxLen);

    vector<vector<vector<int>>> seqEmbedDataset;
    seqEmbedDataset = SequancePreprocessing::GetSequanceDatasetEmbedding(tokenizeDataset, stringIdx, vocabularySize);

    vector<vector<vector<vector<int>>>> reshapedSeqEmbedDataset = SequancePreprocessing::ResahpeSequanceDatasetEmbedding(seqEmbedDataset);

    int shiftNum = 1;

    Tensor<double> X_tensor = GenerateTensorDatasetLast1D(reshapedSeqEmbedDataset);

    //number of input units or embedding size
    int intputUnits = 10;

    //number of hidden neurons
    int hiddenUnits = 50;

    //number of output units i.e vocab size
    int outputUnits = vocabularySize;

    // learning rate
    double learningRate = 0.01;

    // beta1 for V parameters used in adam optimizer
    double beta1 = 0.90;

    // beta2 for S parameters used in adam optimizer
    double beta2 = 0.99;

    int epochs = 500;

    Optimizer* optimizer = new Adam(beta1, beta2, learningRate);

    bool usingEmbedding = true;
    if (usingEmbedding)
    {
        bool decreaseShape = true;
        bool increaseShape = true;

        if (decreaseShape)
            X_tensor = DecreaseShape(X_tensor);

        int windowSize = 2;
        Embedding embedding(vocabularySize, intputUnits, windowSize, optimizer);
        embedding.Train(X_tensor, epochs);
        X_tensor = embedding.GetEmbedding(X_tensor);

        if (increaseShape)
            X_tensor = IncreaseShape(X_tensor);
    }
}

// LSTM with embedding
void Test6(vector<string> seqDataset)
{
    char delimiter = ' ';
    vector<vector<string>> tokenizeDataset = SequancePreprocessing::TokenizeDataset(seqDataset, delimiter);

    int maxLen = SequancePreprocessing::GetMaxSequanceLength(tokenizeDataset);

    set<string> vocabularySet;
    int vocabularySize;
    SequancePreprocessing::GetVocabulary(tokenizeDataset, vocabularySet, vocabularySize);

    map<string, int> stringIdx;
    map<int, string> idxString;
    SequancePreprocessing::GetTokenIdx(vocabularySet, stringIdx, idxString);

    SequancePreprocessing::PaddingSequance(tokenizeDataset, maxLen);

    vector<vector<vector<int>>> seqEmbedDataset;
    seqEmbedDataset = SequancePreprocessing::GetSequanceDatasetEmbedding(tokenizeDataset, stringIdx, vocabularySize);

    vector<vector<vector<vector<int>>>> reshapedSeqEmbedDataset = SequancePreprocessing::ResahpeSequanceDatasetEmbedding(seqEmbedDataset);

    int shiftNum = 1;

    vector<vector<vector<vector<int>>>> X;
    vector<vector<vector<vector<int>>>> y;
    SplitDataSeq2SeqInputOutput(reshapedSeqEmbedDataset, shiftNum, X, y);
    Tensor<double> X_tensor = GenerateTensorDatasetLast1D(X);
    Tensor<double> y_tesor = GenerateTensorDatasetLast1D(y);

    //number of input units or embedding size
    int intputUnits = 10;

    //number of hidden neurons
    int hiddenUnits = 50;

    //number of output units i.e vocab size
    int outputUnits = vocabularySize;

    // learning rate
    double learningRate = 0.01;

    // beta1 for V parameters used in adam optimizer
    double beta1 = 0.90;

    // beta2 for S parameters used in adam optimizer
    double beta2 = 0.99;

    int epochs = 50;

    Optimizer* optimizer = new Adam(beta1, beta2, learningRate);

    bool usingEmbedding = true;
    if (usingEmbedding)
    {
        bool decreaseShape = true;
        bool increaseShape = true;

        if (decreaseShape)
            X_tensor = DecreaseShape(X_tensor);

        int windowSize = 2;
        Embedding embedding(vocabularySize, intputUnits, windowSize, optimizer);
        embedding.Train(X_tensor, epochs);
        X_tensor = embedding.GetEmbedding(X_tensor);

        if (increaseShape)
            X_tensor = IncreaseShape(X_tensor);
    }

    LSTM lstm(intputUnits, hiddenUnits, outputUnits, optimizer, usingEmbedding);

    lstm.Train(X_tensor, y_tesor, epochs);
    vector<vector<Tensor<double>>> yHat = lstm.Predict(X_tensor);
    PrintIdxString(yHat, y_tesor, idxString);
}

int main()
{
    vector<string> seqDataset;
    for (size_t i = 0; i < 10; i++)
    {
        string text = "";
        int min = 1;
        int max = 10;
        int randA = rand() % (max - min + 1) + min;
        for (size_t j = 0; j < randA; j++)        
            text += "a ";
        
        int randB = rand() % (max - min + 1) + min;
        for (size_t j = 0; j < randB; j++)
            text += "b "; 

        text = text.substr(0, text.size() - 1);
        seqDataset.push_back(text);
    }       

    //Test1(seqDataset);
    //Test2(seqDataset);
    //Test3(seqDataset);
    //Test4(seqDataset);
    //Test5(seqDataset);
    Test6(seqDataset);

    cout << "=================DONE======================" << endl;
}