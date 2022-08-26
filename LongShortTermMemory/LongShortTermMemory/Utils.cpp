#include "Utils.h"


namespace SequancePreprocessing
{
	vector<string> Tokenize(string text, char delimiter)
	{
		stringstream sStream(text);
		string item;
		vector<string> result;
		while (getline(sStream, item, delimiter))
			result.push_back(item);
		
		return result;
	}

	vector<vector<string>> TokenizeDataset(const vector<string>& Dataset, char delimiter)
	{
		vector<vector<string>> result;
		for (string text : Dataset)	
			result.push_back(Tokenize(text, delimiter));
				
		return result;
	}

	int GetMaxSequanceLength(const vector<vector<string>>& tokenizeDataset)
	{
		int maxLen = 0;
		for (vector<string> tokens : tokenizeDataset)
			maxLen = max(maxLen, (int)tokens.size());
		
		return maxLen;
	}

	void GetVocabulary(const vector<vector<string>>& tokenizeDataset, set<string>& vocabularySet, int& vocabularySize, string endToken, string paddingToken, string unknownToken)
	{

		for (vector<string> tokens : tokenizeDataset)
			for (string x : tokens)
				vocabularySet.insert(x);
		
		vocabularySet.insert(endToken);
		vocabularySet.insert(paddingToken);
		vocabularySet.insert(unknownToken);
		vocabularySize = vocabularySet.size();
	}

	void GetTokenIdx(const set<string>& vocabularySet, map<string, int>& stringIdx, map<int, string>& idxString)
	{
		int idx = 0;
		for (string x: vocabularySet)
		{
			if (stringIdx.find(x) != stringIdx.end())
				continue;
			stringIdx[x] = idx;
			idxString[idx] = x;
			idx++;
		}		
	}

	void PaddingSequance(vector<vector<string>>& tokenizeDataset, int maxLen, string endToekn, string paddingToken)
	{
		for (vector<string>& sequance : tokenizeDataset)
		{
			int seqLength = sequance.size();
			for (size_t i = sequance.size(); i < maxLen + 1; i++)
			{
				if (i == seqLength)
					sequance.push_back(endToekn);
				else
					sequance.push_back(paddingToken);
			}
		}
	}

	vector<vector<vector<int>>> GetSequanceDatasetEmbedding(const vector<vector<string>>& tokenizeDataset, const map<string, int>& stringIdx, int vocabularySize, string unknownToken)
	{
		vector<vector<vector<int>>> seqEmbedDataset;
		for (vector<string> x: tokenizeDataset)
		{
			vector<vector<int>> seqEmbed;
			for (string& s: x)
			{
				vector<int> m_Embed(vocabularySize, 0);

				if(stringIdx.find(s) != stringIdx.end())
					m_Embed[stringIdx.at(s)] = 1;
				else
					m_Embed[stringIdx.at(unknownToken)] = 1;

				seqEmbed.push_back(m_Embed);
			}
			seqEmbedDataset.push_back(seqEmbed);
		}
		return seqEmbedDataset;
	}

	vector<vector<vector<vector<int>>>> ResahpeSequanceDatasetEmbedding(const vector<vector<vector<int>>>& seqEmbedDataset)
	{
		vector<vector<vector<vector<int>>>> reshapedResult;
		for(vector<vector<int>> x: seqEmbedDataset)
		{
			vector<vector<vector<int>>> resultX;
			for (vector<int>& y: x)
			{
				vector<vector<int>> resultY;
				for (int& val : y)
					resultY.push_back(vector<int>(1, val));				
				resultX.push_back(resultY);
			}
			reshapedResult.push_back(resultX);
		}
		return reshapedResult;
	}

	vector<vector<int>> TextEmbeding(const vector<string>& samples, const map<string, int>& stringIdx, int vocabularySize, string unknownToken)
	{
		vector<vector<int>> seqEmbed;
		for (string s : samples)
		{
			vector<int> m_Embed(vocabularySize, 0);

			if (stringIdx.find(s) != stringIdx.end())
				m_Embed[stringIdx.at(s)] = 1;
			else
				m_Embed[stringIdx.at(unknownToken)] = 1;

			seqEmbed.push_back(m_Embed);
		}
		return seqEmbed;
	}
}

double GetTanhVal(double x)
{
	double safeX = x + 1e-12;
	double result = (exp(safeX) - exp(-safeX)) / (exp(safeX) + exp(-safeX));
	return result;
}

double GetTanhDerivativeVal(double x)
{
	return 1 - pow(GetTanhVal(x), 2);
}

double GetSigmoidVal(double x)
{
	double safeX = x + 1e-12;
	return 1.0 / (1.0 + exp(-safeX));
}

double GetSigmoidDerivativeVal(double x)
{
	double safeX = GetSigmoidVal(x);
	return (safeX * (1.0 - safeX));
}

namespace ActivationFunctions
{
	Tensor<double> Sigmoid(Tensor<double>& m_Input)
	{
		Tensor<double> result(m_Input.numDims, m_Input.m_Dims);
		for (int i = 0; i < m_Input.GetSize(); i++)
		{
			double x = m_Input.GetListIndex(i);
			double value = GetSigmoidVal(x);
			result.SetListIndex(value, i);
		}
		return result;
	}

	Tensor<double> SigmoidDerivative(Tensor<double>& m_Input)
	{
		Tensor<double> result(m_Input.numDims, m_Input.m_Dims);
		for (int i = 0; i < m_Input.GetSize(); i++)
		{
			double x = m_Input.GetListIndex(i);
			double value = GetSigmoidDerivativeVal(x);
			result.SetListIndex(value, i);
		}
		return result;
	}

	Tensor<double> Tanh(Tensor<double>& m_Input)
	{
		Tensor<double> result(m_Input.numDims, m_Input.m_Dims);
		for (int i = 0; i < m_Input.GetSize(); i++)
		{
			double x = m_Input.GetListIndex(i);
			double value = GetTanhVal(x);
			result.SetListIndex(value, i);
		}
		return result;
	}

	Tensor<double> Softmax(Tensor<double>& m_Input)
	{
		assert(m_Input.numDims == 2);
		Tensor<double> probabilities(2, m_Input.m_Dims);

		for (size_t j = 0; j < m_Input.m_Dims[1]; j++)
		{
			double expSum = 0;
			for (size_t i = 0; i < m_Input.m_Dims[0]; i++)
				expSum += std::exp(m_Input.get(i, j) + 1e-12);

			for (size_t i = 0; i < m_Input.m_Dims[0]; i++)
				probabilities.set(std::exp(m_Input.get(i, j) + 1e-12) / expSum, i, j);
		}

		/*double expSum = m_Input.exp().sum() + m_Input.GetSize() * 1e-12;

		for (int i = 0; i < m_Input.GetSize(); i++)		
			expSum += std::exp(m_Input.GetListIndex(i) + 1e-12);
		
		for (int i = 0; i < m_Input.GetSize(); i++)
			probabilities.SetListIndex(std::exp(m_Input.GetListIndex(i) + 1e-12) / expSum, i);*/

		return probabilities;		
	}

	Tensor<double> TanhDerivative(Tensor<double>& m_Input)
	{
		Tensor<double> result(m_Input.numDims, m_Input.m_Dims);
		for (int i = 0; i < m_Input.GetSize(); i++)
		{
			double x = m_Input.GetListIndex(i);
			double value = 1 - pow(GetTanhVal(x), 2);
			result.SetListIndex(value, i);
		}
		return result;
	}
}

vector<vector<vector<vector<int>>>> GenerateDatasetBatch(const vector<vector<vector<int>>>& seqEmbedDataset, int batchSize)
{
	vector<vector<vector<vector<int>>>> batcheDataset;

	int i = 0;
	while (true)
	{
		size_t start = i * batchSize;
		size_t end = start + batchSize;

		if (end > seqEmbedDataset.size())
			break;

		vector<vector<vector<int>>> batchData;
		for (size_t j = start; j < end; j++)		
			batchData.push_back(seqEmbedDataset[j]);

		batcheDataset.push_back(batchData);
		i++;
	}
	return batcheDataset;
}

vector<vector<vector<vector<int>>>> ReshapeDatasetBatch(const vector<vector<vector<vector<int>>>>& batchDataset)
{
	vector<vector<vector<vector<int>>>> reshapedResult;
	for (vector<vector<vector<int>>> batch : batchDataset)
	{
		vector<vector<vector<int>>> batchResult;
		for (size_t i = 0; i < batch[0].size(); i++)
		{
			vector<vector<int>> sample;
			for (size_t j = 0; j < batch.size(); j++)
			{
				sample.push_back(batch[j][i]);
			}
			batchResult.push_back(sample);
		}		
		reshapedResult.push_back(batchResult);
	}
	return reshapedResult;
}


Tensor<double> GenerateTensorDataset(const vector<vector<vector<vector<int>>>>& batchDataset)
{
	int m_Dims[] = { batchDataset.size(), batchDataset[0].size(), batchDataset[0][0].size(), batchDataset[0][0][0].size() };
	Tensor<double> ptr_tensorBatchDataset(4, m_Dims);

	for (size_t i = 0; i < m_Dims[0]; i++)
		for (size_t j = 0; j < m_Dims[1]; j++)
			for (size_t k = 0; k < m_Dims[2]; k++)
				for (size_t l = 0; l < m_Dims[3]; l++)
					ptr_tensorBatchDataset.set((double)batchDataset[i][j][k][l], i, j, k, l);

	return ptr_tensorBatchDataset;
}

Tensor<double> GenerateTensorDatasetLast1D(const vector<vector<vector<vector<int>>>>& batchDataset)
{
	int m_Dims[] = { batchDataset.size(), batchDataset[0].size(), batchDataset[0][0].size(), batchDataset[0][0][0].size() };
	Tensor<double> tensorBatchDataset(4, m_Dims);

	for (size_t i = 0; i < m_Dims[0]; i++)
	{
		size_t j_end = m_Dims[1] > 0 ? m_Dims[1] : 1;
		for (size_t j = 0; j < j_end; j++)
		{
			size_t k_end = m_Dims[2] > 0 ? m_Dims[2] : 1;
			for (size_t k = 0; k < k_end; k++)
			{
				size_t l_end = m_Dims[3] > 0 ? m_Dims[3] : 1;
				for (size_t l = 0; l < l_end; l++)
					tensorBatchDataset.set((double)batchDataset[i][j][k][l], i, j, k, l);
			}
		}
	}
	return tensorBatchDataset;
}

void SplitDataSeq2SeqInputOutput(const vector<vector<vector<vector<int>>>>& dataset, int shiftNum,  
	vector<vector<vector<vector<int>>>>& X, vector<vector<vector<vector<int>>>>& y)
{
	for (vector<vector<vector<int>>> sample : dataset)
	{
		vector<vector<vector<int>>> xSampleLevel;
		vector<vector<vector<int>>> ySampleLevel;

		for (size_t j = 0; j < sample.size() - shiftNum; j++)
		{
			vector<vector<int>> xTLevel;
			for (size_t k = 0; k < sample[0].size(); k++)
			{
				vector<int> xEmbeddingLevel;
				for (size_t l = 0; l < sample[0][0].size(); l++)
					xEmbeddingLevel.push_back(sample[j][k][l]);
				xTLevel.push_back(xEmbeddingLevel);
			}
			xSampleLevel.push_back(xTLevel);
		}

		for (size_t j = shiftNum; j < sample.size(); j++)
		{
			vector<vector<int>> yTLevel;
			for (size_t k = 0; k < sample[0].size(); k++)
			{
				vector<int> yEmbeddingLevel;
				for (size_t l = 0; l < sample[0][0].size(); l++)
					yEmbeddingLevel.push_back(sample[j][k][l]);
				yTLevel.push_back(yEmbeddingLevel);
			}
			ySampleLevel.push_back(yTLevel);
		}
		X.push_back(xSampleLevel);
		y.push_back(ySampleLevel);
	}
}

void SplitDataSeq2SeqInputOutput(const vector<vector<string>>& dataset, int shiftNum,
	vector<vector<string>>& X, vector<vector<string>>& y)
{
	for (vector<string> sample : dataset)
	{
		for (size_t j = 0; j < sample.size() - shiftNum; j++)		
			X.push_back(dataset[j]);		

		for (size_t j = shiftNum; j < sample.size(); j++)
			y.push_back(dataset[j]);
	}
}

void PrintIdxString(vector<vector<Tensor<double>>> yHat, Tensor<double> yTrue, map<int, string>& idxString)
{
	for (size_t i = 0; i < yTrue.m_Dims[0]; i++)
	{
		cout << "Y True:" << endl;
		for (size_t j = 0; j < yTrue.m_Dims[1]; j++)		
			cout << idxString[(int)yTrue[i][j].argmax(0).get(0, 0)] << " ";
		cout << endl;

		cout << "Y Hat:" << endl;

		for (size_t j = 0; j < yHat[i].size(); j++)
			cout << idxString[(int)yHat[i][j].argmax(0).get(0, 0)] << " ";
		cout << endl;
	}
}

Tensor<double> DecreaseShape(Tensor<double> X_tensor)
{
	vector<int> vecDims;
	for (size_t i = 0; i < X_tensor.numDims - 1; i++)
	{
		vecDims.push_back(X_tensor.m_Dims[i]);
	}

	int* newDims = &vecDims[0];
	Tensor<double> result(vecDims.size(), newDims);

	for (size_t i = 0; i < X_tensor.m_Dims[0]; i++)
	{
		if (X_tensor.numDims >= 3)
		{
			for (size_t j = 0; j < X_tensor.m_Dims[1]; j++)
			{
				if (X_tensor.numDims >= 4)
				{
					for (size_t k = 0; k < X_tensor.m_Dims[2]; k++)
						result.set(X_tensor.get(i, j, k, 0), i, j, k);
				}
				else
					result.set(X_tensor.get(i, j, 0), i, j);
			}
		}
		else
			result.set(X_tensor.get(i, 0), i);
	}
	return result;
}

Tensor<double> IncreaseShape(Tensor<double> X_tensor)
{
	vector<int> vecDims;
	for (size_t i = 0; i <= X_tensor.numDims; i++)
	{
		if (i == X_tensor.numDims)
			vecDims.push_back(1);
		else
			vecDims.push_back(X_tensor.m_Dims[i]);
	}

	int* newDims = &vecDims[0];
	Tensor<double> result(vecDims.size(), newDims);

	for (size_t i = 0; i < X_tensor.m_Dims[0]; i++)
	{
		if (X_tensor.numDims >= 2)
		{
			for (size_t j = 0; j < X_tensor.m_Dims[1]; j++)
			{
				if (X_tensor.numDims >= 3)
				{
					for (size_t k = 0; k < X_tensor.m_Dims[2]; k++)
						result.set(X_tensor.get(i, j, k), i, j, k, 0);
				}
				else
					result.set(X_tensor.get(i, j), i, j, 0);
			}
		}
		else
			result.set(X_tensor.get(i), i, 0);
	}
	return result;
}

double GetEcludianDistance(Tensor<double> A, Tensor<double> B)
{
	double distance = 0;
	for (size_t i = 0; i < A.m_Dims[0]; i++)
		for (size_t j = 0; j < A.m_Dims[1]; j++)
			distance += std::pow(A.get(i, j) - B.get(i, j), 2);

	return std::sqrt(distance);
}
