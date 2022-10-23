#include "utils.h"

namespace utils
{
	double ComputeTargetProbability(vector<int>& samplesVec, Data& data)
	{
		double num = 0;
		int total = 0;
		for (auto i : samplesVec)
		{
			if (i != -1)
			{
				num += data.ReadTarget(i);
				total++;
			}
		}
		return num / (total + 1e-8);
	}

	int ComputeTrue(vector<int>& samplesVec, Data& data)
	{
		int total = 0;
		for (auto each : samplesVec) {
			total += data.ReadTarget(each);
		}
		return total;
	}

	double GetSize(vector<int>& samples)
	{
		double num = 0;
		for (auto i : samples)
			if (i != -1)
				num++;

		return num;
	}

	double ComputeGini(int& sideTrue, int& sideSize)
	{
		double trueProb = (sideTrue * 1.0) / (sideSize + 1e-8);
		return (1 - trueProb * trueProb - (1 - trueProb) * (1 - trueProb));
	}

	double ComputeGiniIndex(int& leftTrue, int& leftSize, int& rightTrue, int& rightSize)
	{
		double leftProb = (leftSize * 1.0) / (leftSize + rightSize);
		double rightprob = (rightSize * 1.0) / (leftSize + rightSize);
		return leftProb * ComputeGini(leftTrue, leftSize) + rightprob * ComputeGini(rightTrue, rightSize);
	}

	int _sqrt(int num)
	{
		return int(sqrt(num));
	}

	int _log2(int num)
	{
		return int(log2(num));
	}

	int _none(int num)
	{
		return num;
	}

	void WriteDataToCSV(const vector<int>& results, Data& data, const string& filePath, bool train)
	{
		ofstream out(filePath);
		if (out.is_open())
		{
			out << "id, label";
			if (train)
				out << ",true\n";
			else
				out << "\n";

			int i = 0;
			for (auto each : results)
			{
				out << i << "," << each;

				if (train)
					out << "," << data.ReadTarget(i) << "\n";
				else
					out << "\n";

				i++;
			}
			out.close();
		}
		else
			cerr << "Write File faild" << endl;
	}

	double CalcualteACC(vector<int>& yTrue, vector<int>& yHat)
	{
		int countRight = 0;
		for (size_t i = 0; i < yTrue.size(); i++)
			if ((yTrue[i] == 0 && yHat[i] < 0.5) || (yTrue[i] == 1 && yHat[i] >= 0.5))
				countRight += 1;

		return (double)countRight / yTrue.size();
	}
}