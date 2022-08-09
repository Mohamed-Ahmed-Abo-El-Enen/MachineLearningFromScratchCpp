#include <cmath>
#include <fstream>
#include <filesystem>
#include <opencv2/opencv.hpp>

#include "Util.h"

using namespace std;

namespace fs = std::filesystem;

vector<string> ListAllDirectoryFile(string directoryPath)
{
	vector<string> filePath;
	for (const auto& entry : fs::directory_iterator(directoryPath))
		filePath.push_back(entry.path().string());
	return filePath;
}

vector<string> SpliteLine(string line, string delimiter)
{
	vector<string> values;
	size_t pos = 0;
	string token;
	while ((pos = line.find(delimiter)) && pos != string::npos)
	{
		token = line.substr(0, pos);
		values.push_back(token);
		line.erase(0, pos + delimiter.length());
	}
	values.push_back(line);
	return values;
}

namespace fns
{
	double relu(double x)
	{
		if (x > 0) 
			return x;
		else 
			return (double)0;
	}

	double sigmoid(double x)
	{
		return (1.0 / (1.0 + exp(-x)));
	}

	double tan(double x)
	{
		return tanh(x);
	}

	double relu_gradient(double x)
	{
		if (x > 0)return (double)1;
		else return (double)0.2;
	}

	double sigmoid_gradient(double x)
	{
		return(x * (1 - x));
	}

	double tan_gradient(double x)
	{
		return (1 - (x * x));
	}

	double softmax(double x)
	{
		if (isnan(x)) return 0;
		return exp(x);
	}
}

namespace Prerocess
{
	void ProcessMNISTImage(const char* filePath, vector<unique_ptr<Matrix>>& xSample, vector<unique_ptr<vector<double>>>& ySample, size_t nImages)
	{
		const int width = 28;
		const int height = 28;
		const int labels = 10;

		for (unsigned int i = 0; i < labels; i++)
		{
			vector<string> files = ListAllDirectoryFile(filePath + to_string(i));
			for (unsigned int k = 0; k < (nImages / labels); k++)
			{
				cv::Mat img = cv::imread(files[k]);
				if (img.empty())
					continue;
				unique_ptr<Matrix> image = make_unique<Matrix>(width, height, true);
				for (unsigned int h = 0; h < height; h++)				
					for (unsigned int w = 0; w < width; w++)					
						image->set(h, w, (double)(img.at<uchar>(h, w) / 255.0));									

				xSample.emplace_back(move(image));
				unique_ptr<vector<double>> vr = make_unique<vector<double>>(labels, 0);
				(*vr)[i] = 1.0;
				ySample.emplace_back(move(vr));
			}			
		}
	}

	void ProcessMNISTCSV(const char* filePath, vector<vector<double>>& xSample, vector<vector<double>>& ySample)
	{
		string data(filePath);
		ifstream in(data.c_str());

		if (!in.is_open())
			return;

		string line;
		while(getline(in, line))
		{
			vector<string> lineVec = SpliteLine(line, ",");
			int lableIndex = stoi(lineVec[0]);
			vector<double> lables(10, 0.0);
			lables[lableIndex] = 1.0;

			vector<double> xVec;
			for (size_t i = 1; i < lineVec.size(); i++)
			{
				xVec.push_back(stod(lineVec[i])/255);
			}

			xSample.push_back(xVec);
			ySample.push_back(lables);
		}
		cout << "DONE: Pocesseing the input file" << endl;
	}

	void ProcessImage(const char* filePath)
	{
		vector<double> image;
		cv::Mat img = cv::imread(filePath);
		if (img.empty())		
			cout << "ERROR: No Image" << endl;
		else
		{
			if (img.isContinuous())
			{
				image.assign(img.datastart, img.dataend);
				for (unsigned int i = 0; i < image.size(); i++)
					cout << image[i] << "";
				cout << endl << image.size();
			}
			else
				cout << "ERROR: Not Continous !" << endl;
		}
	}
}