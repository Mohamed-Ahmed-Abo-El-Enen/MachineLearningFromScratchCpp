#include "MNISTDataLoader.h"
#include "Tensor.h"

#include <filesystem>
#include <opencv2/opencv.hpp>

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

// reshape a vector to matrix
vector<vector<double>> reshape(vector<double> vec,  int h, int w)
{
	assert((h * w) == vec.size());
	vector<vector<double> > resultMat = vector<vector<double>>(h, vector<double>(w));
	for (size_t i = 0; i < h; i++)
		for (size_t j = 0; j < w; j++)
			resultMat[i][j] = vec.at((i * h) + j);

	return resultMat;
}

MNISTDataLoader::MNISTDataLoader(unsigned int batchSize, unsigned int rows, unsigned int cols)
{
	this->batchSize = batchSize;
	this->rows = rows;
	this->cols = cols;
	this->numLabels = 10;
}

unique_ptr<Tensor<double>> MNISTDataLoader::GetxTensorShape()
{
	int dims[] = { batchSize, 1, (int)rows, (int)cols };
	return make_unique<Tensor<double>>(4, dims);
}

vector<vector<vector<double>>> MNISTDataLoader::GetxSamples()
{
	return xSample;
}

vector<int> MNISTDataLoader::GetySamples()
{
	return ySample;
}

void MNISTDataLoader::LoadMNISTImage(string filePath,  int nImages)
{
	for (unsigned int i = 0; i < numLabels; i++)
	{
		vector<string> files = ListAllDirectoryFile(filePath + to_string(i));
		int labelSize;
		if (nImages == -1)
			labelSize = files.size();
		else
			labelSize = (nImages / numLabels);

		for (unsigned int k = 0; k < labelSize; k++)
		{
			cv::Mat img = cv::imread(files[k]);
			if (img.empty())
				continue;
			vector<vector<double> > image = vector<vector<double>>(rows, vector<double>(cols));
			for (unsigned int h = 0; h < rows; h++)
				for (unsigned int w = 0; w < cols; w++)
					image[h][w] = (double)(img.at<uchar>(h, w));

			xSample.emplace_back(move(image));
			ySample.emplace_back(i);
		}
	}
	numLabels = ySample.size();
	numImages = ySample.size();
}

void MNISTDataLoader::LoadMNISTCSV(string filePath)
{
	string data(filePath);
	ifstream in(data.c_str());

	if (!in.is_open())
		return;

	string line;
	while (getline(in, line))
	{
		vector<string> lineVec = SpliteLine(line, ",");
		int lableIndex = stoi(lineVec[0]);

		vector<double> xVec;
		for (size_t i = 1; i < lineVec.size(); i++)
		{
			xVec.push_back(stod(lineVec[i]));
		}

		vector<vector<double> > xMat = reshape(xVec, rows, cols);
		xSample.push_back(xMat);
		ySample.push_back(lableIndex);
	}
	numLabels = ySample.size();
	numImages = ySample.size();
}

void MNISTDataLoader::LoadMNISTBytes(const string& imagesPath, const string& labelsPath)
{
	LoadImages(imagesPath);
	LoadLabels(labelsPath);
}

unsigned int MNISTDataLoader::bytesToUInt(const char* bytes)
{
	return((unsigned char)bytes[0] << 24) | ((unsigned char)bytes[1] << 16) | ((unsigned char)bytes[2] << 8) | ((unsigned char)bytes[3]) << 0;
}

void MNISTDataLoader::LoadImages(const string& path)
{
	ifstream file(path, ios::binary | ios::in);
	if (!file)
	{
		cerr << "ERROR: Images File Problem" << endl;
		exit(1);
	}

	file.clear();
	char bytes[4];
	file.read(bytes, 4);
	file.read(bytes, 4);
	numImages = bytesToUInt(bytes);
	file.read(bytes, 4);
	rows = bytesToUInt(bytes);
	file.read(bytes, 4);
	cols = bytesToUInt(bytes);

	xSample.resize(numImages);
	char byte;
	for (int i = 0; i < numImages; i++)
	{
		xSample[i].resize(rows);
		for (int j = 0; j < rows; j++)
		{
			xSample[i][j].resize(cols);
			for (int k = 0; k < cols; k++)
			{
				file.read(&byte, 1);
				xSample[i][j][k] = (unsigned char)(byte & 0xff);
			}
		}
	}
}

void MNISTDataLoader::LoadLabels(const string& path)
{
	ifstream file(path, ios::binary | ios::in);
	if (!file)
	{
		cerr << "ERROR: Labels File Problem" << endl;
		exit(1);
	}

	file.clear();
	char bytes[4];
	file.read(bytes, 4);
	file.read(bytes, 4);
	numImages = bytesToUInt(bytes);
	
	ySample.resize(numImages);
	char byte;
	for (int i = 0; i < numImages; i++)
	{
		file.read(&byte, 1);
		ySample[i] = (byte & 0xff);
	}
}

int MNISTDataLoader::GetNumBatches()
{
	if (numImages % batchSize == 0)
		return numImages / batchSize;
	else
		return (numImages / batchSize) + 1;
}

pair<unique_ptr<Tensor<double>>, vector<int>> MNISTDataLoader::NextBatch()
{
	pair<unique_ptr<Tensor<double>>, vector<int>> xyBatch;
	int remainImages = numImages - batchIdx;
	int size = remainImages > batchSize ? batchSize : remainImages;
	int dims[] = { size, 1, (int)rows, (int)cols };
	unique_ptr<Tensor<double>> tensorImgaes = make_unique<Tensor<double>>(4, dims);
	vector<int> labelsVec;

	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < rows; j++)		
			for (int k = 0; k < cols; k++)			
				tensorImgaes->set(((double)(xSample[batchIdx + i][j][k])) / 255.0, i, 0, j, k);
		labelsVec.push_back(ySample[batchIdx + i]);
	}

	batchIdx += size;
	if (batchIdx == numImages)
		batchIdx = 0;
	xyBatch.first = move(tensorImgaes);
	xyBatch.second = labelsVec;

	return xyBatch;
}
 
void MNISTDataLoader::LoadImage(string filePath)
{
	vector<double> image;
	cv::Mat img = cv::imread(filePath);
	if (img.empty())
		cout << "ERROR: No Image" << endl;
	else
	{
		if (img.isContinuous())
		{
			vector<vector<double> > image = vector<vector<double>>(rows, vector<double>(cols));
			for (unsigned int h = 0; h < rows; h++)
				for (unsigned int w = 0; w < cols; w++)
					image[h][w] = (double)(img.at<uchar>(h, w) / 255.0);
			cout << endl << image.size();
		}
		else
			cout << "ERROR: Not Continous !" << endl;
	}
}
