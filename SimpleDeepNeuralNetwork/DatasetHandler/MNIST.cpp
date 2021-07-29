#include"MNIST.h"

using namespace std;

MNIST::MNIST()
{}

unsigned char* MNIST::LoadMnistData(const string filename, int& row, int& col, int& num)
{
    ifstream file(filename.c_str(), ios::binary);    //Open the file in binary form
    if (!file)return nullptr;
    unsigned char buffer[4];
    row = 0; col = 0; num = 0;
    file.seekg(0, ios::beg);

    file.read((char*)&buffer, 4);
    int type = 0;
    for (int i = 3, k = 1; i >= 0; --i, k *= 256)
        type += k * buffer[i];
    if (type != 2051) 
    {
        return nullptr;
    }

    file.read((char*)&buffer, 4);
    for (int i = 3, k = 1; i >= 0; --i, k *= 256)
        num += k * buffer[i];

    file.read((char*)buffer, 4);
    for (int i = 3, k = 1; i >= 0; --i, k *= 256)
        row += k * buffer[i];

    file.read((char*)buffer, 4);
    for (int i = 3, k = 1; i >= 0; --i, k *= 256)
        col += k * buffer[i];

    unsigned char* pData = new unsigned char[num * row * col];
    int size = row * col * num;
    file.read((char*)(pData), size);

    return pData;
}

unsigned char* MNIST::LoadMnistLabel(const string filename, int& num) 
{
    ifstream file(filename.c_str(), ios::binary);   //Open the file in binary form
    if (!file)return nullptr;
    unsigned char buffer[4];
    num = 0;
    file.seekg(0, ios::beg);

    file.read((char*)&buffer, 4);
    int type = 0;
    for (int i = 3, k = 1; i >= 0; --i, k *= 256)
        type += k * buffer[i];

    if (type != 2049) 
    {
        return nullptr;
    }

    file.read((char*)&buffer, 4);
    for (int i = 3, k = 1; i >= 0; --i, k *= 256)
        num += k * buffer[i];
    unsigned char* pData = new unsigned char[num];
    file.read((char*)(pData), num);

    return pData;
}

void MNIST::GetMNISTDataset(string imagesFilePath, string lableFilePath, vector<vector<double>>& samplesFeature, vector<vector<double>>& samplesLabel)
{
    int num, numl, row, col;
    unsigned char* pData = LoadMnistData(imagesFilePath, row, col, num);
    unsigned char* pLabel = LoadMnistLabel(lableFilePath, numl);
    int imgsize = row * col;
    unsigned char* pImg = pData;

    int id_img = 0;
    while (id_img < num)
    {
        pImg = pData + id_img * imgsize;

        vector<double> sample;
        for (int i = 0; i < imgsize; ++i)
        {
            sample.push_back((double)pImg[i] / 255.0);
        }
        samplesFeature.push_back(sample);

        std::vector<double> sampleLabel(10);
        sampleLabel[pLabel[id_img]] = 1.0;

        samplesLabel.push_back(sampleLabel);
        id_img++;
    }
}

void MNIST::GenerateMNISTDataFormat(vector<vector<double>>& samplesFeature, vector<vector<double>>& samplesLabel)
{
    srand(time(NULL));
    assert(samplesFeature.size() == samplesLabel.size());
    for (int i = 0; i < samplesFeature.size(); i++)
    {
        cout << "in: ";
        for (int j = 0; j < samplesFeature[i].size(); j++)
        {
            cout << samplesFeature[i][j] << " ";
        }
        cout << endl;

        cout << "out: ";
        for (int j = 0; j < samplesLabel[i].size(); j++)
        {
            cout << samplesLabel[i][j] << " ";
        }
        cout << endl;
    }
}
