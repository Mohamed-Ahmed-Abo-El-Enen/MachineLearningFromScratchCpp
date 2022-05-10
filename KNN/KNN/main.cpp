#include "ReadCSV.h"
#include <algorithm>
#include <map>

using namespace std;

map<string, int> class_map;

struct Sample
{
private:
    double distance;

public:
    vector<double> features;
    int val;

    Sample()
    {
        val = -1;
        distance = 0;
    }

    Sample(vector<double> _features, int _val)
    {
        features = _features;
        val = _val;
        distance = 0;
    }

    double GetDistance()
    {
        return distance;
    }

    double CalculateDistance(vector<double> _features)
    {
        for (size_t i = 0; i < features.size(); i++)
            distance += (features[i] - _features[i]) * (features[i] - _features[i]);

        distance = sqrt(distance);
        return distance;
    }

    bool static comparison(Sample a, Sample b)
    {
        return (a.GetDistance() < b.GetDistance());
    }
};

int ClassifyPoint(vector<Sample> arr, int nbClasses, int k, Sample point)
{
    vector<int> classFreq;
    for (int i = 0; i < nbClasses; i++)
        classFreq.push_back(0);

    for (Sample&p : arr)
        p.CalculateDistance(point.features);

    sort(arr.begin(), arr.end(), Sample::comparison);

    for (int i = 0; i < k; i++)
        for (int j = 0; j < nbClasses; j++)
            if (arr[i].val == j)
                classFreq[j] += 1;

    int prediced_class = std::max_element(classFreq.begin(), classFreq.end()) - classFreq.begin();

    return prediced_class;
}



vector<Sample> ConvertCSVSamples(vector<CSVSample> csvArr, int yIndex)
{
    for (int i = 0; i < csvArr.size(); i++)     
        class_map.insert({ csvArr[i].values[yIndex], class_map.size() });    

    vector<Sample> points;
    for (size_t i = 0; i < csvArr.size(); i++)
    {
        Sample pt(slicing(csvArr[i].values, 2, 4), class_map.at(csvArr[i].values[yIndex]));
        points.push_back(pt);
    }
    return points;
}

int main()
{
    int k = 3;
    int nbClasses = 2;

    string csv_file_path = "../Dataset/Mall_data.csv";
    vector<CSVSample> csvArr = readcsv(csv_file_path);

    int yIndex = 1;
    vector<Sample> points = ConvertCSVSamples(csvArr, yIndex);

    Sample p;
    p.features = { 23, 18, 94 };
    p.val = class_map.at("Male");

    printf("The value classified to unknown point"
        " is %d.\n", ClassifyPoint(points, nbClasses, k, p));
    return 0;
}