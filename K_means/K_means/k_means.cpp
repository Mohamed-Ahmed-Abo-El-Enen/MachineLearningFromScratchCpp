#include<ctime>
#include<fstream>
#include<iostream>
#include<sstream>
#include<vector>
#include <algorithm>

using namespace std;

struct Point
{
	vector<string> features;
	int cluster;
	double minDist;

	Point():
		cluster(-1),
		minDist(numeric_limits<double>::max()){}
	
	Point(vector<string> _features) :
		features(_features),
		cluster(-1),
		minDist(numeric_limits<double>::max()) {}

	double distance(vector<string> _features)
	{			
		double dist=0;
		for (size_t i = 0; i < features.size(); i++)		
			dist += (stod(features[i]) - stod(_features[i])) * (stod(features[i]) - stod(_features[i]));
		
		dist = sqrt(dist);
		return dist;
	}
	
};

ostream& operator<<(ostream& os, const vector<string>& features)
{
	string line = "";
	for (int i = 0; i < features.size() - 1; i++)
		line = features[i] + ',';

	line = features[features.size() - 1];

	os << line;
	return os;
}

vector<string> lineSpliter(string line, string delimiter)
{
	vector<string> values;
	size_t pos = 0;
	std::string token;
	while ((pos = line.find(delimiter)) != std::string::npos)
	{
		token = line.substr(0, pos);
		values.push_back(token);
		line.erase(0, pos + delimiter.length());
	}
	values.push_back(line);
	return values;
}

vector<string> slicing(vector<string> arr, int X, int Y)
{
	vector<string> res;
	for (size_t i = X; i < Y; i++)
		res.push_back(arr[i]);
	return res;
}

vector<Point> readcsv(string csv_file_path)
{
	vector<Point> points;
	string line;
	ifstream file(csv_file_path);
	getline(file, line);
	string delimiter = ",";
	while (getline(file, line))
	{
		Point pt(slicing(lineSpliter(line, delimiter), 2, 4));
		points.push_back(pt);
	}
	return points;
}

void kMeansClustering(vector<Point>* points, int epochs, int k)
{
	int n = points->size();
	int numFeat = points->at(0).features.size();
	vector<Point> centroids;
	srand(time(0));
	for (int i = 0; i < k; i++)
	{
		centroids.push_back(points->at(rand() % n));
	}

	for (int i = 0; i < epochs; i++)
	{
		for (vector<Point>::iterator c = begin(centroids); c != end(centroids); c++)
		{
			int clusterId = c - begin(centroids);
			for (vector<Point>::iterator it = points->begin(); it != points->end(); it++)
			{
				Point p = *it;
				double dist = c->distance(p.features);
				if (dist < p.minDist)
				{
					p.minDist = dist;
					p.cluster = clusterId;
				}
				*it = p;
			}
		}

		// Create vectors to keep track of data needed to compute means
		vector<int> nPoints;
		vector<vector<double>> sumFeatures;
		
		for (int j = 0; j < k; j++)
		{
			nPoints.push_back(0);
			vector<double> resetVec(numFeat);
			fill(resetVec.begin(), resetVec.end(), 0);
			sumFeatures.push_back(resetVec);
		}

		// Iterate over points to append data to centroids
		for (vector<Point>::iterator it = points->begin(); it != points->end(); it++)
		{
			int clusterId = it->cluster;
			nPoints[clusterId] += 1;

			for (size_t k = 0; k < numFeat; k++)
				sumFeatures[clusterId][k] += stod(it->features[k]);
			

			it->minDist = numeric_limits<double>::max();
		}

		// Compute the new centroids
		for (vector<Point>::iterator c = begin(centroids); c != end(centroids); c++)
		{
			int clusterId = c - begin(centroids);
			for (size_t k = 0; k < numFeat; k++)
				c->features[k] = to_string(sumFeatures[clusterId][k] / nPoints[clusterId]);
		}
	}
}

void WritecsvFile(vector<Point> *points, string csv_file_path)
{
	ofstream file;
	file.open(csv_file_path);
	string line = "";
	int numFeat = points->at(0).features.size();

	for (size_t k = 0; k < numFeat; k++)
		line += "Feat_" + to_string(k+1)+',';

	line += "Cluster";
	file << line << endl;
	for (vector<Point>::iterator it = points->begin(); it != points->end(); it++)
	{
		line = "";
		for (size_t k = 0; k < numFeat; k++)
			line += it->features[k] + ',';
		line += to_string(it->cluster);
		file << line << endl;
	}
	file.close();
}

int main()
{
	//vector<string> fet1{ "0.0", "0.0" };
	//Point p1 = Point(fet1);
	//cout << p1.features << endl;

	//vector<string> fet2{ "3.0", "4.0" };
	//Point p2 = Point(fet2);
	//cout << p1.distance(p2.features) << endl;


	string csv_file_path = "../Dataset/Mall_data.csv";
	vector<Point> points = readcsv(csv_file_path);
	kMeansClustering(&points, 10, 3);

	csv_file_path = "../Dataset/results.csv";
	WritecsvFile(&points, csv_file_path);
	return 0;
}