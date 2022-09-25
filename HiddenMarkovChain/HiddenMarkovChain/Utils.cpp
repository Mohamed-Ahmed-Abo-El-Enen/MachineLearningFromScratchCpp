#include "Utils.h"

string Utils::trim(const string& str)
{
	size_t first = str.find_first_not_of(' ');
	if (string::npos == first)
	{
		return str;
	}
	size_t last = str.find_last_not_of(' ');
	return str.substr(first, (last - first + 1));
}

string Utils::Lower(string& str)
{
	transform(str.begin(), str.end(), str.begin(), ::tolower);
	return str; 
}

string Utils::RemoveNumbers(string& str)
{
	str.erase(remove_if(str.begin(), str.end(), [](char c) { return isdigit(c); }), str.end());
	return str;
}

string Utils::RemovePunctuations(string& str)
{
	auto it = remove_if(str.begin(), str.end(), [](char const& c) {
		return ispunct(c);
		});

	str.erase(it, str.end());
	return str;
}


vector<string> Utils::split(string line, string delimiter, bool removePunctuations)
{
	vector<string> values;
	size_t pos = 0;
	string token;
	while ((pos = line.find(delimiter)) != string::npos)
	{
		if (removePunctuations)
			line = RemovePunctuations(line);

		token = line.substr(0, pos);
		token = Lower(token);
		token = RemoveNumbers(token);
		values.push_back(token);
		line.erase(0, pos + delimiter.length());
		line = trim(line);
	}
	line = trim(line);	
	line = RemoveNumbers(line);
	line = Lower(line);
	values.push_back(line);
	return values;
}

vector<string> Utils::ReadFile(string filePath, string delimiter)
{
    vector<string> lines;
    fstream file;
    file.open(filePath, ios::in);
    if (file.is_open())
    { 
        string line;
		while (getline(file, line))
		{
			vector<string> lineVec = Utils::split(line, delimiter);
			line = lineVec[lineVec.size() - 1];
			lines.push_back(line);
		}
        file.close();
    }
    return lines;
}