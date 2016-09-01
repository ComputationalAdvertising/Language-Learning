/*
 * File Name: string_utils.cc
 * Author: zhouyong03@meituan.com
 * Created Time: 2016-08-31 17:18:30
 */

#include <string>
#include <vector>
#include <iostream>
using namespace std;
 
// trim
std::string& trim(std::string &);
// split 切分字符串 
void split(std::string& s, std::string& delim, std::vector< std::string >& result);

int main(int argc, char* argv[])
{
	string str = "   I love this ";
	std::cout << "[INFO] origin str: " << str << ", length: " << str.length() << ", size: " << str.size() << std::endl;
	std::cout << "去空格后 ..." << std::endl;
	std::cout << "[INFO] after  str: " << trim(str) << ", length: " << str.length() << ", size: " << str.size() << std::endl;
	// split
	vector<string> result;
	string delim = " ";
	split(str, delim, result);
	std::cout << "result.size: " << result.size() << std::endl;
	vector<string>::iterator it;
	for (it = result.begin(); it != result.end(); ++it)
	{
		std::cout << "(*it): " << (*it) << std::endl;
	}
	const char* text = str.c_str();
	int len = sizeof(text)/sizeof(char*);
	for (int i = 0; i < len; ++i) {
		std::cout << "text[i]: " << text[i] << std::endl;
	}
	std::cout << "text[0]: " << text[0] << std::endl;
	return 0;
}

/** 
 * 去除字符串两头的空格
 */
std::string& trim(std::string &s) 
{
	if (s.empty()) 
	{
		return s;
	}
	s.erase(0, s.find_first_not_of(" "));
	s.erase(s.find_last_not_of(" ")+1);
	return s;
}

/**
 * 按照delim切分字符串
 */
void split(std::string& s, std::string& delim, std::vector< std::string >& result)
{
	size_t last = 0;
	size_t index = s.find_first_of(delim, last);
	while (index != std::string::npos)
	{
		result.push_back(s.substr(last, index-last));		// substr(index, length)
		last = index + 1;
		index = s.find_first_of(delim, last);
	}
	if (index-last > 0)
	{
		result.push_back(s.substr(last, index-last));
	}
}
