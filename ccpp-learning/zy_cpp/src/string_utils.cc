/*
 * File Name: string_utils.cc
 * Author: zhouyong03@meituan.com
 * Created Time: 2016-08-31 17:18:30
 */
#include "io.h"
#include "common.h"
#include <string>
#include <string.h>
#include <vector>
#include <sstream>
#include <iostream>
using namespace std;
 
// trim
//std::string& trim(std::string &);
void trim(string &str);
// split 切分字符串 
void split(std::string& s, std::string& delim, std::vector< std::string >& result);
std::vector<std::string> split(const std::string& s, char delim);
// 字符串转化为指针
char* str2char(string &str);

int main(int argc, char* argv[])
{
	string str = "   I love this ";
	admm::io::print_for_streambuf(str);
	string file_name = "../data/example.txt";
	std:: cout << "[INFO] 读取每一行并打印 ... " << flush;
	admm::io::visit_file(file_name);
	
	std::cout << "[INFO] origin str: " << str << ", length: " << str.length() << ", size: " << str.size() << std::endl;
	std::cout << "去空格后 ..." << std::endl;
	trim(str);
	std::cout << "[INFO] after  str: " << str << ", length: " << str.length() << ", size: " << str.size() << std::endl;
	std::cout << "[INFO] 原有的str大小：" << str.length() << std::endl;
	// split
	vector<string> result = dmlc::split(str, ' ');
	string delim = " ";
	//split(str, delim, result);
	std::cout << "result.size: " << result.size() << std::endl;
	vector<string>::iterator it;
	for (it = result.begin(); it != result.end(); ++it)
	{
		std::cout << "(*it): " << (*it) << std::endl;
	}
	const char* text = str.c_str();
	int len = str.length();
	for (int i = 0; i < len; ++i) {
		std::cout << "text[" << i << "]: " << text[i] << std::endl;
	}
	char* chs = str2char(str);
	for (int i = 0; i < len; ++i) {
		std::cout << "chs[" <<  i << "]: " << chs[i] << std::endl;
	}
	const char* cchs = str.c_str();
	string new_str(cchs);
	const char* ccchs = chs;
	std::cout << "new_str: " << new_str << std::endl;
	string nstr(ccchs);
	std::cout << "ccchs: " <<  nstr << std::endl;
 	return 0;
}


// 字符串转化为 char*
char* str2char(string &str) {
	const int len = str.length();
	char* chs = new char[len+1];
	strcpy(chs, str.c_str());	// strcpy in string.h 
	return chs;
}


/** 
 * 去除字符串两头的空格
 */
/* 
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
*/


void trim(std::string &s) 
{
	if (s.empty()) 
	{
		return ;
	}
	s.erase(0, s.find_first_not_of(" "));
	s.erase(s.find_last_not_of(" ")+1);
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

std::vector<std::string> split(const std::string& s, char delim)
{
	std::string item;
	std::istringstream is(s);
	std::vector<std::string> ret;
	while(std::getline(is, item, delim)) {
		ret.push_back(item);
	}
	return ret;
}
