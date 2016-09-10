/*
 * File Name: test.cc
 * Author: zhouyong03@meituan.com
 * Created Time: 2016-08-31 17:08:06
 */

//#include "file.h"
#include "io.h"
#include <string>
#include <fstream>
#include <cstdlib>
#include <iostream>
using namespace std;

string file_content(const string & file_name) 
{
	string content = "I";
	ifstream in(file_name.c_str());
	if (! in) {
		std::cerr << "unable to file: " << file_name << std::endl;
	}
	string line;
	while (in && std::getline(in, line)) {
		content += line + "\n";
	}
	return content;
}

// 1. file相关
string file_content(const char* file_name) 
{
	ifstream in(file_name, ios::in);
	istreambuf_iterator<char> beg(in), end;
	string strData(beg, end);
	in.close();
	return strData.substr(0, strData.size()-1);			// drop '\n' of last line 
}

// 2. string相关

// 3. class相关

int main(int argc, char* argv[])
{
	string file_name = "../data/example.txt";
	string file_cont = admm::io::file_content(file_name.c_str());
	//string file_cont = file_content(file_name);
	std::cout << "file_name: " << file_name << std::endl;
	std::cout << "file_cont:\n " << file_cont << std::endl;
	return 0;
}
