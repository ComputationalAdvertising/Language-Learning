/*
 * File Name: file.cc
 * Author: zhouyong03@meituan.com
 * Created Time: 2016-08-22 15:28:14
 */
 
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <string>
using namespace std;
 

string fileData(const char* fileName) 
{
	ifstream in(fileName, ios::in);
	istreambuf_iterator<char> beg(in), end;
	string strData(beg, end);
	in.close();
	return strData;
}

int main(int argc, char* argv[]) 
{
	char buffer[256];
	ifstream exampleFile("../../data/example.txt");
	if (! exampleFile.is_open()) {
		std::cout << "Error Opening File. " << std::endl;
		exit(1);
	}
	int count = 0;
	while (! exampleFile.eof()) {
		count++;
		exampleFile.getline(buffer, 100);
		std::cout << buffer << std::endl;
	}
	std::cout << "lines: " << count << std::endl;

	string content = fileData("../../data/example.txt");
	std::cout << "content: " << content << std::endl;
	return 0;
}
