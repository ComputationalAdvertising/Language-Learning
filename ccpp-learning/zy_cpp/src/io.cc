/*
 * File Name: io.cc
 * Author: zhouyong@staff.sina.com.cn
 * Created Time: 2016-09-11 23:12:39
 */
 
#include "io.h"
using namespace admm::io; 

void admm::io::test_print() {
	std::cout << "admm::io::test_print() ... " << std::endl;
}

/*!
 * \brief read file and for each line of file
 * \param file file name
 */
void read_file(const string & file) 
{
	std::ifstream in;
	std::ifstream & instream = open_file(in, file.c_str());
	if (! instream) {
		std::cerr << "[ERROR] Unable to open file: " << file << std::endl;
		return ;
		//return EXIT_FAILURE;
	}
	string line;
	int num_line = 0;
	while (instream && std::getline(instream, line)) {
		std::cout << ++num_line << ": " << line << std::endl;
	}
	return ;
}

/*!
 * \brief streambuf class example.
 *			method: sgetc, snextc 
 *			iterator print buffer content
 * \param str the processed of string
 */
void print_for_streambuf(string &str)
{
	stringstream ss;
	ss << str;
	streambuf *pbuf = ss.rdbuf();
	std::cout << "content: ";
	while (pbuf->snextc() != EOF) {
		char ch = pbuf->sgetc();
		std::cout << ch << "\t";
	}
	std::cout << std::endl;
}

string admm::io::file_content(const char * file)
{
	ifstream in(file, ifstream::in);
	if (! in.good()) {
		std::cerr << "[ERROR] unable to open file: " << file << std::endl;
		exit(-1);
	}
	istreambuf_iterator<char> beg(in), end;
	string strData(beg, end);
	in.close();
	return strData.substr(0, strData.size() - 1);
}
