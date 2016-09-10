/*
 * File Name: io.h
 * Author: zhouyong03@meituan.com
 * Created Time: 2016-09-04 20:21:51
 */

#ifndef ADMM_IO_H_
#define ADMM_IO_H_

#include <string>
#include <sstream>
#include <streambuf>
#include <fstream>
#include <iostream>
#include <cstdlib>
using namespace std;

namespace admm {

namespace io {


/*!
 * \brief open file and return legal and valid ifstream
 * \param in istream of file
 * \param file file name
 */
inline std::ifstream & open_file(ifstream & in, const char * file)
{
	in.close();
	in.clear();
	in.open(file);
	return in;
}

/*!
 * \brief fetch file content 
 * \param file file name
 */
string file_content(const char * file)
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

// 

} // namespace io

} // namespace admm

#endif
