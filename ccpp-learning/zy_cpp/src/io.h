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
#include <iostream>
using namespace std;

namespace admm {

/*!
 * \brief streambuf class example
 */
// method: sgetc, snextc 
// iterator print buffer content
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

}

#endif
