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
#include "./base.h"
using namespace std;

namespace admm {

namespace io {

class Base {

private:
	int b_number;

public:
	Base() {}
	Base(int i): b_number(i) {}
	int get_number() {
		return b_number;
	}

	void print() {
		std::cout << b_number << std::endl;
	}
	~Base() {
		std::cout << "Base destructor!" << std::endl;
	}
};

class Derived: public Base {
private:
	int d_number;

public:
	Derived(int i, int j): Base(i), d_number(j) {};
	void print() {
		std::cout << get_number() << " ";
		std::cout << d_number << std::endl;
	}

	~Derived() {
		std::cout << "Derived destructor!" << std::endl;
	}
};

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
string file_content(const char * file);

/*!
 * \brief read file and for each line of file
 * \param file file name
 */
void read_file(const string & file);

/*!
 * \brief streambuf class example.
 *			method: sgetc, snextc 
 *			iterator print buffer content
 * \param str the processed of string
 */
void print_for_streambuf(string &str);

/*!
 * \brief test func
 */ 
void test_print();

} // namespace io

} // namespace admm

#endif
