/*
 * File Name: admm_learner.cpp
 * Author: zhouyongsdzh@foxmail.com
 * Created Time: 2016-08-21 17:06:12
 */
 
#include "io.h"
#include "hello.h"
#include <iostream>
using namespace std;
 
int main(int argc, char* argv[]) {
	admm::io::Base a(2);
	a.print();
	admm::io::Derived b(30, 40);
	b.print();
	b.Base::print();

	std::cout << "argc: " << argc << std::endl;
	std::cout << "argv: " << argv[0] << std::endl;
	for (auto i = 0; i < 10; ++i) {
		std::cout << "i: " << i << "\t I love this world!!!!" << std::endl;
	}
	HelloFunc();
	return 0;
}
