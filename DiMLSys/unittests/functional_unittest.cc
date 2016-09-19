/*
 * File Name: functional_unittest.cc
 * Author: zhouyong03@meituan.com
 * Created Time: 2016-09-17 22:32:40
 */
 
#include <functional>
#include <iostream>
using namespace std;
 
std::function<int(int)> Gradient;

int TestFunc(int a) 
{
	return a;
}

// Lambda表达式
auto lambda = [](int a)->int{ return a; };

// 仿函数(functor)
class Functor
{
public:
	int operator()(int a) {
		return a;
	}
};

// 类静态函数
class TestClass
{
public:
	int ClassMember(int a) { return a; }

	static int StaticMember(int a) { return a; }
};

int main(int argc, char* argv[]) 
{
	// 普通函数
	Gradient = TestFunc();
	int result = Gradient(10);
	std::cout << "普通函数：" << result << std::endl;

	// Lambda 表达式
	Gradient = lambda;
	result = Gradient(20);
	std::cout << "Lambda表达式：" << result << std::endl;

	// 仿函数
	Functor testFunctor;
	Gradient = testFunctor;
	result = Gradient(40);
	cout << "仿函数：" << result << endl;

	// 类成员函数
	Gradient testObj;
	Gradient = std::bind(&TestClass::ClassMember, testObj, std::placeholders::_1);
	result = Gradient(80);
	cout << "类成员函数：" << result << endl;

	// 类静态函数
	Gradient = TestClass::StaticMember;
	result = Gradient(60);
	cout << "类成员函数：" << result << endl;


	return 0;
}
