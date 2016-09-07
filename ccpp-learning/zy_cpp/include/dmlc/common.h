/*
 * File Name: common.h
 * Author: zhouyong03@meituan.com
 * Created Time: 2016-09-03 08:31:11
 */
#ifndef DMLC_COMMON_H_
#define DMLC_COMMON_H_

#include <vector>
#include <string>
#include <sstream>

namespace dmlc {
	/*!
	 * \brief split a string by delimiter
	 * \param s string to be splitted.
	 * \param delim The delimiter
	 * \return a splitted vector of string
	 */
	inline std::vector<std::string> Split(const std::string& s, char delim) {
		std::string item;
		std::istringstream is(s);
		std::vector<std::string> ret;
		while(std::getline(is, item, delim)) {
			ret.push_back(item);
		}
		return ret;
	}
}	// namespace dmlc

#endif	// DMLC_COMMON_H_
