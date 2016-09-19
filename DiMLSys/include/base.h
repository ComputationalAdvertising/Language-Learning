/*
 * File Name: base.h
 * Author: zhouyong@staff.sina.com.cn
 * Created Time: 2016-09-10 23:22:17
 */
 
#ifndef ADMM_BASE_H_
#define ADMM_BASE_H_

#include <string>
#include <vector>
#include <functional>
#include <dmlc/logging.h>
#include <dmlc/data.h>

// default_nthreads
#define DEFAULT_NTHREADS 2

namespace admm {

typedef std::vector<std::pair<std::string, std::string>> KWArgs;

typedef float real_t;
typedef uint32_t featid_t;
/*! line of the data */
typedef dmlc::Row<featid_t> Row;	
/*! data block of the data */
typedef dmlc::RowBlockIter<featid_t> DataStore;
/*! gradient */
typedef std::function<real_t(real_t pred, real_t label, featid_t idx, real_t value)> Gradient;
/*! updater */
typedef std::function<real_t(real_t* w, featid_t idx, real_t gradient)> Updater;

/*!
 * \brief generate a new feature index containing the feature group id
 *
 * \param idx the feature index
 * \param gid the feature group id
 * \param nbits number of bits used to encode gid
 *
 * \return the new feature index
 */
inline featid_t EncodeFeatGroupId(featid_t idx, int gid, int nbits) {
	CHECK_GE(gid, 0);
	CHECK_LT(gid, 1 << nbits);
	return (idx << nbits) | gid;
}

/*!
 * \brief get the feature group id from a feature index
 *
 * \param idx the feature index
 * \param nbits number of bits to encode gid
 * 
 * \return the feature group id
 */
inline featid_t DecodeFeatGroupId(featid_t idx, int nbits) {
	return idx % (1 << nbits);
}

} // namespace admm


/*!
 * \brief define io for admm
 */
#ifndef ADMM_IO_H_
#define ADMM_IO_H_
#endif

/*!
 * \brief whether use glog for logging
 */
#ifndef ADMM_USER_GLOG
#define ADMM_USER_GLOG 0
#endif


#endif	// ADMM_BASE_H_
