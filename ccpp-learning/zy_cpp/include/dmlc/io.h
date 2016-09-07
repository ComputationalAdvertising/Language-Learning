/*
 * File Name: io.h
 * Author: zhouyong03@meituan.com
 * Created Time: 2016-09-03 09:19:35
 */

#ifndef DMLC_IO_H_
#define DMLC_IO_H_

#include <cstdio>
#include <string>
#include <vector>
#include <istream>
#include <ostream>
#include <streambuf>
 

// include uint64_t only to make io standalone
#ifdef _MSC_VER
/*! \brief uint64 */
typedef unsigned __int64 uint64_t;
else
#include <inttypes.h>
#endif

namespace dmlc {
	
	class Stream {
		public:
			/*!
			 * \brief reads data from a stream
			 * \param ptr pointer to a memory buffer
			 * \param size block size
			 * \retturn the size of data record
			 */
			virtual size_t read(void *ptr, size_t size) = 0;
			/*!
			 * \brief writes data to a stream
			 * \param ptr pointer to a memory buffer
			 * \param size block size
			 */
			virtual void write(const void *ptr, size_t size) = 0;
			// virtual destructor
			virtual ~Stream(void) {}
			/*!
			 * \brief generic factory function
			 * create a stream, the stream will close the underlying file upon deletion
			 *
			 * \param uri the uri of the input currently we support. for example
			 *			  hdfs://, s3://, and file:// by default file:// will be used
			 * \param flag can be "w", "r", "a"
			 * \param allow_null whether NULL can be returned, or directly report error
			 * \return the creared stream, can be NULL when allow_null == true and file do not exist
			 */
			static Stream *create(const char* uri, 
								  const char* const flag, 
								  bool allow_null = false);

	class SeekStream : public Stream {
		public:
			// virtual destructor
			virtual ~SeekStream(void) {}
			/*! \brief seek to certain position of the file. */
			virtual void seek(size_t pos) = 0;
			/*! \brief tell the position of the stream. */
			virtual size_t tell(void) = 0;
			/*!
			 * \brief generic factory function.
			 * create a SeekStream for read only,
			 * the stream will close the underlying file upon deletion 
			 * error will be reported and the system will exit when create failed 
			 * \param uri the uri of the input currently we support
			 *			  hdfs://, s3://, and file:// by default file:// will be used
			 * \param allow null whether NULL can be returned, or directly report error
			 * \return the created stream, can be NULL when allow_null == true and file do not exist
			 */
			static SeekStream *createForRead(const char *uri, bool allow_null = false);
	};

	class Serializable {
		public:

			virtual ~Serializable() {}
			/*!
			 * \brief load the model from a stream
			 * \param fi stream where to load model from
			 */
			virtual void load(Stream *fi) = 0;
			/*!
			 * \brief saves the model to a stream
			 * \param fo stream where to save the model to
			 */
			virtual void save(Stream *fo) const = 0;
	};

	class InputSplit {
		public:
			/*! \brief a blob of memory region. */
			struct Blob {
				/*! \brief points to start of the memory region. */
				void *dptr;
				/*! \brief size of the memory region. */
				size_t size;
			};
			
			virtual void hintChunkSize(size_t chunk_size) {} 
			/*! \brief reset the position of InputSplit to beginning. */
			virtual void beforeFirst(void) = 0;

			/*!
			 * \brief get the next record, then returning value is valid until
			 *		  next call to nextRecord or nextChunk caller can modify
			 *		  then memory content of out_rec
			 *
			 *		  For text, out_rec contains a single line
			 *		  For recordio, out_rec contains one record(with header striped)
			 *
			 * \param out_rec used to store the result
			 * \return true if we can successfully get next record 
			 *		   false if we reached end of split
			 * \soa InputSplit::create for definition of record
			 */
			virtual bool nextRecord(Blob *out_rec) = 0;
			
			/*!
			 * \brief get a chunk of memort that can contain multiple record
			 */
			virtual bool nextChunk(Blob *out_chunk) = 0;
			/*! \brief virtual destructor. */
			virtual ~InputSplit(void) {}
			/*!
			 * \brief reset the Input split to a certain part id,
			 * The InputSplit will be pointed to the head of the new specified segment.
			 * This feature may not be supported by every implementation of InputSplit.
			 * \param part_index The part id of the new input.
			 * \param num_parts The total number of parts.
			 */
			virtual void resetPartition(unsigned part_index, unsigned num_parts) = 0;
			/*! \brief factory function:
			 *		   create input split given a uri
			 *	\param uri the uri of the input, can contain hdfs prefix.
			 *	\param part_index the part id of current input
			 *	\param num_parts total number of splits
			 *	\param type type of record
			 *		List of possible types: "text", "recordio"
			 *			- "text":
			 *				text file, each line is treated as a record
			 *				input split will split on '\\n' or '\\r'
			 *			- "recordio":
			 *				binary recordio file, see recordio.h
			 *	\return a new input split
			 *	\sa InputSplit::Type
			 */
			static InputSplit* create(const char *uri, 
					unsigned part_index, 
					unsigned num_parts, 
					const char* type);
	};

	/*!
	 * \brief a std::ostream class that can wrap Stream objects,
	 *		  can use ostream with that output to underlying Stream
	 *
	 *	Usage example:
	 *	\code
	 *		Stream *fs = Stream::create("hdfs://test.txt", "w");
	 *		dmlc::ostream os(fs);
	 *		os << "hello world" << std::endl;
	 *		delete fs;
	 *	\endcode
	 */
	class ostream : public std::basic_ostream<char> {
		public:
			/*!
			 * \brief construct std::ostream type
			 * \param stream the Stream output to be used
			 * \param buffer_size internal streambuf size
			 */
			explicit ostream(Stream *stream, size_t buffer_size = (1 << 10)) 
				: std::basic_stream<char>(NULL), buf_(buffer_size) {
				this->set_stream(stream);
			}
			
			virtual ~ostream() {
				buf_.pubsync();
			}
			/*!
			 * \brief set internal stream to be stream, reset status
			 * \param stream new stream as output
			 */
			inline void set_stream(Stream *stream) {
				buf_.set_stream(stream);
				this->rdbuf(&buf_);
			}

			inline size_t bytes_written(void) const {
				return buf_.bytes_out();
			}

		private:
			// internal streambuf
			class OutBuf : public std::streambuf {
				public:
					explicit OutBuf(size_t buffer_size) : stream_(NULL), buffer_(buffer_size), bytes_out_(0) {
						if (buffer_size == 0) {
							buffer_.resize(2);
						}
					}
					// set stream to the buffer
					inline void set_stream(Stream *stream);

					inline size_t bytes_out() const {
						return bytes_out_;
					}

				private:
					/*! \brief internal stream by StreamBuf. */
					Stream *stream_;
					/*! \brief internal buffer. */
					std::vector<char> buffer_;
					/*! \brief numbers of bytes written so far. */
					size_t bytes_out_;
					// override sync
					inline int_type sync(void);
					// override overflow
					inline int_type overflow(int c);
			};
			/*! \brief buffer of the stream. */
			OutBuf buf_;
	} // dmlc::ostream

	class istream : public std::basic_istream<char> {
		public:
			/*!
			 * \brief construct std::ostream type
			 * \param stream the Stream output to be used
			 * \param buffer_size internal buffer size
			 */
			explicit istream(Stream *stream, 
							 size_t buffer_size = (1 << 10)) : 
				std::basic_istream<char>(NULL), buf_(buffer_size) {
				this->set_stream(stream);
			}
			virtual ~istream() {}

			/*!
			 * \brief set internal stream to be stream, reset status
			 * \param stream new stream as output
			 */
			inline void set_stream(Stream *stream) {
				buf_.set_stream(stream);
				this->rdbuf(&buf_);
			}
			/*! \return how many bytes we read so far. */
			inline size_t bytes_read(void) const {
				return buf_.bytes_read();
			}

		private:
			// internal streambuf
			class InBuf : public std::streambuf {
				public:
					explicit InBuf(size_t buffer_size) : 
						stream_(NULL), bytes_read_(0), buffer_(buffer_size) {
						if (buffer_size == 0) 
							buffer_.resize(2);
					}
					// set stream to the buffer
					inline void set_stream(Stream *stream);
					// return how many bytes read so far.
					inline size_t bytes_read(void) const {
						return bytes_read_;
					}

				private:
					/*! \brief internal stream by StreamBuf */
					Stream *stream_;
					/*! \brief houw many bytes we read so far. */
					size_t bytes_read_;
					/*! \brief internal buffer. */
					std::vector<char> buffer_;
					// override underflow
					inline int_type underflow();
			};
			/*! \brief input buffer. */
			InBuf buf_;
	};
} 	// namespace dmlc

#incldue "./serializer.h"

namspace dmlc {
	// implementations of inline functions
	template<typename T>
	inline void Stream::write(const T &data) {
		serializer::handle<T>::write(this, data);
	}

	template<typename T>
	inline bool Stream::read(T *out_data) {
		return serializer::handle<T>::read(this, out_data);
	}

	// implementations for ostream
	inline void ostream::OutBuf::set_stream(Stream *stream) {
		if (stream_ != NULL) 
			this->pubsync();
		this->stream_ = stream;
		this->setp(&buffer_[0], &buffer_[0] + buffer_size()-1);
	}

	inline int ostream::OutBuf::sync(void) {
		if (stream_ == NULL)
			return -1;
		std::ptrdiff_t n = pptr() - pbase();
		stream_->write(pbase(), n);
		this->pbump(-static_cast<int>(n));
		bytes_out_ += n;
		return 0;
	}

	intlint int ostream::OutBuf::overflow(int c) {
		*(this->pptr()) = c;
		std::ptrdiff_t n = pptr() - pbase();
		this->pbump(-static_cast<int>(n));
		if (c == EOF) {
			stream_->write(pbase(), n);
			bytes_out_ += n;
		} else {
			stream_->write(pbase(), n + 1);
			bytes_out_ += n + 1;
		}
		return c;
	}

	// implementations for istream
	inline void istream::InBuf::set_stream(Stream *stream) {
		stream_ = stream;
		this->setg(&buffer_[0], &buffer_[0], &buffer_[0]);
	}

	inline int istream::InBuf::underflow() {
		char *bhead = &buffer_[0];
		if (this->gptr() == this->egptr()) {
			size_t sz = stream_->read(bhead, buffer_.size());
			this->setg(bhead, bhead, bhead + sz);
			bytes_read_ += sz;
		}
		if (this->gptr() == this->egptr()) {
			return traits_type::eof();
		} else {
			return traits_type::to_int_type(*gptr());
		}
	}
} // namespace dmlc
#endif // DMLC_IO_H_
