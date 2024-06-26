#pragma once

#include <iostream>
#include <sstream>
#include <string>

namespace richdem {

enum class LogFlag {
  ALG_NAME,
  CITATION,
  CONFIG,
  DEBUG,
  ERROR_,
  MEM_USE,
  MISC,
  PROGRESS,
  TIME_USE,
  WARN
};

#ifdef RICHDEM_LOGGING
void RDLOGfunc(const LogFlag flag, const char* file, const char* func, unsigned line, const std::string& msg);
#endif

class StreamLogger {
 private:
  LogFlag     flag;
  const char* file;
  const char* func;
  unsigned    line;
  std::ostringstream ss;
 public:
  StreamLogger(LogFlag flag0, const char* file0, const char* func0, unsigned line0) : flag(flag0), file(file0), func(func0), line(line0) {}

  ~StreamLogger() noexcept(false) {
    (void)flag; // Suppress unused variable warning
    (void)file; // Suppress unused variable warning
    (void)func; // Suppress unused variable warning
    (void)line; // Suppress unused variable warning
    #ifdef RICHDEM_LOGGING
      RDLOGfunc(flag, file, func, line, ss.str());
    #endif
  }

  template<typename T>
  StreamLogger& operator<<(const T& t){
    (void)t; // Suppress unused variable warning
    #ifdef RICHDEM_LOGGING
      ss << t;
    #endif
    return *this;
  }

  // std::endl and other iomanip:s.
  StreamLogger& operator<<(std::ostream&(*f)(std::ostream&)){
    (void)f; // Suppress unused variable warning
    #ifdef RICHDEM_LOGGING
      f(ss);
    #endif
    return *this;
  }

};

#define RDLOG(flag)    StreamLogger(flag,              __FILE__, __func__, __LINE__)
#define RDLOG_ALG_NAME StreamLogger(LogFlag::ALG_NAME, __FILE__, __func__, __LINE__)
#define RDLOG_CITATION StreamLogger(LogFlag::CITATION, __FILE__, __func__, __LINE__)
#define RDLOG_CONFIG   StreamLogger(LogFlag::CONFIG,   __FILE__, __func__, __LINE__)
#define RDLOG_DEBUG    StreamLogger(LogFlag::DEBUG,    __FILE__, __func__, __LINE__)
#define RDLOG_ERROR    StreamLogger(LogFlag::ERROR_,   __FILE__, __func__, __LINE__)
#define RDLOG_MEM_USE  StreamLogger(LogFlag::MEM_USE,  __FILE__, __func__, __LINE__)
#define RDLOG_MISC     StreamLogger(LogFlag::MISC,     __FILE__, __func__, __LINE__)
#define RDLOG_PROGRESS StreamLogger(LogFlag::PROGRESS, __FILE__, __func__, __LINE__)
#define RDLOG_TIME_USE StreamLogger(LogFlag::TIME_USE, __FILE__, __func__, __LINE__)
#define RDLOG_WARN     StreamLogger(LogFlag::WARN,     __FILE__, __func__, __LINE__)

}
