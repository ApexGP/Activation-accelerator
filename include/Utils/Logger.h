#ifndef LOGGER_H
#define LOGGER_H

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

class Logger
{
public:
    static Logger &getInstance();

    void init(const std::string &log_dir = "log", size_t max_size_kb = 10000);
    void log(const std::string &message);
    void finalize();

    Logger(const Logger &) = delete;
    Logger &operator=(const Logger &) = delete;

private:
    Logger();
    ~Logger();

    void archiveLog(const std::string &source_path, const std::string &archive_path);

    std::string log_dir_;
    std::string log_file_path_;
    std::ofstream log_file_;
    size_t max_size_bytes_;
    size_t current_size_;
    int archive_count_;
};

#define LOG(msg)                              \
    do {                                      \
        std::ostringstream oss;               \
        oss << msg;                           \
        Logger::getInstance().log(oss.str()); \
    } while (0)

#endif  // LOGGER_H
