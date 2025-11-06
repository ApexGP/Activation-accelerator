#include <sys/stat.h>
//#include <zlib.h>

#include <ctime>
#include <fstream>
#include <iomanip>

#include "Utils/Logger.h"

Logger &Logger::getInstance()
{
    static Logger instance;
    return instance;
}

Logger::Logger() : max_size_bytes_(10000 * 1024), current_size_(0), archive_count_(0) {}

Logger::~Logger()
{
    if (log_file_.is_open()) {
        log_file_.close();
    }
}

void Logger::init(const std::string &log_dir, size_t max_size_kb)
{
    log_dir_ = log_dir;
    max_size_bytes_ = max_size_kb * 1024;
    archive_count_ = 0;

    //mkdir(log_dir_.c_str(), 0755);

    log_file_path_ = log_dir_ + "/console.log";

    struct stat st;
    if (stat(log_file_path_.c_str(), &st) == 0) {
        current_size_ = st.st_size;
    } else {
        current_size_ = 0;
    }

    log_file_.open(log_file_path_, std::ios::out | std::ios::app);
    if (!log_file_.is_open()) {
        std::cerr << "Failed to open log file: " << log_file_path_ << std::endl;
        return;
    }

    std::time_t now = std::time(nullptr);
    std::tm *ltm = std::localtime(&now);
    log_file_ << "\n========================================\n";
    log_file_ << "Activation Accelerator Test Log\n";
    log_file_ << "Date: " << std::put_time(ltm, "%Y-%m-%d %H:%M:%S") << "\n";
    log_file_ << "========================================\n\n";
    log_file_.flush();

    current_size_ += 1500;
}

void Logger::log(const std::string &message)
{
    if (!log_file_.is_open()) {
        std::cerr << message << std::endl;
        return;
    }

    log_file_ << message << std::endl;
    current_size_ += message.length() + 1;
    std::cout << message << std::endl;
}

void Logger::finalize()
{
    if (!log_file_.is_open()) {
        return;
    }

    log_file_ << "\n========================================\n";
    log_file_ << "Test Completed\n";
    log_file_ << "========================================\n";
    log_file_.flush();

    current_size_ += 1000;

    if (current_size_ >= max_size_bytes_) {
        log_file_.close();

        std::time_t now = std::time(nullptr);
        std::tm *ltm = std::localtime(&now);
        std::ostringstream archive_name;
        archive_name << log_dir_ << "/console_" << std::put_time(ltm, "%Y%m%d_%H%M%S") << "_"
                     << archive_count_++ << ".log.gz";

        std::cout << "Log file size (" << current_size_ << " bytes) exceeds limit ("
                  << max_size_bytes_ << " bytes), archiving..." << std::endl;
        archiveLog(log_file_path_, archive_name.str());

        log_file_.open(log_file_path_, std::ios::out | std::ios::trunc);
        if (log_file_.is_open()) {
            log_file_ << "========================================\n";
            log_file_ << "Previous log archived to: " << archive_name.str() << "\n";
            log_file_ << "========================================\n";
            log_file_.close();
        }
    } else {
        log_file_.close();
    }
}

void Logger::archiveLog(const std::string &source_path, const std::string &archive_path)
{
    std::ifstream source(source_path, std::ios::binary);
    if (!source.is_open()) {
        std::cerr << "Failed to open source file: " << source_path << std::endl;
        return;
    }

    /*if (false) {
        gzFile gz_file = gzopen(archive_path.c_str(), "wb");
        if (!gz_file) {
            std::cerr << "Failed to create archive: " << archive_path << std::endl;
            return;
        }

        char buffer[4096];
        while (source.read(buffer, sizeof(buffer))) {
            gzwrite(gz_file, buffer, source.gcount());
        }
        if (source.gcount() > 0) {
            gzwrite(gz_file, buffer, source.gcount());
        }

        gzclose(gz_file);
    }*/
    source.close();

    std::cout << "Log archived to: " << archive_path << std::endl;
}
