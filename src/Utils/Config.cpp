#include <cstring>
#include <iostream>

#include "Utils/Config.h"

Config::Config() {}

Config::~Config() {}

bool Config::parse_arguments(int argc, char *argv[])
{
    if (argc > 1) {
        if (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0) {
            print_usage();
            return false;
        }

        int count = (argc - 1 < 8) ? (argc - 1) : 8;
        for (int i = 1; i <= count; i++) {
            int module_id = atoi(argv[i]);
            if (is_valid_module_id(module_id)) {
                modules_to_run.insert(module_id);
            } else {
                std::cerr << "Error: Invalid module ID " << argv[i] << " (valid range: 0-7)\n";
                print_usage();
                return false;
            }
        }
    } else {
        for (int i = 0; i <= 7; i++) {
            modules_to_run.insert(i);
        }
    }

    return true;
}

bool Config::should_run_module(int module_id) const
{
    return modules_to_run.count(module_id) > 0;
}

const char *Config::get_module_name(int module_id)
{
    static const char *module_names[] = {"Eltwise Add", "Eltwise Mult", "Sigmoid",        "SiLU",
                                         "RMS Norm",    "Layer Norm",   "Online Softmax", "GELU"};

    if (module_id >= 0 && module_id <= 7) {
        return module_names[module_id];
    }
    return "Unknown";
}

void Config::print_usage()
{
    std::cout << "Usage: ./bin/main [module_id...]\n";
    std::cout << "\nAvailable modules:\n";
    std::cout << "  0 - Eltwise Add\n";
    std::cout << "  1 - Eltwise Mult\n";
    std::cout << "  2 - Sigmoid\n";
    std::cout << "  3 - SiLU (Swish)\n";
    std::cout << "  4 - RMS Norm\n";
    std::cout << "  5 - Layer Norm\n";
    std::cout << "  6 - Online Softmax\n";
    std::cout << "  7 - GELU\n";
    std::cout << "\nExamples:\n";
    std::cout << "  ./bin/main          # Run all modules\n";
    std::cout << "  ./bin/main 0 1      # Run Eltwise Add and Eltwise Mult\n";
    std::cout << "  ./bin/main 7        # Run GELU only\n";
    std::cout << "\nNote: Maximum 8 module IDs supported\n";
}

bool Config::is_valid_module_id(int module_id) const
{
    return module_id >= 0 && module_id <= 7;
}
