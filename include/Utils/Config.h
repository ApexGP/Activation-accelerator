#ifndef CONFIG_H
#define CONFIG_H

#include <set>

// Module configuration
#define CONFIG_ELTWISE_ADD 0
#define CONFIG_ELTWISE_MULT 1
#define CONFIG_SIGMOID 2
#define CONFIG_SILU 3
#define CONFIG_RMS_NORM 4
#define CONFIG_LAYER_NORM 5
#define CONFIG_ONLINE_SOFTMAX 6
#define CONFIG_GELU 7

class Config
{
public:
    Config();
    ~Config();

    bool parse_arguments(int argc, char *argv[]);

    const std::set<int> &get_modules_to_run() const
    {
        return modules_to_run;
    }

    bool should_run_module(int module_id) const;
    static const char *get_module_name(int module_id);
    static void print_usage();

private:
    std::set<int> modules_to_run;
    bool is_valid_module_id(int module_id) const;
};

#endif  // CONFIG_H
