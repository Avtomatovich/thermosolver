/**
 * Thermosolver
 * 
 * Name: utils.h
 * 
 * Author: Samson Tsegai
 */

#pragma once

#include <string>
#include <iostream>
#include "stats.h"

namespace Utils {

    // NOTE: navigate out of build folder to find csv files
    const std::string solve_file = "./data/solve_data.csv";
    const std::string res_file = "./data/res_data.csv";
    const std::string conv_file = "./data/conv_data.csv";

    void write_file(const std::string& filename, 
                    const std::string& line, 
                    std::ios_base::openmode mode);

    void write_head();

    // stats funcs
    void print_stats(const std::string& file, int N, int steps, 
                     double t, double bytes, double flops);

    void solve_stats(const Stats& stats);

    void res_stats(const Stats& stats);

    inline void write_stats(const Stats& stats) {
        solve_stats(stats);
        res_stats(stats);
    }

    void write_conv(const Stats& stats);

}
