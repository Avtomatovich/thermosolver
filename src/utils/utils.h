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
    const std::string diag_file = "./data/diag_data.csv";

    void write_file(const std::string& filename, 
                    const std::string& line, 
                    std::ios_base::openmode mode);

    void write_head();

    // stats funcs
    void print_stats(const std::string& file, const Stats& stats, 
                     double t, double bytes, double flops);

    void solve_stats(const Stats& stats);

    void diag_stats(const Stats& stats);

    inline void write_stats(const Stats& stats) {
        solve_stats(stats);
        diag_stats(stats);
    }

    void write_diag(const Stats& stats);

}
