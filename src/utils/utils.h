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
#include "method.h"

#define WARP_SIZE 64 // num of threads in AMD wavefront

namespace Utils {

    // NOTE: navigate out of build folder to find csv files
    // performance files
    const std::string ftcs_perf_file = "./data/ftcs_perf.csv";
    const std::string cn_perf_file = "./data/cn_perf.csv";
    const std::string diag_perf_file = "./data/diag_perf.csv";
    // data files
    const std::string diag_data_file = "./data/diag_data.csv";
    const std::string heat_data_file = "./data/heat_data.dat";

    void write_file(const std::string& filename, 
                    const std::string& line, 
                    std::ios_base::openmode mode);

    void write_head(bool diag_log, Method method);

    // stats funcs
    void print_stats(const std::string& file, const Stats& stats, 
                     double t, double bytes, double flops);

    void solve_stats(const Stats& stats, Method method);

    void diag_stats(const Stats& stats);

    inline void write_stats(const Stats& stats, Method method) {
        solve_stats(stats, method);
        if (stats.diag_log) diag_stats(stats);
    }

    void write_diag(const Stats& stats);

    inline void write_heat(const std::string& line, 
                            std::ios_base::openmode mode)
    { write_file(heat_data_file, line, mode); }

}
