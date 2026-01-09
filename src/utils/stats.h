/**
 * Thermosolver
 * 
 * Name: stats.h
 * 
 * Author: Samson Tsegai
 */

#pragma once

#include <vector>

struct Diag {
    double res, min, max, total;
};

struct Stats {
    Stats(int dim, bool plog) : 
        N(dim), in_size((N-2)*(N-2)*(N-2)), out_size(N*N*N), perf_log(plog)
    {}

    // matrix dim
    int N;

    // inner and outer grid size
    int in_size, out_size;

    // time in seconds
    double total_t, solve_t, diag_t;

    // no of iterations until end of simulation
    int steps;

    // diagnostic data
    std::vector<Diag> diag_data;

    // toggle performance logging
    bool perf_log;
};
