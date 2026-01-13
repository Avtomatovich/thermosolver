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
    double min, max, total;
};

struct Stats {
    Stats(int dim, bool dlog, bool plog) : 
        N(dim), in_size((dim-2) * (dim-2) * (dim-2)), out_size(dim * dim * dim), 
        diag_log(dlog), perf_log(plog)
    {}

    // grid dim
    int N;

    // inner and outer grid size
    int in_size, out_size;

    // time in seconds
    double total_t = 0, solve_t = 0, diag_t = 0;

    // logging toggles
    bool diag_log, perf_log;

    // no of temporal iterations
    int steps = 0;

    // no of spatial RBGS iterations in Crank-Nicolson
    int cn_steps = 0;

    // diagnostic data
    std::vector<Diag> diag_data;
};
