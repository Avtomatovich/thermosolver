/**
 * Thermosolver
 * 
 * Name: stats.h
 * 
 * Author: Samson Tsegai
 */

#pragma once

#include <vector>

struct Stats {
    Stats(int dim) : 
        N(dim), in_size((N-2)*(N-2)*(N-2)), out_size(N*N*N)
    {}

    // matrix dim
    int N;

    // inner and outer grid size
    int in_size, out_size;

    // time in seconds
    double total_t, solve_t, res_t;

    // no of iterations until end of simulation
    int steps;

    // residuals and errors
    std::vector<double> conv_data;
};
