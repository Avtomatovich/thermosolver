/**
 * Thermosolver
 * 
 * Name: stats.h
 * 
 * Author: Samson Tsegai
 */

#pragma once

#include <vector>
#include <array>

struct Stats {
    Stats(int dim) : 
        N(dim), in_size((N-2)*(N-2)*(N-2)), out_size(N*N*N)
    {}

    // matrix dim
    int N;

    // no of iterations per step
    int in_size, out_size;

    // time in seconds
    double total_t, solve_t, res_t, mae_t, rmse_t;

    // no of iterations until convergence
    int steps;

    // residuals and errors
    std::vector<std::array<double, 3>> conv_data;
};
