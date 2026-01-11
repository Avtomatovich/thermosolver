/**
 * Thermosolver
 * 
 * Name: grid.h
 * 
 * Author: Samson Tsegai
 */

#pragma once

#include <vector>
#include <algorithm>
#include <cmath>
#include <hip/hip_runtime.h>
#include "utils/stats.h"

class Grid
{
public:
    Grid(int N);

    ~Grid();

    void ftcs(double& t);
    void cn(double& t);

    Diag diagnostics(double& t);

    inline void swap() { std::swap(prev, curr); }

    // debug
    void debug();

private:
    // vars
    int Ns, Nr, Nc;
    double dx, dx_2, recip_dx_2, dt; // step sizes
    double r, r_half; // Courant number
    double ftcs_coeff, cn_coeff; // prev update coefficients
    double recip_denom; // Crank-Nicolson update denominator reciprocal
    std::vector<double> prev, curr; // iteration states

    double *prev_d, *curr_d, *res_d, *min_d, *max_d, *total_d; // GPU vars

    // GPU dim vars
    dim3 grid_dim, block_dim;
    int shared_bytes;
    
    static constexpr double ALPHA = 2.3e-5; // thermal diffusivity of iron
    static constexpr double TOL = 1e-5; // convergence tolerance factor

    // funcs
    void init();
    void init_d();

    void pprint(const std::vector<double>& m);

    inline int idx(int i, int j, int k) {
        // i = slice, j = row, k = col
        // slow -> fast, i -> j -> k
        return i * Nr * Nc + j * Nc + k;
    }
};
