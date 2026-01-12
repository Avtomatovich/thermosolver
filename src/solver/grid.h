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
#include <string>
#include "utils/stats.h"

class Grid
{
public:
    Grid(int N);

    void ftcs(Stats& stats);
    void cn(Stats& stats);

    Diag diagnostics(double& t);

    inline void swap() { std::swap(prev, curr); }

    std::string pprint();

private:
    // vars
    int Ns, Nr, Nc;
    double dx, dx_2, recip_dx_2, dt; // step sizes
    double r, r_half; // Courant number
    double ftcs_coeff, cn_coeff; // prev update coefficients
    double recip_denom; // Crank-Nicolson update denominator reciprocal
    std::vector<double> prev, curr; // iteration states

    static constexpr double ALPHA = 2.3e-5; // thermal diffusivity of iron
    static constexpr double TOL = 1e-5; // convergence tolerance factor

    // funcs
    void init();

    inline int idx(int i, int j, int k) {
        // i = slice, j = row, k = col
        // slow -> fast, i -> j -> k
        return i * Nr * Nc + j * Nc + k;
    }

    inline double neighbors(const std::vector<double>& m, int i, int j, int k) {
        return  m[idx(i - 1, j, k)] + m[idx(i + 1, j, k)] + 
                m[idx(i, j - 1, k)] + m[idx(i, j + 1, k)] + 
                m[idx(i, j, k - 1)] + m[idx(i, j, k + 1)];
    }

    inline double cn_update(int at, int i, int j, int k) {
        return (cn_coeff * prev[at] + 
                r_half * ( neighbors(prev, i, j, k) + neighbors(curr, i, j, k) )) * 
                recip_denom;
    }
};
