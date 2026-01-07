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

class Grid
{
public:
    Grid(int N);

    void jacobi(double& t);
    void rbgs(double& t);
    void sor(double& t);

    double residual(double& t);
    double mae(double& t);
    double rmse(double& t);

    inline void swap() { std::swap(prev, curr); }
    inline void clear_prev() { prev.clear(); }

private:
    // vars
    int Ns, Nr, Nc, size;
    double dx, dx_2, recip_dx_2;
    std::vector<double> prev; // previous iteration to approx soln
    std::vector<double> curr; // current iteration to approx soln
    std::vector<double> rhs; // laplacian of soln to check against
    std::vector<double> soln; // exact soln to solve for

    // relaxation factor for SOR method
    static constexpr double omega = 1.9;

    static constexpr double RECIP_6 = 1.0 / 6.0;

    // funcs
    void discretize();
    void dirichlet();

    inline int idx(int i, int j, int k) {
        // i = slice, j = row, k = col
        // slow -> fast, i -> j -> k
        return i * Nr * Nc + j * Nc + k;
    }

    inline double stencil(const std::vector<double>& m, int i, int j, int k) {
        return (m[idx(i - 1, j, k)] + m[idx(i + 1, j, k)] +
                m[idx(i, j - 1, k)] + m[idx(i, j + 1, k)] +
                m[idx(i, j, k - 1)] + m[idx(i, j, k + 1)] -
                dx_2 * rhs[idx(i, j, k)]) * RECIP_6;
    }

    inline double laplacian(int i, int j, int k) {
        return (curr[idx(i - 1, j, k)] + curr[idx(i + 1, j, k)] + 
                curr[idx(i, j - 1, k)] + curr[idx(i, j + 1, k)] + 
                curr[idx(i, j, k - 1)] + curr[idx(i, j, k + 1)] -
                6.0 * curr[idx(i, j, k)]) * recip_dx_2;
    }
};
