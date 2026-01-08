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

    void step(double& t);

    double residual(double& t);

    inline void swap() { std::swap(prev, curr); }

private:
    // vars
    int Ns, Nr, Nc, size;
    double dx, dx_2, recip_dx_2, dt;
    std::vector<double> prev, curr; // iteration states

    static constexpr double ALPHA = 2.3e-5; // thermal diffusivity of iron

    // funcs
    void initialize();

    inline int idx(int i, int j, int k) {
        // i = slice, j = row, k = col
        // slow -> fast, i -> j -> k
        return i * Nr * Nc + j * Nc + k;
    }

    inline double laplacian(int i, int j, int k) {
        return (prev[idx(i - 1, j, k)] + prev[idx(i + 1, j, k)] + 
                prev[idx(i, j - 1, k)] + prev[idx(i, j + 1, k)] + 
                prev[idx(i, j, k - 1)] + prev[idx(i, j, k + 1)] -
                6.0 * prev[idx(i, j, k)]) * recip_dx_2;
    }
};
