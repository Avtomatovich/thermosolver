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
#include <hip/hip_runtime.h>

class Grid
{
public:
    Grid(int N);

    ~Grid();

    void jacobi(double& t);
    void rbgs(double& t);
    void sor(double& t);

    double residual(double& t);
    double mae(double& t);
    double rmse(double& t);

    inline void swap() { std::swap(prev, curr); }
    inline void clear_prev() { prev.clear(); }

    // debug
    void print_rhs();
    void print_soln();
    void debug();

private:
    // vars
    int Ns, Nr, Nc;
    double dx, dx_2, recip_dx_2;
    std::vector<double> prev; // previous iteration to approx soln
    std::vector<double> curr; // current iteration to approx soln
    std::vector<double> rhs; // laplacian of soln to check against
    std::vector<double> soln; // exact soln to solve for

    int nproc, proc, lproc, rproc, offset; // MPI vars

    double *prev_d, *curr_d, *rhs_d, *soln_d, *red_d; // GPU vars

    // GPU dim vars
    dim3 grid_dim, block_dim;
    int shared_bytes;
    
    static constexpr double RECIP_6 = 1.0 / 6.0;

    // funcs
    void discretize();
    void dirichlet();
    
    void halo_swap(std::vector<double>& m);

    void init_d();

    void pprint(const std::vector<double>& m);

    inline bool is_first() { return proc == 0; }
    inline bool is_last() { return proc == nproc - 1; }

    inline int get_start() { return is_first() ? 2 : 1; }
    inline int get_end() { return is_last() ? Ns - 3 : Ns - 2; } 

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
