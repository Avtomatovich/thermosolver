/**
 * Thermosolver
 * 
 * Name: solver.h
 * 
 * Author: Samson Tsegai
 */

#pragma once

#include "utils/stats.h"
#include "grid.h"

class Solver
{
public:
    Solver(int dim, bool perf_log, bool diag_log);

    void solve();
    
private:
    int N;
    Grid grid;
    Stats stats;

    static constexpr double TOL = 1e-6;
};
