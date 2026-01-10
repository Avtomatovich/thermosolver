/**
 * Thermosolver
 * 
 * Name: solver.h
 * 
 * Author: Samson Tsegai
 */

#pragma once

#include <functional>
#include "utils/method.h"
#include "utils/stats.h"
#include "grid.h"

class Solver
{
public:
    Solver(int dim, Method method, bool perf_log, bool diag_log);

    void solve();
    
private:
    int N;
    Method type;
    Grid grid;
    Stats stats;
    std::function<void(Grid&, double&)> step;
};
