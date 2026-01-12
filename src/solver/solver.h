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
    Solver(int dim, Method method, bool diag_log, bool perf_log);

    void solve(int nsteps, bool state_log);
    
private:
    int N;
    Method type;
    Grid grid;
    Stats stats;
    std::function<void(Grid&, Stats&)> step;
};
