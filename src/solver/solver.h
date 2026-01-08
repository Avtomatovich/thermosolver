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
    Solver(int dim);

    void solve(bool log);
    
private:
    int N;
    Grid grid;
    Stats stats;

    bool measure(bool log);

    static constexpr double TOL = 1e-6;
};
