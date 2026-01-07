/**
 * Thermosolver
 * 
 * Name: solver.h
 * 
 * Author: Samson Tsegai
 */

#pragma once

#include <functional>
#include "utils/stats.h"
#include "grid.h"

enum class Method {
    JACOBI,
    RBGS,
    SOR
};

class Solver
{
public:
    Solver(int dim, Method method);

    void solve(bool log);
    
private:
    int N;
    Method type;
    Grid grid;
    std::function<void(Grid&, double&)> step;
    Stats stats;

    bool measure(bool log);

    static constexpr double TOL = 1e-6;
};
