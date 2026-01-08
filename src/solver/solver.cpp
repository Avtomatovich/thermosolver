/**
 * Thermosolver
 * 
 * Name: solver.cpp
 * 
 * Author: Samson Tsegai
 */

#include "solver.h"
#include "utils/timer.h"
#include "utils/utils.h"
#include <cmath>
#include <stdio.h>

Solver::Solver(int dim) :
    N(dim), stats(dim), grid(dim)
{}

bool Solver::measure(bool log) {
    double res = grid.residual(stats.res_t);

    if (log) stats.conv_data.push_back(res);

    return res < TOL;
}

void Solver::solve(bool log) {
    Timer timer;
    timer.start(); /* TIMER START */

    measure(log); // initial metrics

    for (int i = 1; i <= 3 * N; i++) {
        stats.steps = i;

        grid.step(stats.solve_t);
        
        if (measure(log)) break;

        grid.swap();
    }

    timer.end(stats.total_t); /* TIMER END */

    printf("||==============================================================||\n");
    printf("||=================== Heat Equation Solver =====================||\n");
    printf("* No of time steps: %d, Size: %d x %d x %d\n", stats.steps, N, N, N);
    printf("* Duration: %g sec\n", stats.total_t);
    fflush(stdout);

    // print results
    Utils::write_stats(stats);
    if (log) Utils::write_conv(stats);
}
