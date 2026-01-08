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

#define MAX_ITER 10000

Solver::Solver(int dim, bool perf_log, bool conv_log) :
    N(dim), stats(dim, perf_log, conv_log), grid(dim)
{}

void Solver::solve() {
    Timer timer;
    timer.start(); /* TIMER START */

    for (int i = 1; i <= MAX_ITER; i++) {
        stats.steps = i;

        grid.step(stats.solve_t);

        double res = grid.residual(stats.res_t);
        
        if (stats.conv_log) stats.conv_data.push_back(res);

        if (res < TOL) break;

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
    if (stats.conv_log) Utils::write_conv(stats);
}
