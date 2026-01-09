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

Solver::Solver(int dim, bool perf_log, bool diag_log) :
    N(dim), stats(dim, perf_log, diag_log), grid(dim)
{}

void Solver::solve() {
    Timer timer;
    timer.start(); /* TIMER START */

    for (int i = 1; i <= MAX_ITER; i++) {
        stats.steps = i;

        grid.step(stats.solve_t);

        if (stats.diag_log) {
            stats.diag_data.push_back(grid.diagnostics(stats.diag_t));
        }

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
    if (stats.diag_log) Utils::write_diag(stats);
}
