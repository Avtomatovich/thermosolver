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

#define MAX_STEPS 10000

Solver::Solver(int dim, Method method, bool diag_log, bool perf_log) :
    N(dim), type(method), grid(dim), stats(dim, diag_log, perf_log),
    step(type == Method::FTCS ? &Grid::ftcs : &Grid::cn)
{}

void Solver::solve() {
    Timer timer;
    timer.start(); /* TIMER START */

    if (stats.diag_log) { // initial metrics
        stats.diag_data.push_back(grid.diagnostics(stats.diag_t));
    }

    for (int i = 1; i <= MAX_STEPS; i++) {
        stats.steps = i;

        step(grid, stats.solve_t);

        if (stats.diag_log) {
            stats.diag_data.push_back(grid.diagnostics(stats.diag_t));
        }

        grid.swap();
    }

    timer.end(stats.total_t); /* TIMER END */

    printf("||================================================================||\n");
    if (type == Method::FTCS) {
        printf("||============ Forward Time-Centered Space Heat Solver ===========||\n");
    } else {
        printf("||================= Crank-Nicolson Heat Solver ===================||\n");
    }
    printf("* No of time steps: %d, Size: %d x %d x %d\n", stats.steps, N, N, N);
    printf("* Duration: %g sec\n", stats.total_t);
    fflush(stdout);

    // print results
    if (stats.diag_log) Utils::write_diag(stats);
    Utils::write_stats(stats, type);
}
