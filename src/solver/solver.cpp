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

Solver::Solver(int dim, Method method, bool diag_log, bool perf_log) :
    N(dim), type(method), grid(dim), stats(dim, diag_log, perf_log),
    step(type == Method::FTCS ? &Grid::ftcs : &Grid::cn)
{}

void Solver::solve(int nsteps, bool state_log) {
    Timer timer;
    timer.start(); /* TIMER START */

    // initial metrics
    if (stats.diag_log) stats.diag_data.push_back(grid.diagnostics(stats.diag_t));
    if (state_log) Utils::write_state(grid.pprint(), std::ios::out);

    stats.steps = nsteps;

    for (int i = 1; i <= nsteps; i++) {
        step(grid, stats);

        if (stats.diag_log) stats.diag_data.push_back(grid.diagnostics(stats.diag_t));

        // print state every 100 time steps
        if (state_log && i % 100 == 0) Utils::write_state(grid.pprint(), std::ios::app);

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
