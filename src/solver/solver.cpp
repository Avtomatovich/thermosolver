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
#include <string>
#include <stdio.h>

Solver::Solver(int dim, Method method, bool diag_log, bool perf_log) :
    N(dim), type(method), grid(dim), stats(dim, diag_log, perf_log),
    step(type == Method::FTCS ? &Grid::ftcs : &Grid::cn)
{}

void Solver::solve(int nsteps, bool heat_log) {
    Timer timer;
    timer.start(); /* TIMER START */

    stats.steps = nsteps;
    // NOTE: use ceiling division to keep nframes capped at 100
    int skip = (nsteps + 100 - 1) / 100;
    int nframes = (nsteps + skip - 1) / skip + 1; // increment to include initial frame

    // initial metrics
    if (stats.diag_log) stats.diag_data.push_back(grid.diagnostics(stats.diag_t));
    if (heat_log) {
        Utils::write_heat(std::to_string(nframes), std::ios::out);
        Utils::write_heat(grid.pprint(), std::ios::app);
    }

    for (int i = 1; i <= nsteps; i++) {
        step(grid, stats);

        if (stats.diag_log) stats.diag_data.push_back(grid.diagnostics(stats.diag_t));

        // print state every few time steps
        if (heat_log && i % skip == 0) Utils::write_heat(grid.pprint(), std::ios::app);

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
    Utils::write_stats(stats, type);
    if (stats.diag_log) Utils::write_diag(stats);
}
