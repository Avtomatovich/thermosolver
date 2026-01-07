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

Solver::Solver(int dim, Method method) :
    N(dim), stats(dim), grid(dim), type(method)
{
    switch (type) {
        case Method::JACOBI:
            step = &Grid::jacobi;
            break;
        case Method::RBGS:
            step = &Grid::rbgs;
            grid.clear_prev();
            break;
        case Method::SOR:
            step = &Grid::sor;
            grid.clear_prev();
            break;
    }
}

bool Solver::measure(bool log) {
    double res = grid.residual(stats.res_t);

    if (log) {
        double mae = grid.mae(stats.mae_t);
        double rmse = grid.rmse(stats.rmse_t);
        
        stats.conv_data.push_back({res, mae, rmse});
    }

    return res < TOL;
}

void Solver::solve(bool log) {
    Timer timer;
    timer.start(); /* TIMER START */

    measure(log); // initial metrics

    for (int i = 1; i <= N * N * N; i++) {
        stats.steps = i;

        step(grid, stats.solve_t);
        
        if (measure(log)) break;

        if (type == Method::JACOBI) grid.swap();
    }

    double mae = grid.mae(stats.mae_t);
    double rmse = grid.rmse(stats.rmse_t);

    timer.end(stats.total_t); /* TIMER END */

    printf("||==============================================================||\n");
    if (type == Method::JACOBI) {
        printf("||======================= Jacobi Solver ========================||\n");
    } else if (type == Method::RBGS) {
        printf("||=============== Red-Black Gauss-Seidel Solver ================||\n");
    } else {
        printf("||============= Successive Over-Relaxation Solver ==============||\n");
    }
    printf("* No of iterations: %d, Size: %d x %d x %d\n", stats.steps, N, N, N);
    printf("* Error\n\t* MAE: %g, RMSE: %g\n", mae, rmse);
    printf("* Duration: %g sec\n", stats.total_t);
    fflush(stdout);

    // print results
    Utils::write_stats(stats);
    if (log) Utils::write_conv(stats);
}
