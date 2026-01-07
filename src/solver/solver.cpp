/**
 * Thermosolver
 * 
 * Name: solver.cpp
 * 
 * Author: Samson Tsegai
 */

#include <mpi.h>
#include "solver.h"
#include "utils/timer.h"
#include "utils/utils.h"
#include <cmath>
#include <stdio.h>

Solver::Solver(int dim, Method method) :
    N(dim), stats(dim), grid(dim), type(method)
{
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &proc);

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
    double res;
    double res_local = grid.residual(stats.res_t);
    MPI_Allreduce(&res_local, &res, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    
    if (log) {
        double mae, rmse;

        double mae_local = grid.mae(stats.mae_t);
        double rmse_local = grid.rmse(stats.rmse_t);
        
        MPI_Allreduce(&mae_local, &mae, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&rmse_local, &rmse, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        if (proc == 0) stats.conv_data.push_back({res, 
                                                  mae / stats.out_size, 
                                                  sqrt(rmse / stats.out_size)});
    }

    return res < TOL;
}

void Solver::debug(int i) {
    for (int p = 0; p < nproc; p++) {
        if (p == proc) {
            std::cout << "Step: " << i << std::endl;
            grid.debug();
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
}

void Solver::solve(bool log) {
    Timer timer;
    timer.start(); /* TIMER START */

    measure(log); // initial metrics

    // debug(0);

    for (int i = 1; i <= N * N * N; i++) {
        stats.steps = i;

        step(grid, stats.solve_t);
        
        // debug(i);

        if (measure(log)) break;

        if (type == Method::JACOBI) grid.swap();
    }

    double mae, rmse;
    double mae_global = grid.mae(stats.mae_t);
    double rmse_global = grid.rmse(stats.rmse_t);

    MPI_Allreduce(&mae_global, &mae, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&rmse_global, &rmse, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    timer.end(stats.total_t); /* TIMER END */

    if (proc == 0) {
        printf("||==============================================================||\n");
        if (type == Method::JACOBI) {
            printf("||======================= Jacobi Solver ========================||\n");
        } else if (type == Method::RBGS) {
            printf("||=============== Red-Black Gauss-Seidel Solver ================||\n");
        } else {
            printf("||============= Successive Over-Relaxation Solver ==============||\n");
        }
        printf("* No of iterations: %d, Size: %d x %d x %d\n", stats.steps, N, N, N);
        printf("* Error\n\t* MAE: %g, RMSE: %g\n", mae / stats.out_size, sqrt(rmse / stats.out_size));
        printf("* Duration: %g sec\n", stats.total_t);
        fflush(stdout);

        // print results
        Utils::write_stats(stats);
        if (log) Utils::write_conv(stats);
    }
}
