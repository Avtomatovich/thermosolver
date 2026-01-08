/**
 * Thermosolver
 * 
 * Name: grid.cpp
 * 
 * Author: Samson Tsegai
 */

#include <omp.h>
#include <cmath>
#include <random>
#include "grid.h"
#include "utils/timer.h"

Grid::Grid(int N) :
    Ns(N), Nr(N), Nc(N), size(Ns * Nr * Nc)
{
    // split [0, 1] domain into N parts, dx = dy = dz
    dx = 1.0 / (N - 1);
    dx_2 = dx * dx;
    recip_dx_2 = 1.0 / dx_2;

    // compute time step based on FTCS stability condition
    dt = 0.1 * dx_2 / (6.0 * alpha);

    prev.assign(size, 0.0);
    curr.assign(size, 0.0);

    initialize(); // random initial state
}

void Grid::initialize() {
    #pragma omp parallel
    {
        std::default_random_engine gen(100 + omp_get_thread_num());
        std::uniform_real_distribution<double> distrib(0.0, 1.0);

        // inner grid only
        #pragma omp for
        for (int i = 1; i <= Ns - 2; i++) {
            for (int j = 1; j <= Nr - 2; j++) {
                for (int k = 1; k <= Nc - 2; k++) {
                    int at = idx(i, j, k);
                    prev[at] = curr[at] = distrib(gen);
                }            
            }
        }
    }
}

void Grid::step(double& t) {
    Timer timer;
    timer.start();
    
    // inner grid in [1, N - 2], not [0, N - 1]
    #pragma omp parallel for
    for (int i = 1; i <= Ns - 2; i++) {
        for (int j = 1; j <= Nr - 2; j++) {
            for (int k = 1; k <= Nc - 2; k++) {
                int at = idx(i, j, k);
                curr[at] = prev[at] + dt * alpha * laplacian(i, j, k);
            }
        }
    }

    timer.end(t);
}

double Grid::residual(double& t) {
    Timer timer;
    timer.start();
    
    double res = 0.0;

    // outer grid
    #pragma omp parallel for reduction(max:res)
    for (int i = 0; i <= Ns - 1; i++) {
        for (int j = 0; j <= Nr - 1; j++) {
            for (int k = 0; k <= Nc - 1; k++) {
                // store max (residual = curr - prev)
                int at = idx(i, j, k);
                res = fmax(res, fabs(curr[at] - prev[at]));
            }
        }
    }

    timer.end(t);

    return res;
}
