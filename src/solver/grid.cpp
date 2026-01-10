/**
 * Thermosolver
 * 
 * Name: grid.cpp
 * 
 * Author: Samson Tsegai
 */

#include <omp.h>
#include <random>
#include <iostream>
#include "grid.h"
#include "utils/timer.h"

#define MAX_ITER 100

Grid::Grid(int N) :
    Ns(N), Nr(N), Nc(N)
{
    // split [0, 1] domain into N parts, dx = dy = dz
    dx = 1.0 / (N - 1);
    dx_2 = dx * dx;
    recip_dx_2 = 1.0 / dx_2;

    // time step bound by CFL condition
    dt = 0.8 * dx_2 / (6.0 * ALPHA);

    // Courant number
    r = ALPHA * dt * recip_dx_2;
    r_half = 0.5 * ALPHA * 2.0 * dt * recip_dx_2;

    // miscellaneous constants
    ftcs_coeff = 1.0 - 6.0 * r;
    cn_coeff = 1.0 - 6.0 * r_half;
    recip_denom = 1.0 / (1.0 + 6.0 * r_half);

    int size = Ns * Nr * Nc;
    prev.assign(size, 0.0);
    curr.assign(size, 0.0);

    init(); // random initial state
}

void Grid::init() {
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

void Grid::ftcs(double& t) {
    Timer timer;
    timer.start();

    // inner grid in [1, N - 2], not [0, N - 1]
    #pragma omp parallel for
    for (int i = 1; i <= Ns - 2; i++) {
        for (int j = 1; j <= Nr - 2; j++) {
            for (int k = 1; k <= Nc - 2; k++) {
                int at = idx(i, j, k);
                curr[at] = ftcs_coeff * prev[at] + r * neighbors(prev, i, j, k);
            }
        }
    }

    timer.end(t);
}

void Grid::cn(double& t) {
    Timer timer;
    timer.start();

    for (int s = 1; s <= MAX_ITER; s++) {
        double res = 0.0;

        // RBGS
        // inner grid only
        #pragma omp parallel for reduction(max:res)
        for (int i = 1; i <= Ns - 2; i++) {
            for (int j = 1; j <= Nr - 2; j++) {
                for (int k = 1; k <= Nc - 2; k++) {
                    // red (even)
                    if (!(i + j + k & 1)) {
                        int at = idx(i, j, k);
                        double c = curr[at];
                        double u = cn_update(at, i, j, k);
                        curr[at] = u;
                        res = fmax(res, fabs(u - c));
                    }
                }
            }
        }

        #pragma omp parallel for reduction(max:res)
        for (int i = 1; i <= Ns - 2; i++) {
            for (int j = 1; j <= Nr - 2; j++) {
                for (int k = 1; k <= Nc - 2; k++) {
                    // black (odd)
                    if (i + j + k & 1) {
                        int at = idx(i, j, k);
                        double c = curr[at];
                        double u = cn_update(at, i, j, k);
                        curr[at] = u;
                        res = fmax(res, fabs(u - c));
                    }
                }
            }
        }

        if (res < TOL) break;

        if (s == MAX_ITER) std::cerr << "CN did not converge at residual: " << res << std::endl;
    }

    timer.end(t);
}

Diag Grid::diagnostics(double& t) {
    Timer timer;
    timer.start();
    
    double min_v = 1.0, max_v = 0.0, total = 0.0;

    // outer grid
    #pragma omp parallel for reduction(min:min_v) reduction(max:max_v) reduction(+:total)
    for (int i = 0; i <= Ns - 1; i++) {
        for (int j = 0; j <= Nr - 1; j++) {
            for (int k = 0; k <= Nc - 1; k++) {
                double c = curr[idx(i, j, k)];
                min_v = fmin(min_v, c);
                max_v = fmax(max_v, c);
                total += c;
            }
        }
    }

    timer.end(t);

    return Diag{min_v, max_v, total};
}
