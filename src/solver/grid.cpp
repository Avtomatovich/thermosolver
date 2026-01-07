/**
 * Thermosolver
 * 
 * Name: grid.cpp
 * 
 * Author: Samson Tsegai
 */

#include <omp.h>
#include <cmath>
#include "grid.h"
#include "equation.h"
#include "utils/timer.h"

using namespace Equation;

Grid::Grid(int N) :
    Ns(N), Nr(N), Nc(N), size(Ns * Nr * Nc)
{
    // split [0, 1] domain into N parts, dx = dy = dz
    dx = 1.0 / (N - 1);
    dx_2 = dx * dx;
    recip_dx_2 = 1.0 / dx_2;

    prev.assign(size, 0.0);
    curr.assign(size, 0.0);
    rhs.assign(size, 0.0);
    soln.assign(size, 0.0);

    discretize(); // continuous -> discrete
    dirichlet(); // set Dirichlet boundaries
}

void Grid::discretize() {
    #pragma omp parallel for
    for (int i = 0; i <= Ns - 1; i++) {
        for (int j = 0; j <= Nr - 1; j++) {
            for (int k = 0; k <= Nc - 1; k++) {
                double x = i * dx, y = j * dx, z = k * dx;
                rhs[idx(i, j, k)] = f(x, y, z);
                soln[idx(i, j, k)] = phi(x, y, z);
            }            
        }
    }
}

void Grid::dirichlet() {
    #pragma omp parallel for
    for (int i = 0; i <= Ns - 1; i++) {
        for (int j = 0; j <= Nr - 1; j++) {
            int left = idx(0, i, j), right = idx(Ns - 1, i, j);
            int bottom = idx(i, 0, j), top = idx(i, Nr - 1, j);
            int front = idx(i, j, 0), back = idx(i, j, Nc - 1);

            prev[top] = curr[top] = soln[top];
            prev[bottom] = curr[bottom] = soln[bottom];
            prev[left] = curr[left] = soln[left];
            prev[right] = curr[right] = soln[right];
            prev[front] = curr[front] = soln[front];
            prev[back] = curr[back] = soln[back];
        }
    }
}

void Grid::jacobi(double& t) {
    Timer timer;
    timer.start();
    
    // solve for interior grid in [1, N - 2], not [0, N - 1]
    #pragma omp parallel for
    for (int i = 1; i <= Ns - 2; i++) {
        for (int j = 1; j <= Nr - 2; j++) {
            for (int k = 1; k <= Nc - 2; k++) {
                curr[idx(i, j, k)] = stencil(prev, i, j, k);
            }
        }
    }

    timer.end(t);
}

void Grid::rbgs(double& t) {
    Timer timer;
    timer.start();
    
    // solve for interior grid in [1, N - 2], not [0, N - 1]
    #pragma omp parallel for
    for (int i = 1; i <= Ns - 2; i++) {
        for (int j = 1; j <= Nr - 2; j++) {
            for (int k = 1; k <= Nc - 2; k++) {
                // red update
                if ((i + j + k) % 2 == 0) curr[idx(i, j, k)] = stencil(curr, i, j, k);
            }
        }
    }

    #pragma omp parallel for
    for (int i = 1; i <= Ns - 2; i++) {
        for (int j = 1; j <= Nr - 2; j++) {
            for (int k = 1; k <= Nc - 2; k++) {
                // black update
                if ((i + j + k) % 2 == 1) curr[idx(i, j, k)] = stencil(curr, i, j, k);
            }
        }
    }

    timer.end(t);
}

void Grid::sor(double& t) {
    Timer timer;
    timer.start();

    // solve for interior grid in [1, N - 2], not [0, N - 1]
    #pragma omp parallel for
    for (int i = 1; i <= Ns - 2; i++) {
        for (int j = 1; j <= Nr - 2; j++) {
            for (int k = 1; k <= Nc - 2; k++) {
                // red update
                if ((i + j + k) % 2 == 0) {
                    double c = curr[idx(i, j, k)];
                    curr[idx(i, j, k)] = c + omega * (stencil(curr, i, j, k) - c);
                }
            }
        }
    }

    #pragma omp parallel for
    for (int i = 1; i <= Ns - 2; i++) {
        for (int j = 1; j <= Nr - 2; j++) {
            for (int k = 1; k <= Nc - 2; k++) {
                // black update
                if ((i + j + k) % 2 == 1) {
                    double c = curr[idx(i, j, k)];
                    curr[idx(i, j, k)] = c + omega * (stencil(curr, i, j, k) - c);
                }
            }
        }
    }

    timer.end(t);
}

double Grid::residual(double& t) {
    Timer timer;
    timer.start();
    
    double res = 0.0;

    // interior cells only
    #pragma omp parallel for reduction(max:res)
    for (int i = 1; i <= Ns - 2; i++) {
        for (int j = 1; j <= Nr - 2; j++) {
            for (int k = 1; k <= Nc - 2; k++) {
                // store max (residual = actual - iterative)
                res = fmax(res, fabs(rhs[idx(i, j, k)] - laplacian(i, j, k)));
            }
        }
    }

    timer.end(t);

    return res;
}

double Grid::mae(double& t) {
    Timer timer;
    timer.start();
    
    double mae = 0.0;

    #pragma omp parallel for reduction(+:mae)
    for (int i = 0; i <= Ns - 1; i++) {
        for (int j = 0; j <= Nr - 1; j++) {
            for (int k = 0; k <= Nc - 1; k++) {
                int at = idx(i, j, k);
                mae += fabs(curr[at] - soln[at]);
            }
        }
    }

    timer.end(t);

    return mae / size;
}

double Grid::rmse(double& t) {
    Timer timer;
    timer.start();
    
    double rmse = 0.0;

    #pragma omp parallel for reduction(+:rmse)
    for (int i = 0; i <= Ns - 1; i++) {
        for (int j = 0; j <= Nr - 1; j++) {
            for (int k = 0; k <= Nc - 1; k++) {
                int at = idx(i, j, k);
                double err = fabs(curr[at] - soln[at]);
                rmse += err * err;
            }
        }
    }

    timer.end(t);

    return sqrt(rmse / size);
}
