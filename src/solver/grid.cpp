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
#include <sstream>
#include "grid.h"
#include "utils/utils.h"
#include "utils/timer.h"
#include "gpu/gpu_func.h"

#define MAX_ITER 100

using namespace GPUFunc;

Grid::Grid(int N, Stats& stats) :
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
    r_half = 0.5 * r;

    // TODO: test different dt values
    // r_half *= 2.0;

    // miscellaneous constants
    ftcs_coeff = 1.0 - 6.0 * r;
    cn_coeff = 1.0 - 6.0 * r_half;
    recip_denom = 1.0 / (1.0 + 6.0 * r_half);
    
    int size = Ns * Nr * Nc;
    prev.assign(size, 0.0);
    curr.assign(size, 0.0);
    
    init(); // random initial state

    // init execution config
    block_dim = dim3{8, 8, 8}; // threads per block
    grid_dim = dim3{ // blocks per grid
        (Nc + block_dim.x - 1) / block_dim.x, // fastest
        (Nr + block_dim.y - 1) / block_dim.y,
        (Ns + block_dim.z - 1) / block_dim.z  // slowest
    };
    // shared memory size in bytes to store warp reductions (no of warps = no of threads per block / warp size)
    shared_bytes = ((block_dim.x * block_dim.y * block_dim.z + WARP_SIZE - 1) / WARP_SIZE) * sizeof(double);

    // update stats with num of threads per block and num of blocks per grid
    stats.nthreads = block_dim.x * block_dim.y * block_dim.z;
    stats.nblocks = grid_dim.x * grid_dim.y * grid_dim.z;
    
    // allocate device memory
    // NOTE: size in bytes
    hipMalloc(reinterpret_cast<void**>(&prev_d), size * sizeof(double));
    hipMalloc(reinterpret_cast<void**>(&curr_d), size * sizeof(double));
    hipMalloc(reinterpret_cast<void**>(&res_d), sizeof(double));
    hipMalloc(reinterpret_cast<void**>(&min_d), sizeof(double));
    hipMalloc(reinterpret_cast<void**>(&max_d), sizeof(double));
    hipMalloc(reinterpret_cast<void**>(&total_d), sizeof(double));

    to_device(prev, prev_d); // copy prev to device
    to_device(curr, curr_d); // copy curr to device
}

Grid::~Grid() {
    hipFree(prev_d);
    hipFree(curr_d);
    hipFree(res_d);
    hipFree(min_d);
    hipFree(max_d);
    hipFree(total_d);
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

void Grid::ftcs(Stats& stats) {
    Timer timer;
    timer.start();

    ftcs_kernel<<<grid_dim, block_dim>>>(curr_d, prev_d, Ns, Nr, Nc, ftcs_coeff, r);
    hipDeviceSynchronize();

    timer.end(stats.solve_t);
}

void Grid::cn(Stats& stats) {
    Timer timer;
    timer.start();

    for (int s = 1; s <= MAX_ITER; s++) {
        stats.cn_steps++;

        double res_h = 0.0;

        // copy initial val to device
        to_device(&res_h, res_d);

        // RBGS
        cn_kernel<<<grid_dim, block_dim, shared_bytes>>>(curr_d, prev_d, res_d, Ns, Nr, Nc, 
                                                         cn_coeff, r_half, recip_denom, false);
        hipDeviceSynchronize();
        
        cn_kernel<<<grid_dim, block_dim, shared_bytes>>>(curr_d, prev_d, res_d, Ns, Nr, Nc, 
                                                         cn_coeff, r_half, recip_denom, true);
        hipDeviceSynchronize();

        // copy result to host
        to_host(res_d, &res_h);

        if (res_h < TOL) break;

        if (s == MAX_ITER) std::cerr << "CN did not converge at residual: " << res_h << std::endl;
    }

    timer.end(stats.solve_t);
}

Diag Grid::diagnostics(double& t) {
    Timer timer;
    timer.start();

    double min_h = 1.0, max_h = 0.0, total_h = 0.0;

    // copy initial vals to device
    to_device(&min_h, min_d);
    to_device(&max_h, max_d);
    to_device(&total_h, total_d);

    // NOTE: tripling shared memory array size for 3 reductions
    diag_kernel<<<grid_dim, block_dim, shared_bytes * 3>>>(curr_d, min_d, max_d, total_d,
                                                            Ns, Nr, Nc);
    hipDeviceSynchronize();
    
    // copy results to host
    to_host(min_d, &min_h);
    to_host(max_d, &max_h);
    to_host(total_d, &total_h);

    timer.end(t);

    return Diag{min_h, max_h, total_h};
}

std::string Grid::pprint() {
    // copy device data to host for output
    to_host(curr_d, curr);
    
    std::stringstream buf;
    for (int r = 0; r < Nr; r++) {
        for (int c = 0; c < Nc; c++) {
            // print out first non-boundary 2D slice
            buf << curr[idx(1, r, c)];
            if (c != Nc - 1) buf << ", ";            
        }
        buf << std::endl;
    }
    
    return buf.str();
}
