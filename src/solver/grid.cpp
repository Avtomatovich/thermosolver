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
#include "gpu/gpu_func.h"

#define MAX_ITER 100
#define WARP_SIZE 64 // num of threads in AMD wavefront

using namespace GPUFunc;

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

    init_d(); // allocate device memory
    to_device(prev, prev_d); // copy prev to device
    to_device(curr, curr_d); // copy curr to device
}

Grid::~Grid() {
    hipFree(prev_d);
    hipFree(curr_d);
    hipFree(res_d);
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

void Grid::init_d() {
    block_dim = dim3{8, 8, 8}; // init execution config
    grid_dim = dim3{
        (Nc + block_dim.x - 1) / block_dim.x, // fastest
        (Nr + block_dim.y - 1) / block_dim.y,
        (Ns + block_dim.z - 1) / block_dim.z  // slowest
    };
    // array size of warp reduction partials (no of warps = no of threads per block / warp size)
    shared_bytes = ((block_dim.x * block_dim.y * block_dim.z + WARP_SIZE - 1) / WARP_SIZE) * sizeof(double);

    // NOTE: size in bytes
    hipMalloc(reinterpret_cast<void**>(&prev_d), Ns * Nr * Nc * sizeof(double));
    hipMalloc(reinterpret_cast<void**>(&curr_d), Ns * Nr * Nc * sizeof(double));
    hipMalloc(reinterpret_cast<void**>(&res_d), sizeof(double));
}

void Grid::ftcs(double& t) {
    Timer timer;
    timer.start();

    ftcs_kernel<<<grid_dim, block_dim>>>(curr_d, prev_d, Ns, Nr, Nc, ftcs_coeff, r);
    hipDeviceSynchronize();

    timer.end(t);
}

void Grid::cn(double& t) {
    Timer timer;
    timer.start();

    for (int s = 1; s <= MAX_ITER; s++) {
        double res = 0.0;

        // RBGS
        cn_kernel<<<grid_dim, block_dim, shared_bytes>>>(curr_d, prev_d, res_d, 
                                                         Ns, Nr, Nc, 
                                                         cn_coeff, r_half, recip_denom, false);
        hipDeviceSynchronize();
        
        cn_kernel<<<grid_dim, block_dim, shared_bytes>>>(curr_d, prev_d, res_d, 
                                                         Ns, Nr, Nc, 
                                                         cn_coeff, r_half, recip_denom, true);
        hipDeviceSynchronize();

        to_host(res_d, &res);

        if (res < TOL) break;

        if (s == MAX_ITER) std::cerr << "CN did not converge at residual: " << res << std::endl;
    }

    timer.end(t);
}

void Grid::debug() {
    std::cout << "===Prev===" << std::endl;
    pprint(prev);
    std::cout << "===Curr===" << std::endl;
    pprint(curr);
}

void Grid::pprint(const std::vector<double>& m) {
    std::cout << "* Dims: " << Ns << " x " << Nr << " x " << Nc << std::endl << std::endl;

    for (int i = 0; i < Ns; i++) {
        std::cout << "Slice: " << i << std::endl;
        for (int j = 0; j < Nr; j++) {
            for (int k = 0; k < Nc; k++) {
                std::cout << m[idx(i, j, k)] << ", ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
    std::cout << std::endl << std::endl;
    fflush(stdout);
}
