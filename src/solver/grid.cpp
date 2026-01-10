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
#include "utils/timer.h"
#include "gpu/gpu_func.h"

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
    hipFree(red_d);
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
    hipMalloc(reinterpret_cast<void**>(&red_d), sizeof(double));
}

// void Grid::rbgs(double& t) {
//     Timer timer;
//     timer.start();

//     rbgs_kernel<<<grid_dim, block_dim>>>(curr_d, prev_d, Ns, Nr, Nc, dx_2, false);
//     hipDeviceSynchronize();

//     rbgs_kernel<<<grid_dim, block_dim>>>(curr_d, prev_d, Ns, Nr, Nc, dx_2, true);
//     hipDeviceSynchronize();

//     timer.end(t);
// }

// double Grid::mae(double& t) {
//     Timer timer;
//     timer.start();
    
//     double mae;
    
//     set_val_device(red_d, 0.0);
    
//     err_kernel<<<grid_dim, block_dim, shared_bytes>>>(curr_d, soln_d, Ns, Nr, Nc, false, red_d);
//     hipDeviceSynchronize();
    
//     to_host(red_d, &mae);

//     timer.end(t);

//     return mae;
// }

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
