/**
 * Thermosolver
 * 
 * Name: grid.cpp
 * 
 * Author: Samson Tsegai
 */

#include <mpi.h>
#include <omp.h>
#include <cmath>
#include "grid.h"
#include "equation.h"
#include "utils/timer.h"
#include "gpu/gpu_func.h"

#define TO_LEFT 0
#define TO_RIGHT 1
#define WARP_SIZE 64 // num of threads in AMD wavefront

using namespace Equation;
using namespace GPUFunc;

Grid::Grid(int N) :
    Ns(N), Nr(N), Nc(N)
{
    // split [0, 1] domain into N parts, dx = dy = dz
    dx = 1.0 / (N - 1);
    dx_2 = dx * dx;
    recip_dx_2 = 1.0 / dx_2;
    
    // init MPI vars
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &proc);
    lproc = is_first() ? MPI_PROC_NULL : proc - 1;
    rproc = is_last() ? MPI_PROC_NULL : proc + 1;
    
    int avg = N / nproc, rem = N % nproc;
    Ns = avg + (proc < rem ? 1 : 0) + 2; // add halos
    
    // compute offset
    offset = 0;
    for (int i = 0; i < proc; i++) offset += avg + (i < rem ? 1 : 0);
    
    prev.assign(Ns * Nr * Nc, 0.0);
    curr.assign(Ns * Nr * Nc, 0.0);
    rhs.assign(Ns * Nr * Nc, 0.0);
    soln.assign(Ns * Nr * Nc, 0.0);
    
    discretize(); // continuous -> discrete
    dirichlet(); // set Dirichlet boundaries
    
    init_d(); // allocate device memory
    to_device(curr, curr_d); // copy curr to device
    to_device(rhs, rhs_d); // copy rhs to device
    to_device(soln, soln_d); // copy soln to device
    
    // clear read-only data after copying to device
    rhs.clear();
    soln.clear(); 
}

Grid::~Grid() {
    hipFree(prev_d);
    hipFree(curr_d);
    hipFree(rhs_d);
    hipFree(soln_d);
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
    hipMalloc(reinterpret_cast<void**>(&rhs_d), Ns * Nr * Nc * sizeof(double));
    hipMalloc(reinterpret_cast<void**>(&soln_d), Ns * Nr * Nc * sizeof(double));
    hipMalloc(reinterpret_cast<void**>(&red_d), sizeof(double));
}

void Grid::discretize() {
    #pragma omp parallel for
    for (int i = 1; i <= Ns - 2; i++) { // NOTE: [1, Ns - 2] to avoid halos
        for (int j = 0; j <= Nr - 1; j++) {
            for (int k = 0; k <= Nc - 1; k++) {
                // i_global = offset + i_local - 1
                double x = (offset + i - 1) * dx;
                double y = j * dx;
                double z = k * dx;

                rhs[idx(i, j, k)] = f(x, y, z);
                soln[idx(i, j, k)] = phi(x, y, z);
            }            
        }
    }
}

void Grid::dirichlet() {
    #pragma omp parallel for
    for (int i = 1; i <= Ns - 2; i++) { // NOTE: [1, Ns - 2] to avoid halos
        for (int j = 0; j <= Nr - 1; j++) {
            int bottom = idx(i, 0, j), top = idx(i, Nr - 1, j);
            int front = idx(i, j, 0), back = idx(i, j, Nc - 1);

            prev[top] = curr[top] = soln[top];
            prev[bottom] = curr[bottom] = soln[bottom];
            prev[front] = curr[front] = soln[front];
            prev[back] = curr[back] = soln[back];
        }
    }

    if (is_first()) {
        #pragma omp parallel for
        for (int j = 0; j <= Nr - 1; j++) {
            for (int k = 0; k <= Nc - 1; k++) {
                // avoid left halo at i = 0
                int left = idx(1, j, k);
                prev[left] = curr[left] = soln[left];
            }
        }
    }

    if (is_last()) {
        #pragma omp parallel for
        for (int j = 0; j <= Nr - 1; j++) {
            for (int k = 0; k <= Nc - 1; k++) {
                // avoid right halo at i = Ns - 1
                int right = idx(Ns - 2, j, k);
                prev[right] = curr[right] = soln[right];
            }
        }
    }
}


void Grid::halo_swap(std::vector<double>& m) {
    MPI_Request reqs[4];
    MPI_Status status[4];

    // populate right halo with data from right neighbor (sent to left from right neighbor pov)
    MPI_Irecv(&m[idx(Ns - 1, 0, 0)], Nr * Nc, MPI_DOUBLE, rproc, TO_LEFT, MPI_COMM_WORLD, &reqs[0]);

    // populate left halo with data from left neighbor (sent to right from left neighbor pov)
    MPI_Irecv(&m[idx(0, 0, 0)], Nr * Nc, MPI_DOUBLE, lproc, TO_RIGHT, MPI_COMM_WORLD, &reqs[1]);

    // send right real slice to right neighbor and tag to right
    MPI_Isend(&m[idx(Ns - 2, 0, 0)], Nr * Nc, MPI_DOUBLE, rproc, TO_RIGHT, MPI_COMM_WORLD, &reqs[2]);

    // send left real slice to left neighbor and tag to left
    MPI_Isend(&m[idx(1, 0, 0)], Nr * Nc, MPI_DOUBLE, lproc, TO_LEFT, MPI_COMM_WORLD, &reqs[3]);

    MPI_Waitall(4, reqs, status);
}


void Grid::jacobi(double& t) {
    Timer timer;
    timer.start();

    // NOTE: inner grid boundary handling
    int start = get_start(), end = get_end();
    
    halo_swap(prev);

    to_device(prev, prev_d);
    to_device(curr, curr_d);
    
    jacobi_kernel<<<grid_dim, block_dim>>>(prev_d, curr_d, rhs_d, Nr, Nc, dx_2, start, end);
    hipDeviceSynchronize();

    to_host(prev_d, prev);
    to_host(curr_d, curr);

    timer.end(t);
}

void Grid::rbgs(double& t) {
    Timer timer;
    timer.start();

    // NOTE: inner grid boundary handling
    int start = get_start(), end = get_end();

    rbgs_kernel<<<grid_dim, block_dim>>>(curr_d, rhs_d, Nr, Nc, dx_2, start, end, offset, false);
    hipDeviceSynchronize();
    
    to_host(curr_d, curr);
    halo_swap(curr);
    to_device(curr, curr_d);

    rbgs_kernel<<<grid_dim, block_dim>>>(curr_d, rhs_d, Nr, Nc, dx_2, start, end, offset, true);
    hipDeviceSynchronize();

    timer.end(t);
}

void Grid::sor(double& t) {
    Timer timer;
    timer.start();

    // NOTE: inner grid boundary handling
    int start = get_start(), end = get_end();

    sor_kernel<<<grid_dim, block_dim>>>(curr_d, rhs_d, Nr, Nc, dx_2, start, end, offset, false);
    hipDeviceSynchronize();
    
    to_host(curr_d, curr);
    halo_swap(curr);
    to_device(curr, curr_d);

    sor_kernel<<<grid_dim, block_dim>>>(curr_d, rhs_d, Nr, Nc, dx_2, start, end, offset, true);
    hipDeviceSynchronize();

    timer.end(t);
}

double Grid::residual(double& t) {
    Timer timer;
    timer.start();
    
    double res;

    set_val_device(red_d, 0.0);

    // NOTE: inner grid boundary handling
    int start = get_start(), end = get_end();

    res_kernel<<<grid_dim, block_dim, shared_bytes>>>(curr_d, rhs_d, Nr, Nc, recip_dx_2, start, end, red_d);
    hipDeviceSynchronize();

    to_host(red_d, &res);

    timer.end(t);

    return res;
}

double Grid::mae(double& t) {
    Timer timer;
    timer.start();
    
    double mae;
    
    set_val_device(red_d, 0.0);
    
    err_kernel<<<grid_dim, block_dim, shared_bytes>>>(curr_d, soln_d, Ns, Nr, Nc, false, red_d);
    hipDeviceSynchronize();
    
    to_host(red_d, &mae);

    timer.end(t);

    return mae;
}

double Grid::rmse(double& t) {
    Timer timer;
    timer.start();
    
    double rmse;

    set_val_device(red_d, 0.0);
    
    err_kernel<<<grid_dim, block_dim, shared_bytes>>>(curr_d, soln_d, Ns, Nr, Nc, true, red_d);
    hipDeviceSynchronize();

    to_host(red_d, &rmse);

    timer.end(t);

    return rmse;
}

void Grid::print_rhs() {
    std::cout << "===RHS (f)===" << std::endl;
    pprint(rhs);
}

void Grid::print_soln() {
    std::cout << "===Solution (phi)===" << std::endl;
    pprint(soln);
}

void Grid::debug() {
    if (!prev.empty()) {
        std::cout << "===Prev===" << std::endl;
        pprint(prev);
    }
    std::cout << "===Curr===" << std::endl;
    pprint(curr);
}

void Grid::pprint(const std::vector<double>& m) {
    std::cout << "* Proc: " << proc << std::endl;
    std::cout << "* Offset: " << offset << std::endl;
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
