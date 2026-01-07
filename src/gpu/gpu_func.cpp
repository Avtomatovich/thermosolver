/**
 * Thermosolver
 * 
 * Name: gpu_func.cpp
 * 
 * Author: Samson Tsegai
 */

#include "gpu_func.h"

namespace GPUFunc {

    __global__ void jacobi_kernel(double* prev_d, double* curr_d, double* rhs_d,
                                  int Nr, int Nc, double dx_2, 
                                  int start, int end) 
    {
        int i = blockIdx.z * blockDim.z + threadIdx.z;
        int j = blockIdx.y * blockDim.y + threadIdx.y;
        int k = blockIdx.x * blockDim.x + threadIdx.x;

        // inner grid ops, avoid boundaries
        if (i < start || i > end || j < 1 || j > Nr - 2 || k < 1 || k > Nc - 2) return;

        int at = idx(i, j, k, Nr, Nc);

        curr_d[at] = (prev_d[idx(i - 1, j, k, Nr, Nc)] + prev_d[idx(i + 1, j, k, Nr, Nc)] +
                      prev_d[idx(i, j - 1, k, Nr, Nc)] + prev_d[idx(i, j + 1, k, Nr, Nc)] +
                      prev_d[idx(i, j, k - 1, Nr, Nc)] + prev_d[idx(i, j, k + 1, Nr, Nc)] -
                      dx_2 * rhs_d[at]) * RECIP_6;
    }

    __global__ void rbgs_kernel(double* curr_d, double* rhs_d,
                                int Nr, int Nc, double dx_2, 
                                int start, int end, int offset, bool parity)
    {
        int i = blockIdx.z * blockDim.z + threadIdx.z;
        int j = blockIdx.y * blockDim.y + threadIdx.y;
        int k = blockIdx.x * blockDim.x + threadIdx.x;

        // inner grid ops, avoid boundaries
        if (i < start || i > end || j < 1 || j > Nr - 2 || k < 1 || k > Nc - 2) return;

        // use global index to compute parity
        if (((offset + i - 1) + j + k) % 2 != parity) return;

        int at = idx(i, j, k, Nr, Nc);

        curr_d[at] = (curr_d[idx(i - 1, j, k, Nr, Nc)] + curr_d[idx(i + 1, j, k, Nr, Nc)] +
                      curr_d[idx(i, j - 1, k, Nr, Nc)] + curr_d[idx(i, j + 1, k, Nr, Nc)] +
                      curr_d[idx(i, j, k - 1, Nr, Nc)] + curr_d[idx(i, j, k + 1, Nr, Nc)] -
                      dx_2 * rhs_d[at]) * RECIP_6;

    }

    __global__ void sor_kernel(double* curr_d, double* rhs_d,
                                int Nr, int Nc, double dx_2, 
                                int start, int end, int offset, bool parity)
    {
        int i = blockIdx.z * blockDim.z + threadIdx.z;
        int j = blockIdx.y * blockDim.y + threadIdx.y;
        int k = blockIdx.x * blockDim.x + threadIdx.x;

        // inner grid ops, avoid boundaries
        if (i < start || i > end || j < 1 || j > Nr - 2 || k < 1 || k > Nc - 2) return;

        // use global index to compute parity
        if (((offset + i - 1) + j + k) % 2 != parity) return;

        int at = idx(i, j, k, Nr, Nc);

        double c = curr_d[at];

        double stencil = (curr_d[idx(i - 1, j, k, Nr, Nc)] + curr_d[idx(i + 1, j, k, Nr, Nc)] +
                          curr_d[idx(i, j - 1, k, Nr, Nc)] + curr_d[idx(i, j + 1, k, Nr, Nc)] +
                          curr_d[idx(i, j, k - 1, Nr, Nc)] + curr_d[idx(i, j, k + 1, Nr, Nc)] -
                          dx_2 * rhs_d[at]) * RECIP_6;

        curr_d[at] = c + omega * (stencil - c);

    }

    __global__ void res_kernel(double* curr_d, double* rhs_d,
                                int Nr, int Nc, double recip_dx_2,
                                int start, int end, double* res_d)
    {
        extern __shared__ double max_res[];

        int curr_idx = idx(threadIdx.z, threadIdx.y, threadIdx.x, blockDim.y, blockDim.x);
        int nwarps = blockDim.z * blockDim.y * blockDim.x / warpSize;
        int warp_idx = curr_idx / warpSize;
        int lane_idx = curr_idx % warpSize;

        int i = blockIdx.z * blockDim.z + threadIdx.z;
        int j = blockIdx.y * blockDim.y + threadIdx.y;
        int k = blockIdx.x * blockDim.x + threadIdx.x;

        // init residual
        double res = 0.0;

        // inner grid ops, avoid boundaries
        if (i >= start && i <= end && j >= 1 && j <= Nr - 2 && k >= 1 && k <= Nc - 2) {
            int center = idx(i, j, k, Nr, Nc);            
            // store max (residual = actual laplacian - iterative laplacian)
            res = fabs(rhs_d[center] - (curr_d[idx(i - 1, j, k, Nr, Nc)] + curr_d[idx(i + 1, j, k, Nr, Nc)] + 
                                        curr_d[idx(i, j - 1, k, Nr, Nc)] + curr_d[idx(i, j + 1, k, Nr, Nc)] + 
                                        curr_d[idx(i, j, k - 1, Nr, Nc)] + curr_d[idx(i, j, k + 1, Nr, Nc)] -
                                        6.0 * curr_d[center]) * recip_dx_2);
        }

        __syncwarp();

        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            res = fmax(res, __shfl_down(res, offset));
        }

        if (lane_idx == 0) max_res[warp_idx] = res;

        __syncthreads();

        if (curr_idx == 0) {
            for (int w = 1; w < nwarps; w++) res = fmax(res, max_res[w]);
            atomicMax(res_d, res);
        }
        
    }

    __global__ void err_kernel(double* curr_d, double* soln_d,
                                int Ns, int Nr, int Nc, bool sq, double* err_d)
    {
        extern __shared__ double sum[];

        int curr_idx = idx(threadIdx.z, threadIdx.y, threadIdx.x, blockDim.y, blockDim.x);
        int nwarps = blockDim.z * blockDim.y * blockDim.x / warpSize;
        int warp_idx = curr_idx / warpSize;
        int lane_idx = curr_idx % warpSize;

        int i = blockIdx.z * blockDim.z + threadIdx.z;
        int j = blockIdx.y * blockDim.y + threadIdx.y;
        int k = blockIdx.x * blockDim.x + threadIdx.x;

        // init error
        double err = 0.0;

        // outer grid ops, avoid halos at i = 1 and i = Ns - 2
        if (i >= 1 && i <= Ns - 2 && j >= 0 && j <= Nr - 1 && k >= 0 && k <= Nc - 1) {
            int center = idx(i, j, k, Nr, Nc);
            err = fabs(curr_d[center] - soln_d[center]);
            if (sq) err *= err;
        }

        __syncwarp();

        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            err += __shfl_down(err, offset);
        }

        if (lane_idx == 0) sum[warp_idx] = err;

        __syncthreads();

        if (curr_idx == 0) {
            for (int w = 1; w < nwarps; w++) err += sum[w];
            atomicAdd(err_d, err);
        }
    }

}
