/**
 * Thermosolver
 * 
 * Name: gpu_func.cpp
 * 
 * Author: Samson Tsegai
 */

#include "gpu_func.h"

namespace GPUFunc {

    __global__ void rbgs_kernel(double* curr_d, double* prev_d,
                                int Ns, int Nr, int Nc, double dx_2, bool parity)
    {
        int i = blockIdx.z * blockDim.z + threadIdx.z;
        int j = blockIdx.y * blockDim.y + threadIdx.y;
        int k = blockIdx.x * blockDim.x + threadIdx.x;

        // inner grid ops, avoid boundaries
        if (i < 1 || i > Ns - 2 || j < 1 || j > Nr - 2 || k < 1 || k > Nc - 2) return;

        if ((i + j + k) % 2 != parity) return;

        int at = idx(i, j, k, Nr, Nc);

        curr_d[at] = (curr_d[idx(i - 1, j, k, Nr, Nc)] + curr_d[idx(i + 1, j, k, Nr, Nc)] +
                      curr_d[idx(i, j - 1, k, Nr, Nc)] + curr_d[idx(i, j + 1, k, Nr, Nc)] +
                      curr_d[idx(i, j, k - 1, Nr, Nc)] + curr_d[idx(i, j, k + 1, Nr, Nc)] -
                      dx_2 * rhs_d[at]) * RECIP_6;

    }

    __global__ void res_kernel(double* curr_d, double* prev_d,
                                int Ns, int Nr, int Nc, double recip_dx_2, double* res_d)
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
        if (i >= 1 && i <= Ns - 2 && j >= 1 && j <= Nr - 2 && k >= 1 && k <= Nc - 2) {
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

}
