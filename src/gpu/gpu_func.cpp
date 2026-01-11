/**
 * Thermosolver
 * 
 * Name: gpu_func.cpp
 * 
 * Author: Samson Tsegai
 */

#include "gpu_func.h"

namespace GPUFunc {

    __global__ void ftcs_kernel(double* __restrict__ curr_d, const double* __restrict__ prev_d,
                                int Ns, int Nr, int Nc, double ftcs_coeff, double r)
    {
        int i = blockIdx.z * blockDim.z + threadIdx.z;
        int j = blockIdx.y * blockDim.y + threadIdx.y;
        int k = blockIdx.x * blockDim.x + threadIdx.x;

        // inner grid ops, avoid boundaries
        if (i < 1 || i > Ns - 2 || j < 1 || j > Nr - 2 || k < 1 || k > Nc - 2) return;

        int at = idx(i, j, k, Nr, Nc);

        curr_d[at] = ftcs_coeff * prev_d[at] + r * neighbors(prev_d, i, j, k);
    }

    __global__ void cn_kernel(double* __restrict__ curr_d, const double* __restrict__ prev_d,
                              double* res_d, int Ns, int Nr, int Nc, 
                              double cn_coeff, double r_half, double recip_denom, bool parity)
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
            if ((i + j + k & 1) == parity) {
                int at = idx(i, j, k, Nr, Nc);            
                double c = curr_d[at];
                double u = (cn_coeff * prev_d[at] + 
                            r_half * ( neighbors(prev_d, i, j, k) + neighbors(curr_d, i, j, k) )) * 
                            recip_denom;
                curr_d[at] = u;
                res = fabs(u - c);
            }
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
