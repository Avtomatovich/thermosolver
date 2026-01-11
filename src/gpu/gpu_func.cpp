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

        // inner grid
        if (i < 1 || i > Ns - 2 || j < 1 || j > Nr - 2 || k < 1 || k > Nc - 2) return;

        int at = idx(i, j, k, Nr, Nc);

        curr_d[at] = ftcs_coeff * prev_d[at] + r * neighbors(prev_d, i, j, k, Nr, Nc);
    }

    __global__ void cn_kernel(double* __restrict__ curr_d, const double* __restrict__ prev_d,
                              double* res_d, int Ns, int Nr, int Nc, 
                              double cn_coeff, double r_half, double recip_denom, bool parity)
    { 
        int nwarps = (blockDim.z * blockDim.y * blockDim.x + warpSize - 1) / warpSize;
        
        // array of per-warp reduction vars
        __shared__ double max_arr[nwarps];
        
        int curr_idx = idx(threadIdx.z, threadIdx.y, threadIdx.x, blockDim.y, blockDim.x);
        int warp_idx = curr_idx / warpSize;
        int lane_idx = curr_idx % warpSize;

        int i = blockIdx.z * blockDim.z + threadIdx.z;
        int j = blockIdx.y * blockDim.y + threadIdx.y;
        int k = blockIdx.x * blockDim.x + threadIdx.x;

        // init residual
        double res = 0.0;

        // inner grid
        if (i >= 1 && i <= Ns - 2 && j >= 1 && j <= Nr - 2 && k >= 1 && k <= Nc - 2) {
            if ((i + j + k & 1) == parity) {
                int at = idx(i, j, k, Nr, Nc);            
                double c = curr_d[at];
                double u = (cn_coeff * prev_d[at] + 
                            r_half * ( neighbors(prev_d, i, j, k, Nr, Nc) + neighbors(curr_d, i, j, k, Nr, Nc) )) * 
                            recip_denom;
                curr_d[at] = u;
                res = fabs(u - c);
            }
        }

        __syncwarp();
        
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            res = fmax(res, __shfl_down(res, offset));
        }
    
        if (lane_idx == 0) max_arr[warp_idx] = res;
    
        __syncthreads();
    
        if (curr_idx == 0) {
            for (int w = 1; w < nwarps; w++) res = fmax(res, max_arr[w]);
            atomicMax(res_d, res);
        }
    }

        __global__ void diag_kernel(const double* __restrict__ curr_d, int Ns, int Nr, int Nc,
                                    double* min_d, double* max_d, double* total_d)
        {
            int nwarps = (blockDim.z * blockDim.y * blockDim.x + warpSize - 1) / warpSize;
            
            // arrays of per-warp reduction vars
            __shared__ double min_arr[nwarps], max_arr[nwarps], sum_arr[nwarps];
            
            int thread_idx = idx(threadIdx.z, threadIdx.y, threadIdx.x, blockDim.y, blockDim.x);
            int warp_idx = thread_idx / warpSize;
            int lane_idx = thread_idx % warpSize;

            int i = blockIdx.z * blockDim.z + threadIdx.z;
            int j = blockIdx.y * blockDim.y + threadIdx.y;
            int k = blockIdx.x * blockDim.x + threadIdx.x;

            double thread_min = 1.0, thread_max = 0.0, thread_sum = 0.0;
            if (i >= 0 && i <= Ns - 1 && j >= 0 && j <= Nr - 1 && k >= 0 && k <= Nc - 1) {
                thread_min = thread_max = thread_sum = curr_d[idx(i, j, k, Nr, Nc)];
            }

            __syncwarp();

            for (int offset = warpSize / 2; offset > 0; offset /= 2) {
                thread_min = fmin(thread_min, __shfl_down(thread_min, offset));
                thread_max = fmax(thread_max, __shfl_down(thread_max, offset));
                thread_sum += __shfl_down(thread_sum, offset);
            }

            if (lane_idx == 0) {
                min_arr[warp_idx] = thread_min;
                max_arr[warp_idx] = thread_max;
                sum_arr[warp_idx] = thread_sum;
            }

            __syncthreads();

            if (thread_idx == 0) {
                for (int w = 1; w < nwarps; w++) {
                    thread_min = fmin(thread_min, min_arr[w]);
                    thread_max = fmax(thread_max, max_arr[w]);
                    thread_sum += sum_arr[w];
                }
                atomicMin(min_d, thread_min);
                atomicMax(max_d, thread_max);
                atomicAdd(total_d, thread_sum);
            }

        }

}
