/**
 * Thermosolver
 * 
 * Name: gpu_func.h
 * 
 * Author: Samson Tsegai
 */

#include <vector>
#include <hip/hip_runtime.h>

namespace GPUFunc {

    static inline __device__ int idx(int i, int j, int k, int Nr, int Nc) {
        return i * Nr * Nc + j * Nc + k;
    }

    inline void to_device(const std::vector<double>& src, double* dst) {
        // NOTE: size in bytes
        hipMemcpy(dst, src.data(), src.size() * sizeof(double), hipMemcpyHostToDevice);
    }

    inline void to_host(double* src, std::vector<double>& dst) {
        // NOTE: size in bytes
        hipMemcpy(dst.data(), src, dst.size() * sizeof(double), hipMemcpyDeviceToHost);
    }

    inline void to_host(double* src, double* dst) {
        // NOTE: size in bytes
        hipMemcpy(dst, src, sizeof(double), hipMemcpyDeviceToHost);
    }

    inline __device__ double neighbors(const double* __restrict__ m, int i, int j, int k) {
        return  m[idx(i - 1, j, k)] + m[idx(i + 1, j, k)] + 
                m[idx(i, j - 1, k)] + m[idx(i, j + 1, k)] + 
                m[idx(i, j, k - 1)] + m[idx(i, j, k + 1)];
    }

    __global__ void ftcs_kernel(double* __restrict__ curr_d, const double* __restrict__ prev_d,
                                int Ns, int Nr, int Nc, double ftcs_coeff, double r);

    __global__ void cn_kernel(double* __restrict__ curr_d, const double* __restrict__ prev_d,
                              double* res_d, int Ns, int Nr, int Nc, 
                              double cn_coeff, double r_half, double recip_denom, bool parity);

}
