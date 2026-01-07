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

    // relaxation factor for SOR method
    static __device__ constexpr double omega = 1.8;

    static __device__ constexpr double RECIP_6 = 1.0 / 6.0;

    static inline __device__ int idx(int i, int j, int k, int Nr, int Nc) {
        return i * Nr * Nc + j * Nc + k;
    }

    inline void set_val_device(double* var_d, double val) {
        hipMemset(var_d, val, sizeof(double));
    }

    inline void to_device(const std::vector<double>& src, double* dst) {
        // NOTE: size in bytes
        hipMemcpy(dst, src.data(), src.size() * sizeof(double), hipMemcpyHostToDevice);
    }

    inline void to_device(double* src, double* dst) {
        // NOTE: size in bytes
        hipMemcpy(dst, src, sizeof(double), hipMemcpyHostToDevice);
    }

    inline void to_host(double* src, std::vector<double>& dst) {
        // NOTE: size in bytes
        hipMemcpy(dst.data(), src, dst.size() * sizeof(double), hipMemcpyDeviceToHost);
    }

    inline void to_host(double* src, double* dst) {
        // NOTE: size in bytes
        hipMemcpy(dst, src, sizeof(double), hipMemcpyDeviceToHost);
    }

    __global__ void jacobi_kernel(double* prev_d, double* curr_d, double* rhs_d,
                                  int Nr, int Nc, double dx_2, 
                                  int start, int end);

    __global__ void rbgs_kernel(double* curr_d, double* rhs_d,
                                int Nr, int Nc, double dx_2, 
                                int start, int end, int offset, bool parity);

    __global__ void sor_kernel(double* curr_d, double* rhs_d,
                                int Nr, int Nc, double dx_2, 
                                int start, int end, int offset, bool parity);

    __global__ void res_kernel(double* curr_d, double* rhs_d,
                                int Nr, int Nc, double recip_dx_2,
                                int start, int end, double* res_d);
    
    __global__ void err_kernel(double* curr_d, double* soln_d,
                                int Ns, int Nr, int Nc, bool sq, double* err_d);

}
