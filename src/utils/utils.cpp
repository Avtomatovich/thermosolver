/**
 * Thermosolver
 * 
 * Name: utils.cpp
 * 
 * Author: Samson Tsegai
 */

#include "utils.h"
#include <cmath>
#include <fstream>
#include <sstream>

namespace Utils {

    void write_file(const std::string& filename, 
                    const std::string& line, 
                    std::ios_base::openmode mode)
    {
        std::fstream file(filename, mode);
        if (!file.is_open()) throw std::runtime_error("Failed to open: " + filename);
        file << line << std::endl;
        file.close();
    }
    
    void write_head(bool diag_log) {
        std::string header = "size,steps,time,flop_rate,arithmetic_intensity,bandwidth";
        
        write_file(solve_perf_file, header, std::ios::out);

        if (diag_log) write_file(diag_perf_file, header, std::ios::out);
    }

    void print_stats(const std::string& file, const Stats& stats, double t, double bytes, double flops) {
        double ai = flops / bytes; // arithmetic intensity
        double bw = bytes * 1e-9 / t; // in GB/sec
        double fr = flops * 1e-9 / t; // in GFLOPS/sec

	    printf("\t* Time: %G sec \n\t* FLOPS: %G flops \n\t* Memory: %G bytes\n", t, flops, bytes);
	    printf("\t* Flop Rate: %G GF/s \n\t* Bandwidth: %G GB/s \n\t* AI: %G FLOPS/byte\n\n", fr, bw, ai);
        fflush(stdout);

        if (stats.perf_log) {
            std::stringstream row;
            row << stats.N << "," << stats.steps << "," << t << "," << fr << "," << ai << "," << bw;
            write_file(file, row.str(), std::ios::app);
        }
    }

    void solve_stats(const Stats& stats, Method method) {
        double t = stats.solve_t;
        
        double bytes, flops;
        if (method == Method::FTCS) {
            // bytes per cell = 7 load + 1 write for 8 bytes each
            bytes = (7.0 + 1.0) * sizeof(double) * stats.in_size * stats.steps;

            // flops per cell = 2 mul + 6 add
            flops = (2.0 + 6.0) * stats.in_size * stats.steps;
        } else {
            // number of warps in block (512 threads per block / 64 = 8)
            int nwarps = (stats.nthreads + WARP_SIZE - 1) / WARP_SIZE;

            // bytes per update = 14 load + 1 write for 8 bytes each
            double update_bytes = (14.0 + 1.0) * sizeof(double) * stats.in_size * stats.cn_steps;
            // bytes per shuffle = 1 write of 8 bytes for each warp in block for 2 launches
            double shuffle_bytes = 1.0 * sizeof(double) * nwarps * stats.nblocks * 2.0 * stats.cn_steps;
            // bytes per block-tier reduction = (nwarps - 1) number of load ops in block for 2 launches
            double block_bytes = (nwarps - 1.0) * sizeof(double) * stats.nblocks * 2.0 * stats.cn_steps;
            // total bytes
            bytes = update_bytes + shuffle_bytes + block_bytes;

            // flops per update = 12 add + 3 mul + 1 sub + 1 abs
            double update_flops = (12.0 + 3.0 + 1.0 + 1.0) * stats.in_size * stats.cn_steps;
            // flops per shuffle = log_2(64) = 6 max ops
            double shuffle_flops = log2(WARP_SIZE) * stats.nthreads * stats.nblocks * 2.0 * stats.cn_steps;
            // flops per block-tier reduction = nwarps number of max ops
            double block_flops = nwarps * stats.nblocks * 2.0 * stats.cn_steps;
            // total flops
            flops = update_flops + shuffle_flops + block_flops;
        }

        printf("* ==Solver Stats==\n");
	    print_stats(solve_perf_file, stats, t, bytes, flops);
    }

    void diag_stats(const Stats& stats) {
        double t = stats.diag_t;

        // number of warps in block (512 threads per block / 64 = 8)
        int nwarps = (stats.nthreads + WARP_SIZE - 1) / WARP_SIZE;

        // bytes per grid cell = 1 load of 8 bytes
        double cell_bytes = 1.0 * sizeof(double) * stats.out_size * stats.steps;
        // bytes per shuffle = 3 writes of 8 bytes each for first lane of each warp in block
        double shuffle_bytes = 3.0 * sizeof(double) * nwarps * stats.nblocks * stats.steps;
        // bytes per block-tier reduction = 3 * (nwarps - 1) number of load ops in block
        double block_bytes = 3.0 * (nwarps - 1.0) * sizeof(double) * stats.nblocks * stats.steps;
        // total bytes
        double bytes = cell_bytes + shuffle_bytes + block_bytes;
        
        // flops per shuffle = 1 max + 1 min + 1 add for log_2(64) iterations
        double shuffle_flops = (1.0 + 1.0 + 1.0) * log2(WARP_SIZE) * stats.nthreads * stats.nblocks * stats.steps;
        // flops per block-tier reduction = (1 max + 1 min + 1 add) repeated nwarps times
        double block_flops = (1.0 + 1.0 + 1.0) * nwarps * stats.nblocks * stats.steps;
        // total flops
        double flops = shuffle_flops + block_flops;

        printf("* ==Diagnostic Stats==\n");
	    print_stats(diag_perf_file, stats, t, bytes, flops);
    }

    void write_diag(const Stats& stats) {
        std::ofstream file(diag_data_file);
        if (!file.is_open()) throw std::runtime_error("Failed to open: " + diag_data_file);

        file << "N,steps,min,max,total" << std::endl;
        for (int i = 0; i < stats.diag_data.size(); i++) {
            const Diag& diag = stats.diag_data[i];
            file << stats.N << "," << i << "," << diag.min << "," << diag.max << "," << diag.total << std::endl;
        }

        file.close();
    }

}
