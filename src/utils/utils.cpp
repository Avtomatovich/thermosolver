/**
 * Thermosolver
 * 
 * Name: utils.cpp
 * 
 * Author: Samson Tsegai
 */

#include "utils.h"
#include <fstream>
#include <sstream>
#include <stdexcept>

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
    
    void write_head() {
        std::string header = "size,steps,time,flop_rate,arithmetic_intensity,bandwidth";
        for (const std::string& filename : {solve_perf_file, diag_perf_file}) {
            write_file(filename, header, std::ios::out);
        }
    }

    void print_stats(const std::string& file, const Stats& stats, double t, double bytes, double flops) {
        double ai = flops / bytes; // arithmetic intensity
        double bw = bytes * 1e-9 / t; // in GB/sec
        double fr = flops * 1e-9 / t; // in GFLOPS/sec

	    printf("\t\t* Time: %G sec \n\t\t* FLOPS: %G flops \n\t\t* Memory: %G bytes\n", t, flops, bytes);
	    printf("\t\t* Flop Rate: %G GF/s \n\t\t* Bandwidth: %G GB/s \n\t\t* AI: %G FLOPS/byte\n\n", fr, bw, ai);
        fflush(stdout);

        if (stats.perf_log) {
            std::stringstream row;
            row << stats.N << "," << stats.steps << "," << t << "," << fr << "," << ai << "," << bw;
            write_file(file, row.str(), std::ios::app);
        }
    }

    // TODO: recompute bytes and flops (GPU reduction adds complexity)
    void solve_stats(const Stats& stats, Method method) {
        double t = stats.solve_t;
        
        double bytes, flops;
        switch (method) {
            case Method::FTCS:
                // bytes per cell = 7 load + 1 write for 8 bytes each
                bytes = (7.0 + 1.0) * sizeof(double); 
                bytes *= stats.in_size * stats.steps; // total bytes

                // flops per cell = 2 mul + 6 add
                flops = 2.0 + 6.0; 
                flops *= stats.in_size * stats.steps; // total flops
                break;
            case Method::CN:
                // bytes per cell = 14 load + 1 write for 8 bytes each
                bytes = (14.0 + 1.0) * sizeof(double); 
                bytes *= stats.in_size * stats.steps * stats.cn_steps; // total bytes

                // flops per cell = 12 add + 3 mul + 1 sub + 1 max + 1 abs
                flops = 12.0 + 3.0 + 1.0 + 1.0 + 1.0;
                flops *= stats.in_size * stats.steps * stats.cn_steps; // total flops
                break;
        }

        printf("* ==Solver Stats==\n");
	    print_stats(solve_perf_file, stats, t, bytes, flops);
    }

    // TODO: recompute bytes and flops (GPU reduction adds complexity)
    void diag_stats(const Stats& stats) {
        double t = stats.diag_t;

        // bytes per cell = 1 load for 8 bytes each
        double bytes = 1.0 * sizeof(double);
        bytes *= stats.out_size * stats.steps; // total bytes

        // flops per cell = 1 max + 1 min + 1 add
        double flops = 1.0 + 1.0 + 1.0;
        flops *= stats.out_size * stats.steps; // total flops

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
