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
        for (const std::string& filename : {solve_file, res_file}) {
            write_file(filename, header, std::ios::out);
        }
    }

    void print_stats(const std::string& file, const Stats& stats, double t, double bytes, double flops) {
        double ai = flops / bytes; // arithmetic intensity
        double bw = bytes * 1e-9 / t; // in GB/sec
        double fr = flops * 1e-9 / t; // in GFLOPS/sec

	    printf("\t\t* Time: %g sec \n\t\t* FLOPS: %.1g flops \n\t\t* Memory: %.1g bytes\n", t, flops, bytes);
	    printf("\t* ==Results==\n");
	    printf("\t\t* Flop Rate: %g GF/s \n\t\t* Bandwidth: %g GB/s \n\t\t* AI: %g FLOPS/byte\n\n", fr, bw, ai);
        fflush(stdout);

        if (stats.perf_log) {
            std::stringstream row;
            row << stats.N << "," << stats.steps << "," << t << "," << fr << "," << ai << "," << bw;
            write_file(file, row.str(), std::ios::app);
        }
    }

    void solve_stats(const Stats& stats) {
        double t = stats.solve_t;
        // bytes per step = 8 load + 1 write for 8 bytes each
        double bytes = (8.0 + 1.0) * sizeof(double) * stats.in_size; // total bytes
        // flops per step = 6 add + 1 sub + 4 mul
        double flops = (6.0 + 1.0 + 4.0) * stats.in_size; // total flops

        printf("* ==Solver Stats==\n");
	    print_stats(solve_file, stats, t, bytes, flops);
    }

    void res_stats(const Stats& stats) {
        double t = stats.res_t;
        // bytes per step = 2 load for 8 bytes each
        double bytes = 2.0 * sizeof(double) * stats.out_size; // total bytes
        // flops per step = 1 sub + 1 abs + 1 max
        double flops = (1.0 + 1.0 + 1.0) * stats.out_size; // total flops

        printf("* ==Residual Stats==\n");
	    print_stats(res_file, stats, t, bytes, flops);
    }

    void write_conv(const Stats& stats) {
        std::ofstream file(conv_file);
        if (!file.is_open()) throw std::runtime_error("Failed to open: " + conv_file);

        file << "N,steps,residual" << std::endl;
        for (int i = 0; i < stats.conv_data.size(); i++) {
            file << stats.N << "," << i << "," << stats.conv_data[i] << std::endl;
        }

        file.close();
    }

}
