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
        for (const std::string& filename : {solve_file, diag_file}) {
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

    void solve_stats(const Stats& stats, Method method) {
        double t = stats.solve_t;
        
        double bytes, flops;
        switch (method) {
            case Method::FTCS:
                // bytes per step = 7 load + 1 write for 8 bytes each
                bytes = (7.0 + 1.0) * sizeof(double) * stats.in_size; // total bytes
                // flops per step = 2 mul + 6 add
                flops = (2.0 + 6.0) * stats.in_size; // total flops
                break;
            case Method::CN:
                // bytes per step = 14 load + 1 write for 8 bytes each
                bytes = (14.0 + 1.0) * sizeof(double) * stats.in_size; // total bytes
                // flops per step = 12 add + 3 mul + 1 sub + 1 max + 1 abs
                flops = (12.0 + 3.0 + 1.0 + 1.0 + 1.0) * stats.in_size; // total flops
                break;
        }

        printf("* ==Solver Stats==\n");
	    print_stats(solve_file, stats, t, bytes, flops);
    }

    void diag_stats(const Stats& stats) {
        double t = stats.diag_t;
        // bytes per step = 2 load for 8 bytes each
        double bytes = 2.0 * sizeof(double) * stats.out_size; // total bytes
        // flops per step = 1 sub + 1 abs + 2 max + 1 min + 1 add
        double flops = (1.0 + 1.0 + 2.0 + 1.0 + 1.0) * stats.out_size; // total flops

        printf("* ==Diagnostic Stats==\n");
	    print_stats(diag_file, stats, t, bytes, flops);
    }

    void write_diag(const Stats& stats) {
        std::ofstream file(diag_file);
        if (!file.is_open()) throw std::runtime_error("Failed to open: " + diag_file);

        file << "N,steps,min,max,total" << std::endl;
        for (int i = 0; i < stats.diag_data.size(); i++) {
            const Diag& diag = stats.diag_data[i];
            file << stats.N << "," << i << "," << diag.min << "," << diag.max << "," << diag.total << std::endl;
        }

        file.close();
    }

}
