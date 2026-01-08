/**
 * Thermosolver
 * 
 * Name: main.cpp
 * 
 * Author: Samson Tsegai
 */

#include <string>
#include <stdexcept>
#include "utils/utils.h"
#include "solver/solver.h"

/**
 * @brief Main function for 3D heat equation solver using Forward Time-Centered Space (FTCS).
 * 
 * Arguments: 
 *      argv[1] = Grid size (must be >= 5 and multiple of 5)
 *      argv[2] = Toggles performance logging (0 for false, any non-zero integer for true)
 *      argv[3] = Toggles convergence logging (0 for false, any non-zero integer for true)
 *
 * @param argc Number of arguments
 * @param argv Array of string arguments
 * @return Exit code for program (0 for success, 1 for failure)
 */
int main(int argc, char* argv[]) {
    try {
        if (argc != 4) throw std::invalid_argument("Insufficient arguments.");

        // init grid size
        int N = std::stoi(argv[1]);
        if (N < 5 || N % 5 != 0) throw std::invalid_argument("Invalid grid size.");

        // toggle logging
        bool perf_log = std::stoi(argv[2]);
        bool conv_log = std::stoi(argv[3]);

        if (perf_log) Utils::write_head();

        Solver{N, perf_log, conv_log}.solve();

    } catch (std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
    }

    std::cout << "Execution complete." << std::endl;

    return 0;
}
