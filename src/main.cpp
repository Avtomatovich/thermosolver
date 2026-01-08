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
 * @brief Main function for 3D heat equation solver program.
 * 
 * Arguments: 
 *      argv[1] = Grid size (must be >= 5 and multiple of 5)
 *      argv[2] = Toggles convergence logging (0 for false, any non-zero integer for true)
 *
 * @param argc Number of arguments
 * @param argv Array of string arguments
 * @return Exit code for program (0 for success, 1 for failure)
 */
int main(int argc, char* argv[]) {
    try {
        if (argc != 3) throw std::invalid_argument("Insufficient arguments.");

        // init grid size
        int N = std::stoi(argv[1]);
        if (N < 5 || N % 5 != 0) throw std::invalid_argument("Invalid grid size.");

        // toggle convergence logging
        bool log = std::stoi(argv[2]);

        Utils::write_head();

        Solver{N}.solve(log);

    } catch (std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
    }

    std::cout << "Execution complete." << std::endl;

    return 0;
}
