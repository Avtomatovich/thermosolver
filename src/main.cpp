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
 *      argv[1] = Solver method
 *          0 = Forward Time-Centered Space (FTCS)
 *          1 = Crank-Nicolson (CN)
 *      argv[2] = Grid size (must be >= 5 and multiple of 5)
 *      argv[3] = Toggles performance logging (optional, 0 for false, any non-zero integer for true)
 *      argv[4] = Toggles diagnostics logging (optional, 0 for false, any non-zero integer for true)
 *
 * @param argc Number of arguments
 * @param argv Array of string arguments
 * @return Exit code for program (0 for success, 1 for failure)
 */
int main(int argc, char* argv[]) {
    try {
        if (argc < 3) throw std::invalid_argument("Insufficient arguments.");

        // init method
        int type = std::stoi(argv[1]);
        if (type < 0 || type > 1) throw std::invalid_argument("Invalid method.");
        Method method = static_cast<Method>(type);

        // init grid size
        int dim = std::stoi(argv[2]);
        if (dim < 5 || dim % 5 != 0) throw std::invalid_argument("Invalid grid size.");

        bool perf_log = false, diag_log = false;
        // handle performance toggle if 4 args present
        if (argc == 4) perf_log = std::stoi(argv[3]);
        // handle diagnostics toggle if 5 args present
        if (argc == 5) diag_log = std::stoi(argv[4]);

        if (perf_log) Utils::write_head();

        Solver{dim, method, perf_log, diag_log}.solve();

    } catch (std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
    }

    std::cout << "Execution complete." << std::endl;

    return 0;
}
