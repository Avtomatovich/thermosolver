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
 *      argv[3] = Number of time steps (must be > 0)
 *      argv[4] = Toggles heatmap logging (optional, 0 for false, any non-zero integer for true)
 *      argv[5] = Toggles diagnostics logging (optional, 0 for false, any non-zero integer for true)
 *      argv[6] = Toggles performance logging (optional, 0 for false, any non-zero integer for true)
 *
 * @param argc Number of arguments
 * @param argv Array of string arguments
 * @return Exit code for program (0 for success, 1 for failure)
 */
int main(int argc, char* argv[]) {
    try {
        if (argc < 4) throw std::invalid_argument("Insufficient arguments.");

        // solver method
        int type = std::stoi(argv[1]);
        if (type < 0 || type > 1) throw std::invalid_argument("Invalid method.");
        Method method = static_cast<Method>(type);

        // grid size
        int dim = std::stoi(argv[2]);
        if (dim < 5 || dim % 5 != 0) throw std::invalid_argument("Invalid grid size.");

        // time steps
        int nsteps = std::stoi(argv[3]);
        if (nsteps < 1) throw std::invalid_argument("Invalid number of time steps.");

        bool heat_log = false, perf_log = false, diag_log = false;
        // parse heatmap toggle if 5 args present
        if (argc >= 5) heat_log = std::stoi(argv[4]);
        // parse diagnostics toggle if 6 args present
        if (argc >= 6) diag_log = std::stoi(argv[5]);
        // parse performance toggle if 7 args present
        if (argc >= 7) perf_log = std::stoi(argv[6]);

        if (perf_log) Utils::write_head(diag_log, method);

        Solver{dim, method, diag_log, perf_log}.solve(nsteps, heat_log);

    } catch (std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
    }

    std::cout << "Execution complete." << std::endl;

    return 0;
}
