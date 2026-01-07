/**
 * Thermosolver
 * 
 * Name: main.cpp
 * 
 * Author: Samson Tsegai
 */

#include <mpi.h>
#include <string>
#include <exception>
#include "utils/utils.h"
#include "solver/solver.h"

/**
 * @brief Main function for 3D heat equation solver program.
 * 
 * Arguments: 
 *      argv[1] = Solver method
 *          0 = Jacobi
 *          1 = Red-Black Gauss-Seidel (RBGS)
 *          2 = Successive Over-Relaxation (SOR)
 *      argv[2] = Grid size (defaults to loop in range [50, 200] if < 5 or not multiple of 5)
 *      argv[3] = Toggles convergence logging (0 for false, any non-zero int for true)
 *
 * @param argc Number of arguments
 * @param argv Array of string arguments
 * @return Exit code for program (0 for success, 1 for failure)
 */
 int main(int argc, char* argv[]) {
    // init MPI
    MPI_Init(&argc, &argv);

    // exit if MPI is not initialized
    int flag;
    MPI_Initialized(&flag);
    if (!flag) {
        std::cerr << "MPI failed to initialize." << std::endl;
        return 1;
    }
    
    // fetch current rank
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    try {
        if (argc != 4) throw std::invalid_argument("Insufficient arguments.");

        // init method
        int type = std::stoi(argv[1]);
        if (type < 0 || type > 2) throw std::invalid_argument("Invalid method type.");
        Method method = static_cast<Method>(type);

        // init grid size
        int N = std::stoi(argv[2]);
        
        // toggle convergence logging
        bool log = std::stoi(argv[3]);
        
        if (rank == 0) Utils::write_head();
        
        if (N < 5 || N % 5 != 0) {
            if (rank == 0) std::cout << "Invalid grid size, iterating from N = 50 to 200" << std::endl;
            
            for (int dim = 50; dim <= 200; dim += 25) Solver{dim, method}.solve(false);
        } else {
            Solver{N, method}.solve(log);
        }

    } catch (std::exception& e) {
        if (rank == 0) std::cerr << "Exception: " << e.what() << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    if (rank == 0) std::cout << "Execution complete." << std::endl;

    MPI_Finalize();

    return 0;
}
