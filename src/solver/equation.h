/**
 * Thermosolver
 * 
 * Name: equation.h
 * 
 * Author: Samson Tsegai
 */

#pragma once

#include <cmath>

namespace Equation {

    /**
     * How to solve for phi:
     * - Start with empty matrix
     * - Set edges of empty matrix to phi's value (Dirichlet boundaries)
     * - Estimate 2nd-degree partial derivatives (Laplacian) of phi over empty matrix with update formula
     * - Stop iterating when estimates sufficiently converge to actual Laplacian (defined in f)
     * - You now have an approximate solution to phi
     */

    static constexpr int n = 2, m = 2, k = 2;

    inline double phi(double x, double y, double z) {
        return sin(n * M_PI * x) * cos(m * M_PI * y) * sin(k * M_PI * z);
    }

    inline double f(double x, double y, double z) {
        return -M_PI*M_PI * (n*n + m*m + k*k) * phi(x, y, z);
    }

}