/**
 * Thermosolver
 * 
 * Name: timer.h
 * 
 * Author: Samson Tsegai
 */

#pragma once

#include <chrono>

using namespace std::chrono;

struct Timer {
    high_resolution_clock::time_point start_t, end_t;

    void start() {
        start_t = high_resolution_clock::now();
    }

    void end(double& time) {
        end_t = high_resolution_clock::now();
        time += duration<double>(end_t - start_t).count();
    }
};
