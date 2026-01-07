/**
 * Thermosolver
 * 
 * Name: timer.h
 * 
 * Author: Samson Tsegai
 */

#pragma once

#include <hip/hip_runtime.h>

struct Timer {
    hipEvent_t start_t, end_t;
    
    Timer() {
        hipEventCreate(&start_t);
        hipEventCreate(&end_t);
    }

    ~Timer() {
        hipEventDestroy(start_t);
        hipEventDestroy(end_t);
    }

    void start() {
        hipEventRecord(start_t, 0); 
    }

    void end(double& time) {
        float ms;
        hipEventRecord(end_t, 0);
        hipEventSynchronize(end_t);
        hipEventElapsedTime(&ms, start_t, end_t);
        time += ms * 1e-3; // ms -> sec
    }
};
