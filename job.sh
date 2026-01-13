#!/bin/bash

#SBATCH -J thermosolver       # Job name
#SBATCH -o output/hip.%j.txt  # Name of stdout output file (%j expands to jobId)
#SBATCH -N 1                  # Total number of nodes requested
#SBATCH -n 1                  # Total number of mpi tasks requested
#SBATCH -t 00:10:00           # Run time (hh:mm:ss) - 10 minutes
#SBATCH -p devel              # Desired partition

# %J maps to jobId.stepId

# args = 
#  Solver method (0 = FTCS, 1 = CN)
#  Grid size (must be >= 5 and multiple of 5)
#  Number of time steps (must be > 0)
#  Heatmap logging (optional, 0 = false, non-zero int = true)
#  Diagnostics logging (optional, 0 = false, non-zero int = true)
#  Performance logging (optional, 0 = false, non-zero int = true)
./build/main.exe 0 100 1000

# Profiling command
# rocprofv3 --kernel-trace --memory-copy-trace --output-format pftrace -- ./build/main.exe 1 100 1000

echo "HIP Thermosolver job complete."
