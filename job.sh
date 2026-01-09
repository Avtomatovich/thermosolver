#!/bin/bash

#SBATCH -J thermosolver       # Job name
#SBATCH -o output/omp.%j.txt  # Name of stdout output file (%j expands to jobId)
#SBATCH -N 1                  # Total number of nodes requested
#SBATCH -n 1                  # Total number of mpi tasks requested
#SBATCH -t 00:15:00           # Run time (hh:mm:ss) - 15 minutes
#SBATCH -p devel              # Desired partition

# %J maps to jobId.stepId

# args = grid size, performance log bool, diagnostics log bool
# grid size must be multiple of 5 and greater than 5
./build/main.exe 100 1 0

echo "OpenMP Thermosolver job complete."
