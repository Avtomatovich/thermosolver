#!/bin/bash

#SBATCH -J thermosolver       # Job name
#SBATCH -o output/job.%j.txt  # Name of stdout output file (%j expands to jobId)
#SBATCH -N 1                  # Total number of nodes requested
#SBATCH -n 8                  # Total number of mpi tasks requested
#SBATCH -t 00:15:00           # Run time (hh:mm:ss) - 15 minutes
#SBATCH -p devel              # Desired partition

# %J maps to jobId.stepId

# args = method (Jacobi = 0, RBGS = 1, SOR = 2), grid size, log bool
# if grid size < 5 or not multiple of 5, main uses default N range of [50, 200]
# prun rocprofv3 --kernel-trace --memory-copy-trace --output-format pftrace --marker-trace  -- ./build/main.exe 2 100 0
prun ./build/main.exe 2 100 0

echo "Thermosolver job completed."
