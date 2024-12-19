#!/bin/bash

# Check if MPI_ENABLED environment variable is set
if [[ -n "$MPI_ENABLED" && "$MPI_ENABLED" == "true" ]]; then
    # Launch with mpiexec
    mpiexec -n 4 python -m ipykernel --ip=$1
else
    # Launch without mpiexec
    python -m ipykernel --ip=$1
fi