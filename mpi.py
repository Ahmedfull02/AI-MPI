#!/usr/bin/env python
# coding: utf-8

# In[8]:


from mpi4py import MPI

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()  # Process rank
size = comm.Get_size()  # Total number of processes

# Print "Hello, World!" from each process
print(f"Hello, World from rank {rank} out of {size}")

