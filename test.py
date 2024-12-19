# MPI imports
from mpi4py import MPI

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
    print(f'Hi I am process {rank}, I''am sendnig `Hello` to other process')
    text = 'Hello'
    text = comm.bcast(text, root=0)
else:
    # text = None
    text = comm.bcast(None, root=0)
    print(f'Hi I am {rank} i received your {text} message.')