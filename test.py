import ipyparallel as ipp
def mpi_example():
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    return f"Hello World from rank {comm.Get_rank()}. total ranks={comm.Get_size()}. host={MPI.Get_processor_name()}"

# request an MPI cluster with 24 engines
with ipp.Cluster(controller_ip="*", engines="mpi", n=24) as rc:
    # get a broadcast_view on the cluster which is best
    # suited for MPI style computation
    view = rc.broadcast_view()
    # run the mpi_example function on all engines in parallel
    r = view.apply_sync(mpi_example)
    # Retrieve and print the result from the engines
    print("\n".join(r))
# at this point, the cluster processes have been shutdown