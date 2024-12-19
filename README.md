# project to implement a predict shipping cost using three regression machine learning algorithms with applied MPI.

### What is MPI?
MPI `the Message Passing Interface`, is a standardized and portable message-passing system designed to function on a wide variety of parallel computers. The standard defines the syntax and semantics of library routines and allows users to write portable programs in the main scientific programming languages (Fortran, C, or C++).

### What is Mpi4Py?
MPI for Python provides MPI bindings for the Python programming language, allowing any Python program to exploit multiple processors. 

### What Models are used here?
The models used here are linear regression KNN and Gradient boosting  


> [!NOTE]
> to run this code you need to install scikit-learn, pandas, numpy, matplotlib, mpi4py
> the command is:
> ```
> mpiexec -n 3 python3 models.py
> ```
> if it did not work use this to run it:
> ```
> mpiexec --oversubscribe -n 3 python3 models.py
> ```

> [!WARNING]
> If there is error in reading dataset change codecs until it works:
> data = pd.read_csv('/home/adnane/mpi/a-project-of-supervised-machine-learning-using-gradient-boosting/data.csv', encoding='latin-1')
### Example of execution:
#### With use of comm.Barrier:
![image](https://github.com/user-attachments/assets/0fe22fcb-2ee0-49b9-9ed6-5cf3916844c1)
#### Without use of comm.Barrier:
![image](https://github.com/user-attachments/assets/ccacb30a-6054-4b9c-a1ed-07bf7007012a)

