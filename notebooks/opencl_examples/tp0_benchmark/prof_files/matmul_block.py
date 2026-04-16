16#
# Matrix Multiplication Element By Element
# compute C=A*B
# C[i][j] for i: 0--> N-1, j! 0--> N-1
# The Program asks for 2 inputs:
# 1) Kernel version to execute: 0 for i(row)--> dim (0) , 1 for i(row) --> dim(1)
# 2)localsize (4,8,16 or 32) --> Block Size = localsize*localsize
# 

from helper import *
from definitions import *

import pyopencl as cl
import numpy
from time import time
from time import sleep

# A[N][N], B[N][N], C[N][N]
N = 2048;

# Number of elements in the matrix
size = N * N
blocksize=16
localsize=1

# A matrix
h_A = numpy.empty(size).astype(numpy.float32)
h_A.fill(AVAL)

# B matrix
h_B = numpy.empty(size).astype(numpy.float32)
h_B.fill(BVAL)

# C matrix
h_C = numpy.empty(size).astype(numpy.float32)

#--------------------------------------------------------------------------------
# CHOOSE KERNEL TO EXECUTE (0: i=dim(0),j=dim(1) ; 1:i=dim(1), j=dim(0)
#--------------------------------------------------------------------------
kernel_name="C_block_form.cl"

#--------------------------------------------------------------------------------
# CHOOSE localsize : 2, 4, 8 , 16 or 32
#--------------------------------------------------------------------------------
kernel_size = input("Please enter a value for blocksize/localsize. Possible values: 4, 8, 16 and 32 :\n")

if (kernel_size in ['4','8','16','32'] ):
    locblocksize=int(kernel_size)
    print ("Blocks Size is",locblocksize,"*",locblocksize,"\n")
else:
    print ("=== No valid input. Default Size 16 will be used. Block Size = 256")
    locblocksize=16


# Set up OpenCL
context = cl.create_some_context()
queue = cl.CommandQueue(context)

# Reset host buffers - just to play it safe
h_A = numpy.empty(size).astype(numpy.float32)
h_A.fill(AVAL)
h_B = numpy.empty(size).astype(numpy.float32)
h_B.fill(BVAL)
h_C = numpy.empty(size).astype(numpy.float32)

#h_Awrk = numpy.empty(blocksize).astype(numpy.float32)
#h_Bwrk = numpy.empty(blocksize).astype(numpy.float32)

# Create OpenCL buffers
d_a = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_A)
d_b = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_B)
d_c = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, size=h_C.nbytes)


#--------------------------------------------------------------------------------
# OpenCL matrix multiplication ... Naive: Each WI computes one element
# C_elemnt.cl : i= get_global_id(0) - j=get_global_id(1)
#--------------------------------------------------------------------------------
kernelsource = open(kernel_name).read()
program = cl.Program(context, kernelsource).build()
mmul = program.mmul
mmul.set_scalar_arg_dtypes([numpy.int32, None, None, None, None, None])

# Do the multiplication COUNT times
#locblocksize = 16

print("Starting", COUNT , " OpenCL Matrix Multiplications")
start_time = time()

for i in range(COUNT):    
    h_C.fill(0.0)
    try:
        A_block = cl.LocalMemory(numpy.dtype(numpy.float32).itemsize * locblocksize * locblocksize)
        B_block = cl.LocalMemory(numpy.dtype(numpy.float32).itemsize * locblocksize * locblocksize)   

        # Work-group computes a block of C. This size is also set
        # in a #define inside the kernel function. Note this blocksize
        # must evenly divide the matrix order

        mmul(queue, (N,N), (locblocksize,locblocksize), N, d_a, d_b, d_c, A_block, B_block)
        
        #mmul(queue, (N,N), (localsize,localsize), numpy.int32 (N), d_a, d_b, d_c,d_Awrk, d_Bwrk)
        queue.finish()
    except:
        print (" ===  Error for localsize =", localsize, "===\n")    

run_time = time() - start_time
    
print ("mmum queued")

#reading the result h_C
cl.enqueue_copy(queue, h_C, d_c)

#cl.enqueue_read_buffer(queue, d_c, h_C).wait()
print (h_C[0])


results (N, COUNT, run_time)