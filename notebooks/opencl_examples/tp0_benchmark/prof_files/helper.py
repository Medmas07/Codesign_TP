
from definitions import *

#  Function to compute the matrix product (sequential algorithm, dot prod)
def seq_mat_mul_sdot(N, A, B, C):
    for i in range(N):
        for j in range(N):
            tmp = 0.0
            for k in range(N):
                tmp += A[i*N+k] * B[k*N+j]
            C[i*N+j] = tmp

#  Function to compute errors of the product matrix
def error(N, C):
    cval = float(N) * AVAL * BVAL
    print ("correct value :", cval)
    errsq = 0.0
    for i in range(N):
        for j in range(N):
            matelem=C[i*N+j]
            err = matelem - cval
            #print("one eror:",err)
            if (err>5): 
                print ("too much", matelem, "/n" )
            errsq += abs(err) #* err
            #input("Press Enter...")
            #print("eror:",errsq)
    return errsq


# Function to analyze and output results
def results(N, COUNT, run_time):
    
    mflops = 2.0 * COUNT * N * N * N/(1000000.0* run_time)
    
    print (run_time, "seconds at", mflops, "MFLOPS")
    #errgpu = error(N, C)
    #toterr=errgpu/(N*N)
    #print ("Error per operation :",toterr)
    #if (errgpu > TOL):
    #    print ()#print ("Errors in multiplication:", errsq)