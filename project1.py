"""
Code for Scientific Computation Project 1
Please add college id here
CID:01872353
"""


#===== Code for Part 1=====#
from part1_utilities import * #do not modify

def method1(L_all,x):
    """
    First method for finding location of target in list containing
    M length-N sorted lists
    Input: L_all: A list of M length-N lists. Each element of
    L_all is a list of integers sorted in non-decreasing order.
    Example input for M=3, N=2: L_all = [[1,3],[2,4],[6,7]]

    """
    M = len(L_all)
    for i in range(M):
        ind = bsearch1(L_all[i],x)
        if ind != -1000:
            return((i,ind))

    return (-1000,-1000)




def method2(L_all,x,L_new = []):
    """Second method for finding location of target in list containing
    M length-N sorted lists
    Input: L_all: A list of M length-N lists. Each element of
    L_all is a list of integers sorted in non-decreasing order.
    Example input for M=3, N=2: L_all = [[1,3],[2,4],[6,7]]
    """

    if len(L_new)==0:
        M = len(L_all)
        N = len(L_all[0])
        L_temp = []
        for i in range(M):
            L_temp.append([])
            for j in range(N):
                L_temp[i].append((L_all[i][j],(i,j)))

        def func1(L_temp):
            M = len(L_temp)
            if M==1:
                return L_temp[0]
            elif M==2:
                return merge(L_temp[0],L_temp[1])
            else:
                return merge(func1(L_temp[:M//2]),func1(L_temp[M//2:]))

        L_new = func1(L_temp)

    ind = bsearch2(L_new,x)
    if ind==-1000:
        return (-1000,-1000),L_new
    else:
        return L_new[ind][1],L_new


def time_test(inputs=None):
    """Examine dependence of walltimes on M, N, and P for method1 and method2
        You may modify the input/output as needed.
    """

    #Add code here for part 1, question 2
    from time import time
    import numpy as np
    import matplotlib.pyplot as plt

    #set up P,M,N
    M = np.array([2**10,2**6,16])
    N = np.array([16,2**6,2**10])
    P = np.linspace(50,500,9,dtype=int)
    
    m = len(M)
    p = len(P)
    
    #empty matrix for containing time
    vals1 = np.zeros((m,p))
    vals2 = np.zeros((m,p))
    #a random value to search(for the first time of method 2)
    x = 0 
    
    for i in range(m):
        L = make_L_all(M[i],N[i]) #generate a random list L_all
        for j in range(p):
            #wall time of method 1
            t1 = time()
            for k in range(P[j]):
                a = method1(L,k)
            t2 = time()
            vals1[i,j] = t2-t1
            #wall time of method 2
            T1 = time()
            b = method2(L,x) #first search by method2
            for k in range(1,P[j]):
                c = method2(L,k,b[1]) #use L_new
            T2 = time()
            vals2[i,j] = T2-T1


    #calculate average running time
    AT1 = vals1/P
    AT2 = vals2/P
    
    #figure1: Average Running time of each cases
    fig1=plt.figure()
    plt.loglog(P,AT1[3],label='method1, M, N large')
    plt.loglog(P,AT2[3],label='method2, M, N large')
    plt.loglog(P,AT1[0],linestyle=':',label='method1, M $\gg$ N')
    plt.loglog(P,AT2[0],linestyle=':',label='method2, M $\gg$ N')
    plt.loglog(P,AT1[2],linestyle='-.',label='method1, N $\gg$ M')
    plt.loglog(P,AT2[2],linestyle='-.',label='method2, N $\gg$ M')
    plt.grid()
    plt.legend(loc=(1.01,0.51))
    plt.xlabel('Number of Targets, P ')
    plt.ylabel('Average Running Time')
    

    #figure2: testing relative scale of P and M, N
    ind = 1 #M=N=2^6
    fig2=plt.figure()
    plt.loglog(P,vals1[ind], label='method1')
    plt.loglog(P,vals2[ind], label='method2')
    plt.grid()
    plt.legend()
    plt.xlabel('Number of Targets, P')
    plt.ylabel('Wall Time (s)')
    
    return fig1,fig2 #Modify if needed



#===== Code for Part 2=====#

def findGene(L_in,L_p):
    """Find locations within adjacent strings (contained in input list,L_in)
    that contain patterns in input list L_p
    Input:
    L_in: A list containing two length-n strings
    L_p: A list containing p length-m strings

    Output:
    L_out: A length-p list whose ith element is a list of locations where the
    ith pattern has been found (see project description for further details)
    """
    #Size parameters
    n = len(L_in[0]) #length of a sequence
    p = len(L_p) #number of patterns
    m = len(L_p[0]) #length of pattern

    L_out = [[] for i in range(p)]

    #Add code here for part 2, question 1
    #define a function converts the gene codes to numbers 
    def Nchar(S):
        char = {}
        char['A']=0
        char['C']=1
        char['G']=2
        char['T']=3
        L=[]
        for s in S:
            L.append(char[s])
        return L

    #Hash function
    b=4; q=997 #set default values
    def hval(L,B=b,P=q):
        f = 0
        for l in L[:-1]:
            f = B*(l+f)
        h = (f + (L[-1])) % P
        return h

    seq1 = Nchar(L_in[0])
    seq2 = Nchar(L_in[1])
    bm = (4**m) % q      #a constant for later
    
    #lists for p length-m patterns
    Hvals = []  
    Xp = []
    for i in range(p):
        N_p = Nchar(L_p[i])
        Xp.append(N_p)
        hi = hval(N_p)
        Hvals.append(hi)
    
    #compare patterns for each small pieces of S1
    #first hash
    ind = 0  
    hS = hval(seq1[:m])
    #matching steps
    for i in range(p):
        if Hvals[i]==hS:
            if seq1[:m]==Xp[i]:
                if seq2[:m]==Xp[i]:
                    L_out[i].append(ind)
                    
    #update rolling hash
    for ind in range(1,n-m+1):
        hS = (b*hS-int(seq1[ind-1])*bm + int(seq1[ind-1+m])) % q 
        #matching
        for i in range(p):
            if Hvals[i]==hS:
                if seq1[ind:ind+m]==Xp[i]:
                    if seq2[ind:ind+m]==Xp[i]:
                        L_out[i].append(ind)
    
    return L_out


if __name__=='__main__':
    #Small example for part 2
    S1 = 'ATCGTACTAGTTATC'
    S2 = 'ATCTTAGTAGTCGTC'
    L_in = [S1,S2]
    L_p = ['ATC','AGT']
    out = findGene(L_in,L_p)

    #Large gene sequences
    infile1,infile2 = open("S1example.txt"), open("S2example.txt")
    S1,S2 = infile1.read(), infile2.read()
    infile1.close()
    infile2.close()
