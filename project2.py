"""
Code for Scientific Computation Project 2
Please add college id here
CID: 01872353
"""
import heapq
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import networkx as nx

#===== Codes for Part 1=====#
def part1q1(Hlist, Hdict={}, option=0, x=[]):
    """
    Code for part 1, question 1
    Hlist should be a list of 2-element lists.
    The first element of each of these 2-element lists
    should be an integer. The second elements should be distinct and >-10000 prior to using
    option=0.
    Sample input for option=0: Hlist = [[8,0],[2,1],[4,2],[3,3],[6,4]]
    x: a 2-element list whose 1st element is an integer and x[1]>-10000
    """
    if option == 0:
        print("=== Option 0 ===")
        print("Original Hlist=", Hlist)
        heapq.heapify(Hlist)    #create a binary heap using Hlist and store as Hlist
        print("Final Hlist=", Hlist)    #print the binary heap: in decresing order of Hlist[i][0]
        Hdict = {}
        for l in Hlist:
            Hdict[l[1]] = l    #use the smaller element as keys
        print("Final Hdict=", Hdict)
        return Hlist, Hdict
    elif option == 1:
        while len(Hlist)>0:
            wpop, npop = heapq.heappop(Hlist)
            if npop != -10000:  #remove the smallest list
                del Hdict[npop]
                return Hlist, Hdict, wpop, npop
    elif option == 2:
        if x[1] in Hdict:
            l = Hdict.pop(x[1])
            l[1] = -10000
            Hdict[x[1]] = x
            heapq.heappush(Hlist, x)
            return Hlist, Hdict
        else:
            heapq.heappush(Hlist, x)
            Hdict[x[1]] = x
            return Hlist, Hdict


def part1q2(G,s,x):
    """Input:
    G: weighted NetworkX graph with n nodes (numbered as 0,1,...,n-1)
    s: an integer corresponding to a node in G
    x: an integer corresponding to a node in G
    """

    dinit = np.inf
    Fdict = {}
    Mdict = {}
    n = len(G)
    Plist = [[] for l in range(n)]

    Mdict[s]=1
    Plist[s] = [s]

    while len(Mdict)>0:
        dmin = dinit
        for n,delta in Mdict.items():
            if delta<dmin:
                dmin=delta
                nmin=n
        if nmin == x:
            return dmin, Plist[nmin]
        Fdict[nmin] = Mdict.pop(nmin)
        for m,en,wn in G.edges(nmin,data='weight'):
            if en in Fdict:
                pass
            elif en in Mdict:
                dcomp = dmin*wn
                if dcomp<Mdict[en]:
                    Mdict[en]=dcomp
                    Plist[en] = Plist[nmin].copy()
                    Plist[en].append(en)
            else:
                dcomp = dmin*wn
                Mdict[en] = dcomp
                Plist[en].extend(Plist[nmin])
                Plist[en].append(en)
    return Fdict


def part1q3(G,s,x):
    """Input:
    G: weighted NetworkX graph with n nodes (numbered as 0,1,...,n-1)
    s: an integer corresponding to a node in G
    x: an integer corresponding to a node in G
    Output: Should produce equivalent output to part1q2 given same input
    """

    #Add code here
    # check if the weights are valid
    def Wpositive(G):
        for a, b, w in G.edges(data='weight'):
            if w <= 0:
                return False
        return True
    
    #modify part1q1 so it would not print
    def part1q1n(Hlist, Hdict={}, option=0, x=[]):
        if option == 0:
            heapq.heapify(Hlist)
            Hdict = {}
            for l in Hlist:
                Hdict[l[1]] = l
            return Hlist, Hdict
        elif option == 1:
            while len(Hlist)>0:
                wpop, npop = heapq.heappop(Hlist)
                if npop != -10000:
                    del Hdict[npop]
                    return Hlist, Hdict, wpop, npop
        elif option == 2:
            if x[1] in Hdict:
                l = Hdict.pop(x[1])
                l[1] = -10000
                Hdict[x[1]] = x
                heapq.heappush(Hlist, x)
                return Hlist, Hdict
            else:
                heapq.heappush(Hlist, x)
                Hdict[x[1]] = x
                return Hlist, Hdict
        
    #initialise
    Fdict = {}
    Mlist,Mdict = part1q1n([(1, s)]) 
    Pdict = {s:[s]} #use dictionary instead of list

    #search the graph
    if Wpositive(G):  
        while len(Mlist)>0:
            try: 
                Mlist, Mdict, dmin, nmin = part1q1n(Mlist, Mdict, option = 1)   #heappop
            except TypeError:
                return Fdict
            if nmin == x:    #reach the node
                return dmin, Pdict[nmin]
            Fdict[nmin] = dmin    
            for m,en,wn in G.edges(nmin,data='weight'):   #for each neighbour
                if en in Fdict:  #the path has been already found
                    pass
                elif en in Mdict:   #explored but not finalised
                    dcomp = dmin*wn
                    if dcomp<Mdict[en][0]:
                        Mlist,Mdict = part1q1n(Mlist,Mdict,option = 2,x = [dcomp,en])
                        Pdict[en] = Pdict[nmin] + [en] #avoid empty lists and copying 
                else:    #unexplored
                    dcomp = dmin*wn
                    Mlist,Mdict = part1q1n(Mlist,Mdict,option = 2,x = [dcomp,en])
                    Pdict[en] = Pdict[nmin] + [en]   #avoid empty lists and copying 
        return Fdict  
    else:
        raise ValueError("Weights must be positive!")
    

#===== Code for Part 2=====#
def part2q1(n=50,tf=100,Nt=4000,seed=1):
    """
    Part 2, question 1
    Simulate n-individual opinion model

    Input:

    tf,Nt: Solutions are computed at Nt time steps from t=0 to t=tf (see code below)
    seed: ensures same intial condition is generated with each simulation
    Output:
    tarray: size Nt+1 array
    xarray: n x Nt+1 array containing x for the n individuals at
            each time step including the initial condition.
    """
    tarray = np.linspace(0,tf,Nt+1)
    xarray = np.zeros((Nt+1,n))

    def RHS(t,y):
        """
        Compute RHS of model
        t is a scalar, y is n*tf array 
        """
        #add code here
        m = len(y)
        dxdt = np.zeros(m)
        for i in range(m):   
            subtract = y[i]-y   #use broadcast
            f = -subtract*np.exp(-subtract**2)
            dxdt[i] = np.mean(f)
            
        return dxdt #modify return statement

    #Initial condition
    np.random.seed(seed)
    x0 = n*(np.random.rand(n)-0.5)

    #Compute solution
    out = solve_ivp(RHS,[0,tf],x0,t_eval=tarray,rtol=1e-8)
    xarray = out.y

    return tarray,xarray


def part2q2(n=50, methods = ['hybr','krylov','broyden1','broyden2','anderson','linearmixing','excitingmixing','df-sane']): #add input variables if needed
    """
    Add code used for part 2 question 2.
    Code to save your equilibirium solution is included below
    """
    xeq = np.zeros(n) #modify/discard as needed
    #Add code here
    Sdict = {} # a dictionary for storing the solutions
    Alist = [] #list of accuracies
    
    #define a function for RHS
    def fun(x,m=n):
        '''define the RHS 
           input x is a n*1 array
        '''
        function = np.zeros(m)
        for i in range(m):
            subtract = x[i]-x
            function[i] = np.mean(-subtract*np.exp(-subtract**2))
            
        return function

    #testfy if the solution meets the requierments
    def Test(a):
        b = np.abs(a)
        if np.sum(b <= 1000) == n:
            if len(np.unique(b)) > n/2:
                return True
        else:
            return False
        
    import scipy
    #set up a initial guess
    x0 = np.linspace(100,140,50)
    np.random.shuffle(x0)
    #use different methods to find the roots
    for i in methods:
        sol = scipy.optimize.root(fun, x0, args=(), method=i)['x']
        if Test(sol) == True:
            Sdict[i] = sol
            accur = np.linalg.norm(fun(sol))
            Alist.append([accur,i])
    
    if len(Sdict) == 0:
        print("No method can provide a feasible solution.")
        
    #test accuracy
    heapq.heapify(Alist)
    bacc, meth = heapq.heappop(Alist)
    xeq = Sdict[meth]    #select the one most close to 0
    
    np.savetxt('xeq.txt',xeq) #saves xeq in file xeq.txt
    
    return xeq, bacc, meth #return xeq


def part2q3(n=50,epsilon=0.1,iterations = 8): #add input variables if needed
    """
    Add code used for part 2 question 3.
    Code to load your equilibirium solution is included below
    """
    #load saved equilibrium solution
    xeq = np.loadtxt('xeq.txt') #modify/discard as needed

    #Add code here
    def part2q2_perturbed(x0,n=50, methods = ['krylov','broyden1','anderson','linearmixing','excitingmixing']):
        """part2q2 with some modifications
        """
        Sdict = {} # a dictionary for storing the solutions
        Alist = [] #list of accuracies
        
        def fun(x,m=n):
            '''define the RHS 
               input x is a n*1 array
            '''
            function = np.zeros(m)
            for i in range(m):
                subtract = x[i]-x
                function[i] = np.mean(-subtract*np.exp(-subtract**2))

            return function

        #testfy if the solution meets the requierments
        def Test(a):
            b = np.abs(a)
            if np.sum(b <= 1000) == n:
                if len(np.unique(b)) > n/2:
                    return True
            else:
                return False
            
        import scipy
        #use different methods to find the roots
        for i in methods:
            sol = scipy.optimize.root(fun, x0, args=(), method=i)['x']
            if Test(sol) == True:
                Sdict[i] = sol
                accur = np.sum(np.abs(fun(sol)))
                Alist.append([accur,i])

        #test accuracy
        heapq.heapify(Alist)
        bacc, meth = heapq.heappop(Alist)
        pertx = Sdict[meth]   

        return pertx

    #generate perturbations
    pertx1 = np.zeros((n,iterations))
    pertx2 = np.zeros((n,iterations))
    #small perturbations
    epsilon = 0.1
    x0 = xeq.copy()
    for j in range(iterations):
        xtil = epsilon * np.random.randn(n)    
        x0 = part2q2_perturbed(xtil+x0)
        pertx1[:,j] = x0
    #larger perturbations
    epsilon = 1
    x0 = xeq.copy()
    for j in range(iterations):
        xtil = epsilon * np.random.randn(n)    
        x0 = part2q2_perturbed(xtil+x0)
        pertx2[:,j] = x0
        
    #plots
    fig1 = plt.figure(1)
    plt.title(r'Small perturbation, $\epsilon$=0.1')
    plt.xlabel(r'$i$')
    plt.ylabel(r'$x_i$')
    plt.plot(pertx1)
    
    fig2 = plt.figure(2)
    plt.title(r'Large perturbation, $\epsilon$=1')
    plt.xlabel(r'$i$')
    plt.ylabel(r'$x_i$')
    plt.plot(pertx2)

    #Find eigenvalues for analysizing the stability
    M = np.zeros((n,n))
    for i in range(n):    #define M as discussed in report
        partx = xeq[i]-xeq
        M[i,:] = -1/n*np.exp(-partx**2)
        M[i,i] = np.sum(M[i,:])

    evals = np.linalg.eigvalsh(M)   #use eigenvalues to check stability
    print(evals < 0)
    
    return fig1,fig2 #modify as needed


def part2q4(n=50,m=100,tf=40,Nt=10000,mu=0.2,seed=1):
    """
    Simulate stochastic opinion model using E-M method
    Input:
    n: number of individuals
    m: number of simulations
    tf,Nt: Solutions are computed at Nt time steps from t=0 to t=tf
    mu: model parameter
    seed: ensures same intial condition is generated with each simulation

    Output:
    tarray: size Nt+1 array
    Xave: size n x Nt+1 array containing average over m simulations
    Xstdev: size n x Nt+1 array containing standard deviation across m simulations
    """

    #Set initial condition
    np.random.seed(seed)
    x0 = n*(np.random.rand(1,n)-0.5)
    X = np.zeros((m,n,Nt+1)) #may require substantial memory if Nt, m, and n are all very large
    X[:,:,0] = np.ones((m,1)).dot(x0)


    Dt = tf/Nt
    tarray = np.linspace(0,tf,Nt+1)
    dW= np.sqrt(Dt)*np.random.normal(size=(m,n,Nt))

    #Iterate over Nt time steps
    for j in range(Nt):
        #Add code here
        f = np.zeros((m,n))
        for i in range(n):
            Xi = X[:,i,j]
            subtract = Xi[:,np.newaxis]-X[:,:,j]  #np.newaxis allows us to use broadcast
            f[:,i] = np.mean(-subtract*np.exp(-subtract**2),axis=1)
        X[:,:,j+1] = X[:,:,j] + Dt*f + mu*dW[:,:,j]

    #compute statistics
    Xave = X.mean(axis=0)
    Xstdev = X.std(axis=0)

    return tarray,Xave,Xstdev


def part2Analyze(n=50,tf=50,Nt=10000): #add input variables as needed
    """
    Code for part 2, question 4(b)
    """
    #Add code here to generate figures included in your report
    # graph 1: compare stochastic and deterministic
    # deterministic model
    tarray0, Xave0, Xstdev0 = part2q4(tf=50,Nt=Nt,mu=0,seed=1)
    # stocahstic, mu = 0.2
    tarray1, Xave1, Xstdev1 = part2q4(tf=50,Nt=Nt,mu=0.2,seed=1)
    
    fig1 = plt.figure(1)
    plt.plot(Xave0[:,-1],color = 'g',label='Deterministic model')
    plt.plot(Xave1[:,-1],color = 'y',linestyle='--',label=r'Stochastic Model with $\mu$  = 0.2 and tf = 50')
    plt.xlabel(r'$i$')
    plt.ylabel(r'$x_i$')
    plt.legend(loc=(0.26,1.01))
    
    # graph 2: see how it behave when mu varies
    muval = [0.5,0.8,1,8]
    xvals = []         
    sdlist = []
    tlist = []
    
    fig2 = plt.figure(2,figsize=(30,30))    
    fig2.tight_layout()
    nrows, ncols = 5,2
    fig2.suptitle(r"Comparing Deterministic and Stochastic Opinion Models for Different $\mu$",fontsize=25)
    fig2.subplots_adjust(top=0.94)
    # generate several stochastic models with different mu and plot
    for i, u in enumerate(muval):
        a,xu,xsd= part2q4(tf=tf,Nt=Nt,mu=u,seed=1)
        tlist.append(a)
        xvals.append(xu)
        sdlist.append(xsd)
        
    for i,u in enumerate(muval):
        #mean
        ax = fig2.add_subplot(nrows, ncols, 2*i+1)
        ax.plot(tlist[i],xvals[i].T)
        ax.plot(tarray0,Xave0.T, linestyle='dashed',alpha=0.5)
        ax.set_title(f"Mean, $\mu$={u}", fontsize=20)
        ax.set_ylabel(r"$x_i$", fontsize=15)
        ax.set_xlabel("t", fontsize=15)
        #sd
        ax = fig2.add_subplot(nrows, ncols, 2*i+2)
        ax.plot(tlist[i],sdlist[i].T)
        ax.plot(tarray0,u*np.sqrt(tarray0), 'k--',label=r"theorectial s.d, $\mu*\sqrt{t}$",linewidth=3)
        ax.plot(tarray0,Xstdev0[0], 'g-.',label="s.d of deterministic Model")
        ax.set_title(f"Standard Deviation, $\mu$={u}", fontsize=20)
        ax.set_ylabel(r"$x_i$", fontsize=15)
        ax.set_xlabel("t", fontsize=15)
        ax.legend(fontsize=14)
        
    return fig1,fig2 #modify as needed