import numpy as np

def logint(F,a):
    
    FM = np.array([logsum(F[i:i+2]) - np.log(2) for i in range(len(F)-1)])
    AD = (a[1:] - a[:-1])

    return logsum(FM,AD)

def logsum(f,w = 1):
    m = np.max(f)

    return np.log(np.sum(w*np.exp(f - m))) + m

def logdif(a,b,ABS = True):
    m = np.max([a,b])

    if ABS:
        return np.log(np.abs((np.exp(a - m) - np.exp(b - m)))) + m
    else:
        return np.log(np.exp(a - m) - np.exp(b - m)) + m
    
def splitpass(f,F,A,ind):
    LS = logint(F,A)
    
    newA = np.array([])
    newF = np.array([])
    perr = np.array([])

    newind = []
    istep = 0
    
    itemp = 0
    for j in range(len(ind)):
        
        i = ind[j]
        lsi = logint(F[i:i+2],A[i:i+2])
        
        atemp = np.array([A[i],(A[i] + A[i+1])/2,A[i+1]])
        ftemp = np.array([F[i],f(atemp[1]),F[i+1]])
        
        lsf = logint(ftemp,atemp)

        LTtemp = logdif(logsum(np.array([LS,lsf])),lsi,ABS = False)
        
        PE = logdif(LTtemp,LS) - LS

        LS = LTtemp

        perr = np.append(perr,[PE,PE])

        newA = np.append(newA,A[itemp:i])
        newA = np.append(newA,atemp[:-1])
        
        newF = np.append(newF,F[itemp:i])
        newF = np.append(newF,ftemp[:-1])
        
        itemp = i+1

        newind += [istep + i,istep + i+1]
        istep += 1

    newA = np.append(newA,[A[itemp:]])
    newF = np.append(newF,[F[itemp:]])
#    newF = np.array([f(a) for a in newA])
    return newF,newA,perr,newind

def integrate_log(f,low,high,start = [],perror=1e-5):
    '''
    I want to start with a grid in the integration variable, and then refine it as needed. 

    I need to come up with a criterion to split an interval. 

    I need to estimate the error somehow... 

    The percent error in a region is the error estimate in that region, divided by the total integral so far.
    '''

    a = np.array([low] + start + [high])
#    a = np.array([low,(low+start)/2,start,(high+start)/2,high])    
    F = np.array([f(x) for x in a])
    
    tot = logint(F,a)

    IND = np.arange(len(a)-1)
    step = 0

    while len(IND) > 0 and step < 100:
        step +=1

        F,a,perr,IND = splitpass(f,F,a,IND)

        itemp = []

        for i in range(len(IND)):
            if perr[i] > np.log(perror):
                itemp.append(IND[i])

        IND = np.array(itemp)

#        print(IND)
#        print(a)
#        raw_input("Press Enter to continue...")        
        

    out = logint(F,a)
    return out,{"points":a,"func_vals":F,"steps":step}

if __name__ == "__main__":
    f = lambda x:-((x)**2)/2

    '''
    a = np.array([-20,0,20])
    fp = f(a)
    ONE = logint(fp,a)
    
    a = np.array([-20,-10,0,10,20])
    fp = f(a)
    TWO = logint(fp,a)

    print((np.exp(ONE) - np.exp(TWO))/np.exp(TWO))
    '''

    pnts = []
    for pe in [1e-1,1e-2,1e-3,1e-4,1e-5,1e-6,1e-10]:
        res,diag = integrate_log(f,-20,20,pnts,perror = pe)
#        pnts = diag["points"]
        print(np.exp(res))#,diag["steps"])
        print(len(diag["points"]))

    print(np.sqrt(2*np.pi))

