import numpy as np

def inv(x):
    return np.linalg.inv(x)

def dot(x):
    if len(x) == 1:
        return x[0]
    elif len(x) == 2:
        return np.dot(x[0],x[1])
    else:
        return dot([x[0],dot(x[1:])])

def logterm(m,C,logdet = None):
    if logdet is None:
        _,logdet = np.linalg.slogdet(C)
        
    return -(len(m) * np.log(2*np.pi) + logdet + dot([m,inv(C),m]))/2

def att_PIA_iter(method,I,a,cov,ncov,qcov,F):

    C = a*a*cov
    Q = a*a*qcov

    n = I.shape[-1]

    l2p = n*np.log(2*np.pi)
    _,ldcov = np.linalg.slogdet(C + ncov)
    
    if len(I) == 1:
        return logterm(I[0],C + ncov)

    fi = inv(F)
    _,ldfi = np.linalg.slogdet(fi)

    out = ldfi

    mu = dot([fi,I[-1]])
    xi = dot([fi,Q + ncov,fi.transpose()])

    if method == 1:
        iterator = iterate_1
    elif method == 2:
        iterator = iterate_2
    elif method == 3:
        iterator = iterate_3
    else:
        print("Method not recognized")
        exit()
        
    for n in reversed(range(len(I[:-1]))):
        mu,xi,contrib = iterator(mu,xi,I[n],C,Q,ncov,F,n)
        out += contrib

    return out

def iterate_1(mu,xi,I,C,Q,ncov,F,n):

    if n == 0:
        F = 0*F
        Q = C
    
    X = inv(inv(Q) + inv(ncov))

    if n == 0:        
        eta = dot([X,inv(ncov),I])
        return 0,0,logterm(I,ncov + Q) + logterm(mu - eta,xi + X)

    O = dot([X,inv(Q),F])
    
    A = dot([inv(F),ncov + Q,inv(F).transpose()])
    B = dot([inv(O),xi + X,inv(O).transpose()])

    m = dot([inv(F),I]) - dot([inv(O),mu - dot([X,inv(ncov),I])])
    cov = A + B

    distcon = logterm(m,cov)

    _,fcon = np.linalg.slogdet(F)
    _,ocon = np.linalg.slogdet(O)

    contrib = - fcon - ocon + distcon

    xio = inv(inv(A) + inv(B))
    mtemp = dot([inv(A),inv(F),I]) + dot([inv(B),inv(O),(mu - dot([X,inv(ncov),I]))])
    
    muo = dot([xio,mtemp])

    return muo,xio,contrib

def iterate_2(mu,xi,I,C,Q,ncov,F,n):

    if n == 0:
        F = 0*F
        Q = C

    X = inv(inv(xi) + inv(ncov))
    eta = dot([X,(dot([inv(xi),mu]) + dot([inv(ncov),I]))])

    if n == 0:
        return 0,0,logterm(mu - I, xi + ncov) + logterm(eta,X + Q)

    _,fcon = np.linalg.slogdet(F)
    
    xio = dot([inv(F),X + Q, inv(F).transpose()])
    muo = dot([inv(F),eta])

    contrib = -fcon + logterm(mu - I,xi + ncov)
                          
    return muo,xio,contrib
              
              
def iterate_3(mu,xi,I,C,Q,ncov,F,n):

    '''
    Double checked this all.
    '''

    if n == 0:
        F = 0*F
        Q = C

    X = inv(inv(xi) + inv(Q))

    if n == 0:
        eta = dot([X,dot([inv(xi),mu])])
        return 0,0,logterm(mu,Q + xi) + logterm(eta - I,X + ncov)

    O = dot([X,inv(Q),F])
    nu = (I - dot([X,inv(xi),mu]))

    A = dot([inv(F),(Q + xi),inv(F).transpose()])
    B = dot([inv(O),(X + ncov),inv(O).transpose()])

    _,fcon = np.linalg.slogdet(F)
    _,ocon = np.linalg.slogdet(O)

    contrib = - fcon - ocon + logterm(dot([inv(F),mu]) - dot([inv(O),nu]),A + B)

    xio = inv(inv(A) + inv(B))
    mutemp = dot([inv(B),inv(O),nu]) + dot([inv(A),inv(F),mu])
    muo = dot([xio,mutemp])

    return muo,xio,contrib

def big_PIA_comp(I,a,cov,ncov,F,G):
    def C(n,m):
        g = a*a*cov
        N = ncov
        
        A = dot([np.linalg.matrix_power(F,abs(m-n)),g])#,a*a*cov)
        B = dot([np.linalg.matrix_power(G,abs(m-n)),N])#,ncov)

        if m <= n:
            return A+B
        else:
            return (A+B).transpose()

    CI = np.array([[C(i,j) for j in range(len(I))] for i in range(len(I))]).transpose([0,2,1,3])

    size = len(I) * len(cov)
    
    CI = np.reshape(CI,[size,size])

    i = np.reshape(I,[-1])

    print("CISUM: {}".format(np.sum(CI - CI.transpose())))
    
    return logterm(i,CI)
    
if __name__ == "__main__":
    from GSM.ATT_GSM_inference import Q_self_con
    np.random.seed(0)
    
    def mat_sq(m):
        return np.dot(m,m.transpose())
    
    I = np.random.randn(20,10)#np.ones([100,10])
    
    a = 2.
    cov = mat_sq(np.random.randn(10,10))
    ncov = mat_sq(np.random.randn(10,10))

    F = .1*mat_sq(np.random.randn(10,10))
    
    qcov = Q_self_con(cov,F)#mat_sq(np.random.randn(10,10))
    print("Testing PIA")

    for m in [2]:
        p1 = att_PIA_iter(m,I,a,cov,ncov,qcov,F)
        print(p1)

    print(big_PIA_comp(I,a,cov,ncov,F,0*F))
