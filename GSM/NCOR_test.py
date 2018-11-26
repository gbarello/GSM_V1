import MGSM_inference as MGSM
import numpy as np
import time


def GSM_test():
    NN = 24
    MM = 1
    
    X = np.random.randn(MM,NN)

    cov = np.random.randn(NN,NN)/2
    cov = np.identity(NN)
    cov = (cov + cov.T)/2
    cov = np.dot(cov,cov.T)
    
    ncor = np.random.randn(NN,NN)/2
    ncor = np.identity(NN)
    ncor = (ncor + ncor.T)/2
    ncor = np.dot(ncor,ncor.T)

    print(np.linalg.det(ncor))
    print(np.linalg.det(cov))
    
    t1 = time.time()
    
    g1 = MGSM.gexp(0,X,cov,ncor,precom = False)
    
    t2 = time.time()
    
    g2 = MGSM.gexp(0,X,cov,0.0000001*ncor)
    
    t3 = time.time()
    
    g3 = MGSM.gexp(0,X,cov,ncor,precom = True,npnt = 10000,ahigh = 500)
    
    t4 = time.time()
    
    g4 = MGSM.gnn(X,cov)[:,0]
    
    t5 = time.time()
    
    print(g1[0],g3[0])
    print(g2[0],g4[0])
    print(t2 - t1,t3-t2,t4-t3,t5-t4)

def make_R_cov(n):
    cov = np.random.randn(n,n)/np.sqrt(n)
    cov = (cov + cov.T)/2

    cov = np.dot(cov,cov)
    
    return cov

def MGSM_test():

    NN = 2
    
    X = np.random.randn(NN,9,8)
    
    c1 = make_R_cov(8)
    c2 = np.array([make_R_cov(24) for i in range(5)])
    c3 = np.array([make_R_cov(16) for i in range(5)])
        
    n1 = make_R_cov(8)
    n2 = np.array([make_R_cov(24) for i in range(5)])
    n3 = np.array([make_R_cov(16) for i in range(5)])

    P = np.random.rand(5)
    P = P/np.sum(P)

    t1 = time.time()
    
    g1 = MGSM.MGSM_g(X,c1,c2,c3,n1,n2,n3,P,"coen_cagli")
    
    t2 = time.time()
    
    g2 = MGSM.MGSM_g(X,c1,c2,c3,.0001*n1,.0001*n2,.0001*n3,P,"coen_cagli")
    
    t3 = time.time()
    
    g3 = MGSM.MGSM_g(X,c1,c2,c3,n1,n2,n3,P,"coen_cagli",prec = True)
    
    t4 = time.time()
    
    g4 = MGSM.MGSM_gnn(X,c1,c2,c3,P,"coen_cagli")    
    
    t5 = time.time()
    
    print(g1[0,0],g3[0,0])
    print(g2[0,0],g4[0,0])
    print(t2 - t1,t3-t2,t4-t3,t5-t4)

if __name__ == "__main__":
    GSM_test()
#    MGSM_test()
