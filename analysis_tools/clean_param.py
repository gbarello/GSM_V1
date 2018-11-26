import numpy as np
import GSM.MGSM_inference as MGSM
import scipy

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 25
plt.rcParams['lines.linewidth'] = 4.

cm = plt.get_cmap('gist_rainbow')

def get(f):
    F = open(f,"r")

    out = []
    for l in F:
        temp = l.split(",")
        temp[-1] = temp[-1][:-1]

        out.append([float(x) for x in temp])

    return np.array(out)

def get_trans(curv,con):
    i = np.argmax(curv)

    while curv[i] > 1 and i < len(curv)-1:
        i += 1

    slope = (curv[i] - curv[i-1])/(con[i] - con[i-1])
        
    con_t = con[i-1] + (1. - curv[i-1])/slope

    return con_t

def cov(n,w,dr = 0):
    
    distance = np.array([[((i - j + (n/2.))%n) - n/2. for i in range(n)] for j in range(n)])
    
    gauss = np.exp((np.cos(np.pi*distance/(n-1)) - 1)/(w*w)) - dr
    
    #    out = np.dot(gauss,gauss)
    out = gauss
    
    return out/out[0,0]

def inp(n,w,c1,c2,i):
    stim1 = np.array([((j + (n/2.))%n) - (n/2.) for j in range(n)])/(n-1)
    stim2 = np.array([(((j - (i%n)) + (n/2.))%n) - (n/2.) for j in range(n)])/(n-1)
    
    gauss1 = np.exp((np.cos(2*np.pi*stim1) - 1)/(w*w))
    gauss2 = np.exp((np.cos(2*np.pi*stim2) - 1)/(w*w))
    
    return c1*gauss1 + c2*gauss2

def run_k():

    #what it is that we want to do here? We want to look at COS

    resp1 = []
    resp2 = []
    resp3 = []
    
    nresp1 = []
    nresp2 = []
    nresp3 = []

    NUM_CURVE = 10
    NUM_CON = 50
    
    CMAX = 1.
    CMIN = -2.
    N = 8

    I1 = inp(N,.6,1,1,-1 + N/2)
    I2 = inp(N,.6,1,0,-1 + N/2)
    I3 = inp(N,.6,0,1,-1 + N/2)

    KK = np.logspace(0,1,NUM_CURVE)

    con = np.logspace(CMIN,CMAX,NUM_CON)

    if 0:
        for k in KK:
            print(k)
            W = .6
            CC = cov(N,W)
            NC = CC
            
            np.savetxt("./covariance_test.csv",CC)
            
            stim1 = np.array([cc * I1 for cc in np.logspace(CMIN,CMAX,NUM_CON)])
            stim2 = np.array([cc * I2 for cc in np.logspace(CMIN,CMAX,NUM_CON)])
            stim3 = np.array([cc * I3 for cc in np.logspace(CMIN,CMAX,NUM_CON)])
            
            resp1.append(MGSM.gexp(0,stim1,CC,NC/(k*k),precom = False))
            resp2.append(MGSM.gexp(0,stim2,CC,NC/(k*k),precom = False))
            resp3.append(MGSM.gexp(0,stim3,CC,NC/(k*k),precom = False))
            
            nresp1.append(MGSM.gnn(stim1,CC)[:,0])
            nresp2.append(MGSM.gnn(stim2,CC)[:,0])
            nresp3.append(MGSM.gnn(stim3,CC)[:,0])

        np.savetxt("./param_files/kresp1.csv",resp1)
        np.savetxt("./param_files/kresp2.csv",resp2)
        np.savetxt("./param_files/kresp3.csv",resp3)
        
        np.savetxt("./param_files/knnresp1.csv",nresp1)
        np.savetxt("./param_files/knnresp2.csv",nresp2)
        np.savetxt("./param_files/knnresp3.csv",nresp3)
    else:
        resp1 = np.loadtxt("./param_files/kresp1.csv")
        resp2 = np.loadtxt("./param_files/kresp2.csv")
        resp3 = np.loadtxt("./param_files/kresp3.csv")
        
        nresp1 = np.loadtxt("./param_files/knnresp1.csv")
        nresp2 = np.loadtxt("./param_files/knnresp2.csv")
        nresp3 = np.loadtxt("./param_files/knnresp3.csv")

    resp1 = np.array(resp1)
    resp2 = np.array(resp2)
    resp3 = np.array(resp3)

    nresp1 = np.array(nresp1)
    nresp2 = np.array(nresp2)
    nresp3 = np.array(nresp3)

    AI = resp1/(resp2+resp3)
    nAI = nresp1/(nresp2+nresp3)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    K = np.linspace(.5,5,NUM_CURVE)
    cc = np.logspace(CMIN,CMAX,NUM_CON)/(10**CMAX)
    
    for k in range(0,len(AI),2):
        ax.plot(con/np.max(con),AI[k],label=str(K[k]))
        ax.plot(con/np.max(con),nAI[k],'--')
        
    ax.plot(cc,[1 for k in range(len(cc))],'k--')
    plt.tight_layout()
    plt.xlabel("contrast")
    plt.ylabel("AI")

    plt.ylim(0,1.5)


    fig.savefig("./paramfig_K.pdf")

    MAX = np.array([[1./KK[k],np.max(AI[k])] for k in range(len(AI))])
    plt.figure()
    plt.plot(MAX[:,0],MAX[:,1],"r")
    plt.xscale('log')
    plt.tight_layout()
    plt.xlabel("1/SNR")
    plt.ylabel("Max AI")

    plt.savefig("./max_AI_by_k.pdf")

    plt.figure()
    plt.plot(con/np.max(con),resp2[NUM_CURVE/2,:],"r")
    plt.plot(con/np.max(con),nresp2[NUM_CURVE/2,:],"r--")
    plt.xscale('log')
    plt.tight_layout()

    plt.xlabel("Contrast")
    plt.ylabel("Response")

    plt.savefig("./param_CRF.pdf")
    
    pts = []
    for k in range(len(AI)):
        pts.append([KK[k],get_trans(AI[k],con)])

    pts = np.array(pts)

    plt.figure()

    plt.plot(1./pts[:,0],pts[:,1])
    plt.xscale('log')
#    plt.yscale('log')
    
    plt.tight_layout()
 
    plt.xlabel("1/SNR")
    plt.ylabel("Transition (con.)")

    plt.savefig("./trans_by_K.pdf")

def run_w():

    #what it is that we want to do here? We want to look at COS

    resp1 = []
    resp2 = []
    resp3 = []
    
    nresp1 = []
    nresp2 = []
    nresp3 = []

    NUM_CURVE = 20
    NUM_CON = 50
    
    CMAX = 1
    CMIN = -2
    N = 8

    I1 = inp(N,.6,1,1,-1 + N/2)
    I2 = inp(N,.6,1,0,-1 + N/2)
    I3 = inp(N,.6,0,1,-1 + N/2)

    KK = np.linspace(.27,.75,NUM_CURVE)
    
    con = np.logspace(CMIN,CMAX,NUM_CON)

    if 0:
        for k in KK:
            print(k)
            W = k
            CC = cov(N,W)
            NC = cov(N,W)
            
            np.savetxt("./covariance_test.csv",CC)
            
            stim1 = np.array([cc * I1 for cc in np.logspace(CMIN,CMAX,NUM_CON)])
            stim2 = np.array([cc * I2 for cc in np.logspace(CMIN,CMAX,NUM_CON)])
            stim3 = np.array([cc * I3 for cc in np.logspace(CMIN,CMAX,NUM_CON)])
            
            resp1.append(MGSM.gexp(0,stim1,CC,NC/(2*2),precom = False))
            resp2.append(MGSM.gexp(0,stim2,CC,NC/(2*2),precom = False))
            resp3.append(MGSM.gexp(0,stim3,CC,NC/(2*2),precom = False))
            
            nresp1.append(MGSM.gnn(stim1,CC)[:,0])
            nresp2.append(MGSM.gnn(stim2,CC)[:,0])
            nresp3.append(MGSM.gnn(stim3,CC)[:,0])

        np.savetxt("./param_files/wresp1.csv",resp1)
        np.savetxt("./param_files/wresp2.csv",resp2)
        np.savetxt("./param_files/wresp3.csv",resp3)
        
        np.savetxt("./param_files/wnnresp1.csv",nresp1)
        np.savetxt("./param_files/wnnresp2.csv",nresp2)
        np.savetxt("./param_files/wnnresp3.csv",nresp3)
    else:
        resp1 = np.loadtxt("./param_files/wresp1.csv")
        resp2 = np.loadtxt("./param_files/wresp2.csv")
        resp3 = np.loadtxt("./param_files/wresp3.csv")
        
        nresp1 = np.loadtxt("./param_files/wnnresp1.csv")
        nresp2 = np.loadtxt("./param_files/wnnresp2.csv")
        nresp3 = np.loadtxt("./param_files/wnnresp3.csv")

    resp1 = np.array(resp1)
    resp2 = np.array(resp2)
    resp3 = np.array(resp3)

    nresp1 = np.array(nresp1)
    nresp2 = np.array(nresp2)
    nresp3 = np.array(nresp3)

    AI = resp1/(resp2+resp3)
    nAI = nresp1/(nresp2+nresp3)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)

    cc = con/np.max(con)
    
    for k in range(0,len(AI),2):
        ax.plot(cc,AI[k],label=str(KK[k]))
        ax.plot(cc,nAI[k],'--')
        
    ax.plot(con/np.max(con),[1 for k in range(len(cc))],'k--')

    plt.xlabel("contrast")
    plt.ylabel("A.I.")

    plt.ylim(.5,1.25)

    plt.tight_layout()

    fig.savefig("./paramfig_W.pdf")

    MAX = np.array([[KK[k]*180/np.pi,np.max(AI[k])] for k in range(len(AI))])
    plt.figure()
    plt.plot(MAX[:,0],MAX[:,1],"r")
    plt.xscale('log')
    plt.tight_layout()

    plt.xlabel("w")
    plt.ylabel("Max AI")

    plt.savefig("./max_AI_by_W.pdf")
    
    pts = []
    for k in range(len(AI)):
        pts.append([KK[k],get_trans(AI[k],con)])

    pts = np.array(pts)

    plt.figure()

    plt.plot(pts[:,0],pts[:,1])
    plt.xscale('log')
    plt.yscale('log')
    plt.tight_layout()

    plt.xlabel("w")
    plt.ylabel("Transition (con)")

    plt.savefig("./trans_by_W.pdf")

    plt.figure()

    plt.plot(KK,AI[:,-6])
    plt.tight_layout()

    plt.title("con. = {}".format(KK[-6]))

    plt.xlabel("w")
    plt.ylabel("High Contrast AI")

    plt.savefig("./HighAI_by_W.pdf")

if __name__ == "__main__":
    run_k()
    run_w()
