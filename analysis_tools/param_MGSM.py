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

def c2(c):
    return np.array([[1.,c],[c,1.]])

def cov(n,w,dr = 0):
    
    distance = np.array([[((i - j + (n/2.))%n) - n/2. for i in range(n)] for j in range(n)])
    
    gauss = np.exp((np.cos(np.pi*distance/n) - 1)/(w*w)) - dr
    
    #    out = np.dot(gauss,gauss)
    out = gauss
    
    return out/out[0,0]

def inp(n,w,c1,c2,i):
    stim1 = np.array([((j + (n/2.))%n) - (n/2.) for j in range(n)])
    stim2 = np.array([(((j - (i%n)) + (n/2.))%n) - (n/2.) for j in range(n)])
    
    gauss1 = np.exp(-(stim1**2)/(2.*w*w))
    gauss2 = np.exp(-(stim2**2)/(2.*w*w))
    
    return c1*gauss1 + c2*gauss2

def run():

    #what it is that we want to do here? We want to look at COS


    resp1 = []
    resp2 = []
    resp3 = []
    
    nresp1 = []
    nresp2 = []
    nresp3 = []

    NUM_CURVE = 20
    NUM_CON = 50
    
    CMAX = 2.5
    CMIN = -2
    N = 8

    I1 = inp(N,float(N)/5,1,1,-1 + N/2)
    I2 = inp(N,float(N)/5,1,0,-1 + N/2)
    I3 = inp(N,float(N)/5,0,1,-1 + N/2)

    KK = np.logspace(-1,1,NUM_CURVE)
    
    for k in KK:
        print(k)
        W = float(N)/(5)
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
    ax.set_prop_cycle("color",[cm(1.*i/NUM_CURVE) for i in range(NUM_CURVE)])

    K = np.linspace(.5,5,NUM_CURVE)
    cc = np.logspace(CMIN,CMAX,NUM_CON)/(10**CMAX)
    
    for k in range(0,len(AI),1):
        ax.plot(cc,AI[k],label=str(K[k]))
        ax.plot(cc,nAI[k],'--')
        
    handles, labels = ax.get_legend_handles_labels()
#    ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
#               ncol=2, mode="expand", borderaxespad=0.)
    ax.plot(cc,[1 for k in range(len(cc))],'k--')

    plt.xscale('log')
    fig.savefig("./paramfig.pdf")

    MAX = np.array([[KK[k],np.max(AI[k])] for k in range(len(AI))])
    plt.figure()
    plt.plot(MAX[:,0],MAX[:,1],"r")
    plt.xscale('log')
    plt.savefig("./max_AI_by_k.pdf")
    
    pts = []
    for k in range(len(AI)):
        print(np.argmax(AI[k]))
        for j in range(np.argmax(AI[k]),len(AI[k])-1):
            if AI[k][j] >= 1 and AI[k][j+1]<=1:
                pts.append([KK[k],cc[j]])
                break

    pts = np.array(pts)

    plt.figure()

    plt.plot(pts[:,0],pts[:,1])
    plt.xscale('log')
    plt.yscale('log')

    plt.savefig("./trans_by_K.pdf")
    
if __name__ == "__main__":
    run()
