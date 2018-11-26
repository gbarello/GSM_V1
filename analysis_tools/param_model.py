import numpy as np
import GSM.MGSM_inference as MGSM
import scipy

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 24
plt.rcParams['lines.linewidth'] = 4.

cm = plt.get_cmap('gist_rainbow')

colortab = ['r','m','b']

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
    
    gauss = np.exp((np.cos(2*np.pi*distance/(n-1)) - 1)/(w*w)) - dr
    
    #    out = np.dot(gauss,gauss)
    out = gauss
    
    return out/out[0,0]

def cov_special(n,w,dr = 0):

    if w == 0:
        return np.identity(n)
    elif w == -1:
        return np.ones([n,n])
    else:
        return cov(n,w,dr)

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
    NUM_W = 20
    
    CMAX = 1.
    CMIN = -2.
    N = 8

    I1 = inp(N,.6,1,1,-1 + N/2)
    I2 = inp(N,.6,1,0,-1 + N/2)
    I3 = inp(N,.6,0,1,-1 + N/2)

    KK = np.logspace(0,1,NUM_CURVE)

    con = np.logspace(CMIN,CMAX,NUM_CON)

    WW = np.linspace(.1,2,NUM_W)

    for w in range(len(WW)):
        resp1.append([])
        resp2.append([])
        resp3.append([])

        nresp1.append([])
        nresp2.append([])
        nresp3.append([])

        for k in KK:
            print(k)
            W = WW[w]
            CC = cov(N,W)
            NC = CC
            
            np.savetxt("./covariance_test.csv",CC)
            
            stim1 = np.array([cc * I1 for cc in np.logspace(CMIN,CMAX,NUM_CON)])
            stim2 = np.array([cc * I2 for cc in np.logspace(CMIN,CMAX,NUM_CON)])
            stim3 = np.array([cc * I3 for cc in np.logspace(CMIN,CMAX,NUM_CON)])
            
            resp1[w].append(MGSM.gexp(0,stim1,CC,NC/(k*k)))#,precom = False))
            resp2[w].append(MGSM.gexp(0,stim2,CC,NC/(k*k)))#,precom = False))
            resp3[w].append(MGSM.gexp(0,stim3,CC,NC/(k*k)))#,precom = False))
            
            nresp1[w].append(MGSM.gnn(stim1,CC)[:,0])
            nresp2[w].append(MGSM.gnn(stim2,CC)[:,0])
            nresp3[w].append(MGSM.gnn(stim3,CC)[:,0])

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
    cc = 100*np.logspace(CMIN,CMAX,NUM_CON)/(10**CMAX)

    for w in [1]:
        for k in range(0,len(AI[w]),len(AI[w])/5):
            ax.plot(100*con/np.max(con),AI[w,k],label=str(K[k]))
            ax.plot(100*con/np.max(con),nAI[w,k],'--')
        
    handles, labels = ax.get_legend_handles_labels()
#    ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
#               ncol=2, mode="expand", borderaxespad=0.)
    ax.plot(cc,[1 for k in range(len(cc))],'k--')

    plt.xlabel("Contrast")
    plt.ylabel("A.I.")

    plt.tight_layout()

    fig.savefig("./paramfig_K.pdf")

    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_prop_cycle("color",[cm(1.*i/NUM_CURVE) for i in range(NUM_CURVE)])

    WW = np.linspace(.1,2,NUM_W)
    cc = 100*np.logspace(CMIN,CMAX,NUM_CON)/(10**CMAX)

    for k in [5]:
        for w in range(0,len(AI),len(AI)/5):
            ax.plot(100*con/np.max(con),AI[w,k],label=str(WW[w]))
            ax.plot(100*con/np.max(con),nAI[w,k],'--')
        
    handles, labels = ax.get_legend_handles_labels()
#    ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
#               ncol=2, mode="expand", borderaxespad=0.)
    ax.plot(cc,[1 for k in range(len(cc))],'k--')

    plt.xlim([0,20])
    plt.xlabel("Contrast")
    plt.ylabel("A.I.")

    plt.tight_layout()

    fig.savefig("./paramfig_W.pdf")
    
    plt.figure()
    for w in [5]:
        MAX = np.array([[1./KK[k],np.max(AI[w,k])] for k in range(len(AI[w]))])
        plt.plot(MAX[:,0],MAX[:,1],colortab[0])
#    plt.xscale('log')

    plt.xlabel("1/SNR")
    plt.ylabel("A.I.")

    plt.tight_layout()

    plt.savefig("./max_AI_by_k.pdf")

    plt.figure()
    for k in [5]:
        MAX = np.array([[WW[w]*180./np.pi,np.max(AI[w,k])] for w in range(len(AI))])
        plt.plot(MAX[:,0],MAX[:,1],colortab[0])
        
    plt.xlabel("Width (deg.)")
    plt.ylabel("A.I.")
    plt.tight_layout()

    plt.savefig("./max_AI_by_W.pdf")

    
    plt.figure()

    for w in [5]:
        pts = []
        for k in range(len(AI[w])):
            pts.append([1./KK[k],get_trans(AI[w,k],100*con/np.max(con))])
            
        pts = np.array(pts)
        
        plt.plot(pts[:,0],pts[:,1],colortab[0])

    plt.xlabel("1/SNR")
    plt.ylabel("Contrast")

    plt.xscale('log')
    plt.yscale('linear')
        
    plt.tight_layout()

    plt.savefig("./trans_by_K.pdf")

    plt.figure()

    for k in [5]:
        pts = []
        for w in range(len(AI)):
            pts.append([WW[w]*180./np.pi,get_trans(AI[w,k],100*con/np.max(con))])
            
        pts = np.array(pts)
        
        plt.plot(pts[:,0],pts[:,1],colortab[0])


    plt.xlabel("Width (deg.)")
    plt.ylabel("Contrast")

    plt.yscale('linear')
        
    plt.tight_layout()

    plt.savefig("./trans_by_W.pdf")

    
def run_WDIFF_plots():

    #what it is that we want to do here? We want to look at COS


    resp1 = []
    resp2 = []
    resp3 = []
    
    nresp1 = []
    nresp2 = []
    nresp3 = []

    NUM_CURVE = 10
    NUM_CON = 50
    NUM_W = 20
    
    CMAX = 1.
    CMIN = -2.
    N = 8

    I1 = inp(N,.6,1,1,-1 + N/2)
    I2 = inp(N,.6,1,0,-1 + N/2)
    I3 = inp(N,.6,0,1,-1 + N/2)

    WW = np.logspace(.1,10,NUM_CURVE)

    con = np.logspace(CMIN,CMAX,NUM_CON)

    for w in [0,1,-1]:
        resp1.append([])
        resp2.append([])
        resp3.append([])

        nresp1.append([])
        nresp2.append([])
        nresp3.append([])

        for k in WW:
            W = w
            CC = cov(N,.6)

            if W == 0 or W == -1:
                NC = cov_special(N,W)
            else:
                NC = CC.copy()
                
            NC = NC*np.linalg.norm(CC)/np.linalg.norm(NC)
            
            np.savetxt("./covariance_test.csv",CC)
            
            stim1 = np.array([cc * I1 for cc in np.logspace(CMIN,CMAX,NUM_CON)])
            stim2 = np.array([cc * I2 for cc in np.logspace(CMIN,CMAX,NUM_CON)])
            stim3 = np.array([cc * I3 for cc in np.logspace(CMIN,CMAX,NUM_CON)])
            
            resp1[-1].append(MGSM.gexp(0,stim1,CC,NC/(k*k)))
            resp2[-1].append(MGSM.gexp(0,stim2,CC,NC/(k*k)))
            resp3[-1].append(MGSM.gexp(0,stim3,CC,NC/(k*k)))
            
            nresp1[-1].append(MGSM.gnn(stim1,CC)[:,0])
            nresp2[-1].append(MGSM.gnn(stim2,CC)[:,0])
            nresp3[-1].append(MGSM.gnn(stim3,CC)[:,0])

    resp1 = np.array(resp1)
    resp2 = np.array(resp2)
    resp3 = np.array(resp3)

    nresp1 = np.array(nresp1)
    nresp2 = np.array(nresp2)
    nresp3 = np.array(nresp3)

    AI = resp1/(resp2+resp3)
    nAI = nresp1/(nresp2+nresp3)

    for w in range(len(AI)):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_prop_cycle("color",[cm(1.*i/NUM_CURVE) for i in range(NUM_CURVE)])

        K = np.linspace(.5,5,NUM_CURVE)
        cc = 100*np.logspace(CMIN,CMAX,NUM_CON)/(10**CMAX)
        
        for k in range(0,len(AI[w]),len(AI[w])/5):
            ax.plot(100*con/np.max(con),AI[w,k],label=str(K[k]))
            ax.plot(100*con/np.max(con),nAI[w,k],'--')
            
        handles, labels = ax.get_legend_handles_labels()
        #    ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
        #               ncol=2, mode="expand", borderaxespad=0.)
        ax.plot(cc,[1 for k in range(len(cc))],'k--')
        plt.xlim([0,20])
        plt.xlabel("Contrast")
        plt.ylabel("A.I.")

        plt.tight_layout()
        
        fig.savefig("./paramfig_K_{}_by_W.pdf".format(w))

def run_cov():

    #what it is that we want to do here? We want to look at COS


    resp1 = []
    resp2 = []
    resp3 = []
    
    nresp1 = []
    nresp2 = []
    nresp3 = []

    NUM_CURVE = 10
    NUM_CON = 50
    NUM_W = 10
    
    CMAX = 1.
    CMIN = -1.
    N = 8

    I1 = inp(N,.6,1,1,-1 + N/2)
    I2 = inp(N,.6,1,0,-1 + N/2)
    I3 = inp(N,.6,0,1,-1 + N/2)

    KK = np.logspace(0,1,NUM_CURVE)

    con = np.logspace(CMIN,CMAX,NUM_CON)

    WW = [.1,.5,1.]
    
    for k in WW:
        
        print(k)
        CC = cov(N,.5)
        if k == WW[0]:
            NC = np.identity(N)
        else:
            NC = cov(N,k)

        NC = NC  * np.linalg.norm(CC)/(np.linalg.norm(NC)*4)

        sn = 1
        
        stim1 = np.array([cc * I1 for cc in np.logspace(CMIN,CMAX,NUM_CON)])
        stim2 = np.array([cc * I2 for cc in np.logspace(CMIN,CMAX,NUM_CON)])
        stim3 = np.array([cc * I3 for cc in np.logspace(CMIN,CMAX,NUM_CON)])
        
        resp1.append(sn*MGSM.gexp(0,stim1/sn,CC,NC,precom = False))
        resp2.append(sn*MGSM.gexp(0,stim2/sn,CC,NC,precom = False))
        resp3.append(sn*MGSM.gexp(0,stim3/sn,CC,NC,precom = False))

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

    print(AI[-1])
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_prop_cycle("color",[cm(1.*i/NUM_CURVE) for i in range(NUM_CURVE)])

    K = np.linspace(.5,5,NUM_CURVE)
    cc = 100*np.logspace(CMIN,CMAX,NUM_CON)/(10**CMAX)

    for k in range(0,len(AI)):
        ax.plot(100*con/np.max(con),AI[k],colortab[k],label=str(K[k]))
        ax.plot(100*con/np.max(con),nAI[k],colortab[k] + '--')
        
    handles, labels = ax.get_legend_handles_labels()
#    ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
#               ncol=2, mode="expand", borderaxespad=0.)
    ax.plot(cc,[1 for k in range(len(cc))],'k--')

    plt.xlabel("Contrast")
    plt.ylabel("A.I.")
    
    plt.tight_layout()

    fig.savefig("./paramfig_NC.pdf")
    

if __name__ == "__main__":
#    run_WDIFF_plots()
    run_k()
    #run_cov()
