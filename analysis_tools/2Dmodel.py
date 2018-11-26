import numpy as np
import GSM.MGSM_inference as GSM

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 25
plt.rcParams['lines.linewidth'] = 4.

def get(f):
    F = open(f,"r")

    out = []
    for l in F:
        temp = l.split(",")
        temp[-1] = temp[-1][:-1]

        out.append([float(x) for x in temp])

    return np.array(out)

def run():

    def c2(c):
        return np.array([[1.,c],[c,1.]])

    def cov(n,w,dr = 0):
        distance = np.array([[((i - j + (n/2.))%n) - n/2. for i in range(n)] for j in range(n)])

        gauss = np.exp(-(distance**2)/(2.*w*w)) - dr

        out = np.dot(gauss,gauss)

        return out/out[0,0]

    def inp(n,w,c1,c2,i):
        stim1 = np.array([((j + (n/2.))%n) - (n/2.) for j in range(n)])
        stim2 = np.array([(((j - (i%n)) + (n/2.))%n) - (n/2.) for j in range(n)])

        gauss1 = np.exp(-(stim1**2)/(2.*w*w))
        gauss2 = np.exp(-(stim2**2)/(2.*w*w))
        
        return c1*gauss1 + c2*gauss2

    LO= []
    HI = []
    nnLO = []
    nnHI = []

    ip = np.array([[1,0],[0,1],[1,1]])

    for c in np.linspace(-.9,.9,20):
        cov = c2(c)
        nc = np.identity(2)*(1 + c)

        sHI = GSM.gexp(0,20*ip,cov,nc*(2*2),precom = True)
        sLO = GSM.gexp(0,ip,cov,nc*(2*2),precom = False)
        snnLO = GSM.gnn(ip,cov)[:,0]
        snnHI = GSM.gnn(20*ip,cov)[:,0]
        
        HI.append([c,sHI[2]/(sHI[0])])
        LO.append([c,sLO[2]/(sLO[0])])
        nnHI.append([c,snnHI[2]/(snnHI[0])])
        nnLO.append([c,snnLO[2]/(snnLO[0])])
        
    HI = np.array(HI)
    LO = np.array(LO)
    nnHI = np.array(nnHI)
    nnLO = np.array(nnLO)

    plt.plot(HI[:,0],[1 for x in HI],'k--',linewidth = 1)

    plt.plot(HI[:,0],HI[:,1],'k')
    plt.plot(nnHI[:,0],nnHI[:,1],'k--')
    plt.plot(LO[:,0],LO[:,1],'r')        
    plt.plot(nnLO[:,0],nnLO[:,1],'r--')        

    plt.xlabel("Correlation")
    plt.ylabel("Modulation Ratio")

    plt.xlim(-1,1)
    plt.ylim(0,2)

    plt.xticks([-.5,0,.5])
     
    plt.tight_layout()    

    plt.savefig("./2DAIparam.pdf")
    print("done with AI")
    #what it is that we want to do here? We want to look at COS

    con = 20*np.logspace(-2,0,50)

    NN = 2
    WW = np.array([.15])

    out1 = []
    out2 = []
    
    nnout1 = []
    nnout2 = []


    out12 = []
    out22 = []
    
    nnout12 = []
    nnout22 = []
    
    NCOV = np.identity(2)
    
    k = .75
    
    I1 = np.array([[c1,c1/10] for c1 in con])
    I2 = np.array([[c1 + con[-1]/10,con[-1] + c1/10]for c1 in con])

    I12 = np.array([[c1,0] for c1 in con])
    I22 = np.array([[c1,con[-1]]for c1 in con])
    
    print(I1.shape)
    print(I2.shape)
    
    CC = c2(.6)
    NCOV = np.identity(2)
    
    out1 = GSM.gexp(0,I1,CC,NCOV/(k*k),precom = False)
    out2 = GSM.gexp(0,I2,CC,NCOV/(k*k),precom = False)
    
    nnout1 = GSM.gnn(I1,CC).T
    nnout2 = GSM.gnn(I2,CC).T

    out12 = GSM.gexp(0,I12,CC,NCOV/(k*k),precom = False)
    out22 = GSM.gexp(0,I22,CC,NCOV/(k*k),precom = False)
    
    nnout12 = GSM.gnn(I12,CC).T
    nnout22 = GSM.gnn(I22,CC).T

    print(nnout2)

    plt.figure()

    plt.plot(con/con[-1],out1,'r')
    plt.plot(con/con[-1],out2,'k')

    plt.plot(con/con[-1],nnout1[0],'r--')
    plt.plot(con/con[-1],nnout2[0],'k--')

    plt.xlabel("contrast")
    plt.ylabel("Respose")

#    plt.yscale("log")
    plt.xscale("log")

    plt.tight_layout()

   
    plt.savefig("./2Dssfig_0.pdf")

    plt.figure()

    plt.plot(con/con[-1],out12,'r')
    plt.plot(con/con[-1],out22,'k')

    plt.plot(con/con[-1],nnout12[0],'r--')
    plt.plot(con/con[-1],nnout22[0],'k--')

    plt.xlabel("contrast")
    plt.ylabel("Respose")

#    plt.yscale("log")
    plt.xscale("log")

    plt.tight_layout()

    plt.savefig("./2Dssfig_1.pdf")
    
if __name__ == "__main__":
    run()
