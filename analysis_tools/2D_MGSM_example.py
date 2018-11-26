import GSM.MGSM_inference as MGSM
import numpy as np

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 25
plt.rcParams['lines.linewidth'] = 4.

def cov(S,c):
    return S*S*np.array([[1.,c],[c,1.]])

def get_seg_probs(X,CC,NC,P):
    print("segprob")
    SEG1 = MGSM.NPShared(X,CC,NC)*P[0]

    SEG21 = MGSM.NPShared(np.reshape(X[:,0],[-1,1]),np.array([[CC[0,0]]]),np.array([[NC[0,0]]]))
    SEG22 = MGSM.NPShared(np.reshape(X[:,1],[-1,1]),np.array([[CC[1,1]]]),np.array([[NC[1,1]]]))
    SEG2 = SEG21*SEG22*P[1]

    NORM = SEG1 + SEG2


    nnSEG1 = MGSM.LPShared(X,CC)*P[0]

    nnSEG21 = MGSM.LPShared(np.reshape(X[:,0],[-1,1]),np.array([[CC[0,0]]]))
    nnSEG22 = MGSM.LPShared(np.reshape(X[:,1],[-1,1]),np.array([[CC[1,1]]]))
    nnSEG2 = nnSEG21*nnSEG22*P[1]

    nnNORM = nnSEG1 + nnSEG2

    O1 = np.reshape(np.array([SEG1/NORM,SEG2/NORM]),[2,-1])
    O2 = np.reshape(np.array([nnSEG1/nnNORM,nnSEG2/nnNORM]),[2,-1])

    print(np.max(O1))
    print(np.max(nnSEG2))
    print(np.min(O1))
    print(np.min(nnSEG2))

    return O1,O2

def main():

    #what are my goals here? 
    #I want to emulate the SS experiment. 

    Ncon = 100

    stim = np.reshape(np.array([[c1,c2] for c1 in np.logspace(-1,2,Ncon) for c2 in [0,50.]]),[-1,2])

    O1,O2 = get_seg_probs(stim,cov(1,.25),cov(10,0),[.5,.5])
    nO1,nO2 = get_seg_probs(stim,cov(1,.25),cov(.1,0),[.5,.5])

    O1 = np.reshape(O1,[2,Ncon,2])
    nO1 = np.reshape(nO1,[2,Ncon,2])
    stim = np.reshape(stim,[Ncon,2,2])

    plt.plot(stim[:,0,0],O1[0,:,0],'r')
    plt.plot(stim[:,1,0],O1[0,:,1],'b')

    plt.plot(stim[:,0,0],nO1[0,:,0],'r--')
    plt.plot(stim[:,1,0],nO1[0,:,1],'b--')
    plt.ylim(0,1)

    plt.xscale("log")


#    plt.plot(OUT[1,2,1,:,0,0],'r--')
#    plt.plot(OUT[1,2,1,:,1,0],'b--')

    plt.savefig("./2DMGSM_plot.pdf")


if __name__ == "__main__":
    main()
