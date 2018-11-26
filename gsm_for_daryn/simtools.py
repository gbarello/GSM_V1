import numpy as np
import GSM.MGSM_inference as MGSM

def make_resp(filt,model,quenched,K,mgs,tag,COR,ncor,n_trial = 1,NOISE = True):

    P = COR[0]
    CNS = COR[1]
    C1 = COR[2]
    C2 = COR[3]

    print(P.shape)
    print(CNS.shape)
    print(C1.shape)
    print(C2.shape)
    
    GVnn = MGSM.MGSM_gnn(filt,CNS,C1,C2,P,model)

    return GVnn
    
