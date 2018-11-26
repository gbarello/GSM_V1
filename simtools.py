import numpy as np
import GSM.MGSM_inference as MGSM

def GG(x,a,b,k,x0):
    return a + b * np.exp(1 - k*k*np.cos(2*(x - x0)))

def HWHH(a,b,k,x0):
    L1 = np.log(2*np.exp(k*k*np.cos(2*x0) - 1))
    L2 = np.log(1 + np.exp(2*k*k*np.cos(2*x0)))
    
    return (np.pi/2) - (2*x0 + np.arccos((1 + L1 - L2)/(k*k)))

def get_cor(Ncor,MODEL):
    if MODEL == "ours":
    
        #[CNS,C1,C2]
        NC1 = np.array(Ncor,copy = True)
        NC2 = np.array(Ncor[:8*6,:8*6])
        NC3 = np.array(Ncor[8*6:,8*6:])
        
        Nfull = Ncor
                
        return NC1,NC2,NC3,Nfull
        
    elif MODEL == "coen_cagli":
        #[CNS,C1,C2]
        
        NC1 = Ncor[:8,:8]
        
        ind = np.concatenate([np.arange(8),np.reshape(zip(np.arange(9,len(Ncor),8),np.arange(10,len(Ncor),8)),-1)])
        
        #    noind = [x for x in np.arange(len(Ncor)) if x not in ind]
        
        noind = [range(8 + i,len(Ncor),4) for i in range(4)]
        
        NC2 = np.array([[[Ncor[a,b] for a in ind] for b in ind] for k in range(4)])
        NC3 = np.array([[[Ncor[a,b] for a in noind[k]] for b in noind[k]] for k in range(4)])
        
        Nfull = Ncor
                
        return NC1,NC2,NC3,Nfull
    else:
        print("model not recognized!")
        exit()

def get_normalized_NCOR(Ncor,CNS,MODEL):

    NC1,NC2,NC3,Nfull = get_cor(Ncor,MODEL)

    nnorm = np.linalg.norm(NC1[:8,:8])
    cnorm = np.linalg.norm(CNS[:8,:8])
    
    NC1 = NC1 * cnorm / nnorm
    NC2 = NC2 * cnorm / nnorm
    NC3 = NC3 * cnorm / nnorm
    Nfull = Nfull * cnorm / nnorm
    
    return [NC1,NC2,NC3,Nfull]
        
def fit_ori_tuning(dat):
    
    k = 30. * np.pi / 180
    a = np.min(dat[:,1])
    b = (np.max(dat[:,1]) - np.min(dat[:,1]))/np.exp(1 + k*k)
    x0 = dat[np.argmax(dat[:,1]),0] + (np.pi/2)
    
    def e_part(dat,a,b,x0,k):
        return np.exp(1 - np.cos(2*(dat[:,0] - x0))*(k**2))
    
    def DLa(dat,a,b,x0,k):
        base = (dat[:,1] - (a + b*e_part(dat,a,b,x0,k)))
        return -2*np.sum(base)
    
    def DLb(dat,a,b,x0,k):
        base = (dat[:,1] - (a + b*e_part(dat,a,b,x0,k)))
        return -2*np.sum(base * e_part(dat,a,b,x0,k))
    
    def DLx0(dat,a,b,x0,k):
        base = (dat[:,1] - (a + b*e_part(dat,a,b,x0,k)))
        return 4*np.sum(base * b * e_part(dat,a,b,x0,k) * k * k * np.sin(2*(dat[:,0] - x0)))
    
    def DLk(dat,a,b,x0,k):
        base = (dat[:,1] - (a + b*e_part(dat,a,b,x0,k)))
        return 4*np.sum(base * b * e_part(dat,a,b,x0,k) * k * np.cos(2*(dat[:,0] - x0)))
    
    def LL(dat,a,b,x0,k):
        base = (dat[:,1] - (a + b*e_part(dat,a,b,x0,k)))
        return np.sum(base**2)
    
    da = DLa(dat,a,b,x0,k)
    db = DLb(dat,a,b,x0,k)
    dk = DLk(dat,a,b,x0,k)
    dx0 = DLx0(dat,a,b,x0,k)
    
    eps = 10**-5
    eta = .001
    
    ll0 = LL(dat,a,b,x0,k)
    
    while np.abs(da) > eps or np.abs(db) > eps or np.abs(dk) > eps or np.abs(dx0) > eps:
        
        ll1 = ll0
        
        a = a - eta*da
        b = b - eta*db
        k = k - eta*dk
        x0 = x0 - eta*dx0
        
        if x0 > np.pi:
            x0 -= np.pi
        if x0 < 0:
            x0 += np.pi
            
        da = DLa(dat,a,b,x0,k)
        db = DLb(dat,a,b,x0,k)
        dk = DLk(dat,a,b,x0,k)
        dx0 = DLx0(dat,a,b,x0,k)
        
        ll0 = LL(dat,a,b,x0,k)

        if ll0 > ll1:
            eta /= 3
        else:
            eta *= 1.1
            
    return a,b,k,x0

def make_resp(filt,model,quenched,K,mgs,tag,COR,ncor,n_trial = 1):


    P = COR[0]
    CNS = COR[1]
    C1 = COR[2]
    C2 = COR[3]

    NCNS = ncor[0]
    NC1 = ncor[1]
    NC2 = ncor[2]

    fullcor = ncor[3]

    print(CNS.shape)
    print(C1.shape)

    #make sure everything is the right type
    assert model in ("ours","coen_cagli")
    assert quenched in (True,False)
    assert type(ncor) == list

    GV = []
    GVnn = []

    #generate samples for each filter set
    if quenched:
        nftemp = np.tile(filt,[n_trial,1,1])

        noise = np.reshape(np.random.multivariate_normal(np.zeros(np.prod(nftemp.shape[1:])),fullcor,n_trial*filt.shape[0]),tuple([n_trial*filt.shape[0]]) + filt.shape[1:]) 
        nftemp += noise
    else:
        nftemp = filt

    #run it for each value of k

    SP = []
    SPnn = []
    
    for k in K:
        print(k)
        print(nftemp.shape)
        GV.append(MGSM.MGSM_g(nftemp,[CNS,C1,C2],[NCNS*(k*k),NC1*(k*k),NC2*(k*k)],P))            
        GVnn.append(MGSM.MGSM_gnn(filt,[CNS,C1,C2],P))
        if model == "ours":
            SP.append(MGSM.get_noisy_seg_weight(filt,CNS,C1,C2,NCNS*(k*k),NC1*(k*k),NC2*(k*k),P))
            SPnn.append(MGSM.get_seg_weight(filt,CNS,C1,C2,P))
        elif model == "coen_cagli":
            SP.append(MGSM.get_noisy_CC_seg_weight(filt,CNS,C1,C2,NCNS*(k*k),NC1*(k*k),NC2*(k*k),P))
            SPnn.append(MGSM.get_CC_seg_weight(filt,CNS,C1,C2,P))
            
            
    GV = np.concatenate(GV)
    GVnn = np.concatenate(GVnn)

    SP = np.concatenate(SP)
    SPnn = np.concatenate(SPnn)

    np.savetxt(tag + "_noisy.csv",GV)
    np.savetxt(tag + "_clean.csv",GVnn)
    
    np.savetxt(tag + "_SP_noisy.csv",SP)
    np.savetxt(tag + "_SP_clean.csv",SPnn)

    return GV,GVnn

def make_att_resp(filt1,filt2,model,quenched,K,mgs,tag,COR,ncor,qcor,fcor,n_trial = 1):


    P = COR[0]
    CNS = COR[1]
    C1 = COR[2]
    C2 = COR[3]

    NCNS = ncor[0]
    NC1 = ncor[1]
    NC2 = ncor[2]
    
    QCNS = qcor[0]
    QC1 = qcor[1]
    QC2 = qcor[2]

    FCNS = fcor[0]
    FC1 = fcor[1]
    FC2 = fcor[2]

    fullcor = ncor[3]
    fullqcor = qcor[3]

    print(CNS.shape)
    print(C1.shape)

    #make sure everything is the right type
    assert model in ("ours","coen_cagli")
    assert quenched in (True,False)
    assert type(ncor) == list

    GV = []
    GVnn = []

    #run it for each value of k

    SP = []
    SPnn = []
    
    for k in K:
        print(k)
        GV.append(MGSM.MGSM_att_g(filt1,filt2,[CNS,C1,C2],[NCNS*(k*k),NC1*(k*k),NC2*(k*k)],[QCNS,QC1,QC2],[FCNS,FC1,FC2],P))
        GVnn.append(MGSM.MGSM_gnn(filt2,[CNS,C1,C2],P))

        SP.append(MGSM.get_att_CC_seg_weight(filt1,filt2,CNS,C1,C2,NCNS*(k*k),NC1*(k*k),NC2*(k*k),QCNS,QC1,QC2,FCNS,FC1,FC2,P))
        SPnn.append(MGSM.get_CC_seg_weight(filt2,CNS,C1,C2,P))
            
            
    GV = np.concatenate(GV)
    GVnn = np.concatenate(GVnn)

    SP = np.concatenate(SP)
    SPnn = np.concatenate(SPnn)

    np.savetxt(tag + "_noisy.csv",GV)
    np.savetxt(tag + "_clean.csv",GVnn)
    
    np.savetxt(tag + "_SP_noisy.csv",SP)
    np.savetxt(tag + "_SP_clean.csv",SPnn)

    return GV,GVnn

def make_GSM_resp(filt,quenched,K,mgs,tag,COR,ncor,n_trial = 1):

    #make sure everything is the right type
    assert quenched in (True,False)
    assert type(ncor) == np.ndarray

    GV = []
    GVnn = []

        
    #generate samples for each filter set
    if quenched:
        nftemp = np.tile(filt,[n_trial,1])

        print(nftemp.shape)

        noise = np.reshape(np.random.multivariate_normal(np.zeros(np.prod(nftemp.shape[1:])),ncor,n_trial*filt.shape[0]),tuple([n_trial*filt.shape[0]]) + filt.shape[1:])

        nftemp += noise
    else:
        nftemp = filt

    #run it for each value of k
    for k in K:
        GV.append(np.array([MGSM.gexp(i,nftemp,COR,k*k*ncor,precom = True) for i in range(filt.shape[1])]).transpose())

        GVnn.append(MGSM.gnn(filt,COR))
        print(GV[-1].shape)
            
    GV = np.concatenate(GV)
    GVnn = np.concatenate(GVnn)
        
    print(tag)
    print(GV.shape)
    print(GVnn.shape)
        
    np.savetxt(tag + "_noisy.csv",GV)
    np.savetxt(tag + "_clean.csv",GVnn)
