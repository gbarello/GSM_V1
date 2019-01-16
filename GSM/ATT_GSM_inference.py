from MGSM_inference import *

def att_egia(n,I,a,cov,ncov,qcov,F,getall = False,getP = False):
    #we're basically going to perform 2 steps of kalman update equations starting with the prior
    #I can be a list of images, and I will write it recursively to handle any number
    
    g = np.zeros(len(cov))
    p = cov
    out = [g]
    pout = [p]
    
    for i in I:        
        g,p = att_egia_update(a,i,g,p,ncov,qcov,F)
        out.append(g)
        pout.append(p)
        g = np.dot(F,g)
        p = np.dot(F,np.dot(p,np.transpose(F))) + qcov
        
    if getP:
        return np.array(out),np.array(pout)
    
    if getall:
        return np.array(out)
    
    return out[-1][n]

def att_egia_update(a,I,g,p,ncov,q,F):
    K = a*np.dot(p,np.linalg.inv(a*a*p + ncov))

    go = g + np.dot(K,I - a*g)
    P = np.dot(np.identity(len(I)) - a*K,p)

    return go,P

def att_PIA(I1,I2,a,cov,ncov,qcov):
    C1 = a*a*cov + ncov
    C2,ldC2 = att_helper_1_cov(a,cov,ncov,qcov)
    s,ldC1 = np.linalg.slogdet(C1)

    Iatt = att_I_helper(I1,I2,a,cov,ncov,qcov)
 
    ldist = -(0.5)*(IP(I1,np.linalg.inv(C1),I1) + IP(Iatt,C2,Iatt)) - len(C1)*np.log(2*np.pi) - (0.5)*(ldC1 + ldC2)
#    ldist = np.exp(-(0.5)*(IP(I1,np.linalg.inv(C1),I1) + IP(Iatt,C2,Iatt)))/(np.power(2*np.pi,len(C1))*np.sqrt(np.exp(ldC1+ldC2)))

    return np.exp(ldist)

def att_I_helper(I1,I2,a,cov,ncov,qcov):
    xi = np.linalg.inv(np.linalg.inv(cov) + a*np.linalg.inv(ncov))
    out = I2 - a*a*np.dot(xi,np.dot(np.linalg.inv(ncov),I1))
    return out
    
def att_helper_1_cov(a,cov,ncov,qcov):
    xi = np.linalg.inv(np.linalg.inv(cov) + a*np.linalg.inv(ncov))
    CC = a*a*qcov + a*a*xi + ncov
    s,logdet = np.linalg.slogdet(CC)

    return np.linalg.inv(CC),logdet

def att_PIA_iter_OLD(I,a,cov,ncov,qcov,F,log = True):
    
    n = len(cov)
    m = len(I)

    iF = np.linalg.inv(F)

    s,logdetF = np.linalg.slogdet(F)

    em = I[0]/a
    xi = ncov/(a*a)
    
    out = - n*m*np.log(a)

    for k in range(m-1):
        tempcov = (ncov/a/a) + np.dot(np.dot(F.transpose(),xi + qcov),F)
        temparg = np.dot(iF,em) - (I[k+1]/a)
        
        s,logdet = np.linalg.slogdet(tempcov)
        
        out -= n*np.log(2*np.pi)/2 + logdetF
        out -= logdet/2
        out -= np.dot(np.dot(temparg,np.linalg.inv(tempcov)),temparg)/2
        
        em,xi = get_new_mc_old(em,xi,I[k+1]/a,ncov/a/a,qcov,F)

    s,logdet = np.linalg.slogdet(xi + cov)
    
    out -= n*np.log(2*np.pi)/2
    out -= logdet/2
    out -= np.dot(np.dot(em,np.linalg.inv(xi + cov)),em)/2

    if log:
        return(out)
    else:
        return np.exp(out)


def att_PIA_iter(I,a,cov,ncov,qcov,F,log = True):
    
    n = len(cov)
    m = len(I)

    C = a*a*cov
    Q = a*a*qcov

    iF = np.linalg.inv(F)
    s,ldF = np.linalg.slogdet(F)

    mu = I[-1]
    xi = ncov

    if len(I) == 1:
        Vtemp = mu
        Ctemp = xi + C
        _,ld = np.linalg.slogdet(Ctemp)
        return (- n * np.log(2*np.pi) - ld - dot([Vtemp,inv(Ctemp),Vtemp]))/2

    out = 0

    for k in reversed(range(1,len(I))):
        Ctemp = ncov + dot([iF,xi + Q,iF.transpose()])
        _,ldtemp = np.linalg.slogdet(Ctemp)
        Vtemp = dot([iF,mu]) - I[k-1]
        out += ldF - (n * np.log(2*np.pi) + ldtemp + dot([Vtemp,inv(Ctemp),Vtemp]))/2
        mu,xi = get_new_mc(mu,xi,I[k-1],C,Q,ncov,iF)
        
    Ctemp = C + xi
    _,ldtemp = np.linalg.slogdet(Ctemp)
    Vtemp = mu

    out += ldF - (n * np.log(2*np.pi) + ldtemp + dot([Vtemp,inv(Ctemp),Vtemp]))/2
    
    if log:
        return(out)
    else:
        return np.exp(out)


def inv(x):
    return np.linalg.inv(x)

def get_new_mc_old(m,x,mu,cov,qcov,F):
    tempcov = np.dot(np.dot(np.transpose(F),x+qcov),F)
    xi = inv(inv(cov) + inv(tempcov))
    em = np.dot(xi,np.dot(inv(cov),mu)+np.dot(inv(tempcov),np.dot(inv(F),m)))

    return em,xi

def get_new_mc(mu,xi,I,cov,qcov,ncov,iF):
    tempcov = dot([iF,xi + qcov,iF.transpose()])
    ixi = inv(ncov) + inv(tempcov)
    xi = inv(ixi)
    em = dot([ixi,(dot([ncov,iF,mu]) + dot([tempcov,I]))])

    return em,xi

def att_gnn(x,cov,qcov,F):
    lam = sum([LAM(x[:,0],cov)**2] + [LAM(x[:,i] - np.tensordot(x[:,i-1],F,axis = [1,1]),qcov)**2 for i in range(1,len(x[0]))])

    cdet = np.linalg.det(cov)
    qdet = np.linalg.det(qcov)

    m = len(x[0])
    n = len(x[0][0])

    lcoef = np.power(lam,(1. - m*n)/4)
    bess = np.float32(mpbesselk((1. - m*n)/2,np.sqrt(lam) + EPS))

    return x[:,-1]*lcoef*bess

def att_gexp(n,x,cov,ncov,qcov,F):
    
    TOP = [INT(lambda a: att_integrand(n,x[k],a,cov,ncov,qcov,F),0,np.inf)[0] for k in range(len(x))]
    
    NORM = [INT(lambda a: att_norm(x[k],a,cov,ncov,qcov,F),0,np.inf)[0] for k in range(len(x))]
    
    OUT = [TOP[k]/NORM[k] if NORM[k] != 0 else 0 for k in range(len(TOP))]
    
    return np.array(OUT)

def att_pexp(n,x,cov,ncov,qcov,F):
    
    TOP = [INT(lambda a: att_P_integrand(n,x[k],a,cov,ncov,qcov,F),0,np.inf)[0] for k in range(len(x))]
    
    NORM = [INT(lambda a: att_norm(x[k],a,cov,ncov,qcov,F),0,np.inf)[0] for k in range(len(x))]
    
    OUT = [TOP[k]/NORM[k] if NORM[k] != 0 else 0 for k in range(len(TOP))]
    
    return np.array(OUT)

def general_att_gexp(x,cov,ncov,qcov,F,ind):
    
    TOP = [np.array([INT(lambda a: att_integrand(j,x[k],a,cov,ncov,qcov,F),0,np.inf)[0] for j in ind]) for k in range(len(x))]
    
    NORM = [INT(lambda a: att_norm(x[k],a,cov,ncov,qcov,F),0,np.inf)[0] for k in range(len(x))]
    
    OUT = [TOP[k]/NORM[k] if NORM[k] != 0 else np.zeros_like(TOP[k]) for k in range(len(TOP))]

    return np.array(OUT)

def general_att_pexp(x,cov,ncov,qcov,F,ind):
    
    TOP = [np.array([INT(lambda a: att_P_integrand(j,x[k],a,cov,ncov,qcov,F),0,np.inf)[0] for j in ind]) for k in range(len(x))]
    
    NORM = [INT(lambda a: att_norm(x[k],a,cov,ncov,qcov,F),0,np.inf)[0] for k in range(len(x))]
    
    OUT = [TOP[k]/NORM[k] if NORM[k] != 0 else np.zeros_like(TOP[k]) for k in range(len(TOP))]

    return np.array(OUT)

def att_gexp_full(x,cov,ncov,qcov,F):
    return general_att_gexp(x,cov,ncov,qcov,F,range(x.shape[2]))

def att_pexp_full(x,cov,ncov,qcov,F):
    return general_att_pexp(x,cov,ncov,qcov,F,range(x.shape[2]))

def get_MGSM_att_weights(F,segs,C,NC,QC,FC,P):
    #(I,cov,ncov,qcov,F)
    gsm_resp = np.array([np.log(P[k]) + np.sum([np.log(att_PShared(F[:,:,segs[k][j]],C[k][j],NC[k][j],QC[k][j],FC[k][j])) for j in range(len(segs[k]))],axis = 0) for k in range(len(segs))]) # shape [nseg,ndata]

    b = np.amax(gsm_resp,axis = 0,keepdims = True)
    
    out = np.transpose(np.squeeze(np.exp(gsm_resp-b)/np.sum(np.exp(gsm_resp-b),axis = 0,keepdims = True)))

    if np.any(np.isnan(out)):
        print("wtf MGSM_att_weights")
        print(gsm_resp)
        exit()

    return out


def find_GSM_pia_max(I,C,NC,QC,FC,low,high,start,eps):
    out = find_f_max(lambda a:Pa(a,log = True) + att_PIA_iter(I,a,C,NC,QC,FC,log = True),low=low,high=high,start=start,eps = eps)
    return out
    
def stable_get_MGSM_att_weights(F,segs,C,NC,QC,FC,P,ind,rev_ind,ifrac = 100,npnt = 1000,op = True,calc_p = False):
    #(I,cov,ncov,qcov,F)

    if op == True:
        PIA_max = np.array([[[find_GSM_pia_max(F[i][:,segs[k][j]],C[k][j],NC[k][j],QC[k][j],FC[k][j],low=0,high=np.inf,start=10.,eps = 1.) for i in range(F.shape[0])] for j in range(len(segs[k]))] for k in range(len(segs))]) # shape [nseg,ndata]
        
        logints = np.array([[[logint.integrate_log(lambda a:Pa(a,log = True) + att_PIA_iter(F[i][:,segs[k][j]],a,C[k][j],NC[k][j],QC[k][j],FC[k][j],log = True),PIA_max[k][j][i][0]/ifrac,ifrac*PIA_max[k][j][i][0],[PIA_max[k][j][i][0]]) for i in range(len(PIA_max[k][j]))] for j in range(len(PIA_max[k]))] for k in range(len(PIA_max))])
    else:
        logints = np.array([[[logint.integrate_log(lambda a:Pa(a,log = True) + att_PIA_iter(F[i][:,segs[k][j]],a,C[k][j],NC[k][j],QC[k][j],FC[k][j],log = True),.0001,5000,[]) for i in range(F.shape[0])] for j in range(len(segs[k]))] for k in range(len(segs))])
        
    if calc_p:
        GNlogints = np.array([np.concatenate([np.array([expectation_log(lambda a:att_egia(ind[k][j],F[i][:,segs[k][j]],a,C[k][j],NC[k][j],QC[k][j],FC[k][j],getP = True)[1][-1],lambda a: Pa(a,log = True) + att_PIA_iter(F[i][:,segs[k][j]],a,C[k][j],NC[k][j],QC[k][j],FC[k][j],log = True),logints[k][j][i][1]["points"],n_d_exp = 2) if len(ind[k][j])>0 else [] for i in range(F.shape[0])]) for j in range(len(segs[k]))],axis = 1) for k in range(len(segs))])
        
    else:
        GNlogints = np.array([np.concatenate([np.array([expectation_log(lambda a:att_egia(ind[k][j],F[i][:,segs[k][j]],a,C[k][j],NC[k][j],QC[k][j],FC[k][j]),lambda a: Pa(a,log = True) + att_PIA_iter(F[i][:,segs[k][j]],a,C[k][j],NC[k][j],QC[k][j],FC[k][j],log = True),logints[k][j][i][1]["points"]) if len(ind[k][j])>0 else [] for i in range(F.shape[0])]) for j in range(len(segs[k]))],axis = 1)[:,rev_ind[k]] for k in range(len(segs))])
    
    mgsm_resp = np.array([np.log(P[k]) + np.array([np.sum([logints[k][j][i][0] for j in range(len(segs[k]))]) for i in range(F.shape[0])]) for k in range(len(segs))]) # shape [nseg,ndata]

    b = np.amax(mgsm_resp,axis = 0,keepdims = True)
    
    out = np.squeeze(np.exp(mgsm_resp-b)/np.sum(np.exp(mgsm_resp-b),axis = 0,keepdims = True))

    if np.any(np.isnan(out)):
        print("wtf MGSM_att_weights")
        print(gsm_resp)
        exit()


    return out,GNlogints

def general_MGSM_g_att(F,segs,C,NC,QC,FC,P,ind,ifrac = 1.5,npnt = 1000,stable = True,op=True):

    '''
    Description:performs MGSM inference on a general MGSM model with attention and noise.

    args:
     F - filters [ndata,nfilt]
     segs - a list of indices for each seg [[S11...S1n1],...,[Sm1...Smnm]] where each Sij is a list of indices for the filters in the ith segment of teh mth segmentation
     C - a list of g covariance matrices [[C11...Cn1]...[Cm1...Cmnm]]
     NC - a list of noise covariance matrices [[C11...Cn1]...[Cm1...Cmnm]]
     QC - a list of process noise covariance matrices [[C11...Cn1]...[Cm1...Cmnm]]
     F - a list of dynamics matrices [[C11...Cn1]...[Cm1...Cmnm]]
     P - a list of probabilities for each seg. [P1,...,Pm]
     ind - a list of indices specifying which filters you want to compute
    '''

    #seg_to_index = [reverse_index(np.concatenate(s)) for s in segs]
    seg_index = get_seg_indices(segs,ind)
    seg_to_ind = [reverse_indices(np.concatenate([np.array(segs[s][k])[seg_index[s][k]] for k in range(len(seg_index[s]))]),ind) for s in range(len(seg_index))]

    if stable:
        probs,gsm_resp = stable_get_MGSM_att_weights(F,segs,C,NC,QC,FC,P,seg_index,seg_to_ind,ifrac=ifrac,npnt=npnt,op=op,calc_p=False)#[nseg,ndata]
    else:
        probs = np.transpose(get_MGSM_att_weights(F,segs,C,NC,QC,FC,P))#[nseg,ndata]

    if np.any(np.isnan(probs)):
        print("probs")
        exit()
        
    #att_gexp(n,x,cov,ncov,qcov,F)
    if stable == False:
        gsm_resp = np.array([np.concatenate([general_att_gexp(F[:,:,segs[k][j]],C[k][j],NC[k][j],QC[k][j],FC[k][j],seg_index[k][j]) for j in range(len(segs[k]))],axis = 1)[:,seg_to_ind[k]] for k in range(len(segs))]) # shape [nseg,ndata,nfilt]

    if np.any(np.isnan(gsm_resp)) or np.any(np.isinf(gsm_resp)):
        print("gsm_resp")
        exit()

        
    result = np.sum(np.expand_dims(probs,-1) * gsm_resp,axis = 0)

    return result

def general_MGSM_p_att(F,segs,C,NC,QC,FC,P,ind,ifrac = 1.5,npnt = 1000,stable = True,op=True):

    '''
    Description:performs MGSM inference on a general MGSM model with attention and noise.

    args:
     F - filters [ndata,nfilt]
     segs - a list of indices for each seg [[S11...S1n1],...,[Sm1...Smnm]] where each Sij is a list of indices for the filters in the ith segment of teh mth segmentation
     C - a list of g covariance matrices [[C11...Cn1]...[Cm1...Cmnm]]
     NC - a list of noise covariance matrices [[C11...Cn1]...[Cm1...Cmnm]]
     QC - a list of process noise covariance matrices [[C11...Cn1]...[Cm1...Cmnm]]
     F - a list of dynamics matrices [[C11...Cn1]...[Cm1...Cmnm]]
     P - a list of probabilities for each seg. [P1,...,Pm]
     ind - a list of indices specifying which filters you want to compute
    '''

    #seg_to_index = [reverse_index(np.concatenate(s)) for s in segs]
    seg_index = get_seg_indices(segs,ind)
    seg_to_ind = [reverse_indices(np.concatenate([np.array(segs[s][k])[seg_index[s][k]] for k in range(len(seg_index[s]))]),ind) for s in range(len(seg_index))]

    if stable:
        probs,gsm_resp = stable_get_MGSM_att_weights(F,segs,C,NC,QC,FC,P,seg_index,seg_to_ind,ifrac=ifrac,npnt=npnt,op=op,calc_p = True)#[nseg,ndata]
    else:
        probs = np.transpose(get_MGSM_att_weights(F,segs,C,NC,QC,FC,P))#[nseg,ndata]

    if np.any(np.isnan(probs)):
        print("probs")
        exit()
        
    #att_gexp(n,x,cov,ncov,qcov,F)
    if stable == False:
        gsm_resp = np.array([np.concatenate([general_att_pexp(F[:,:,segs[k][j]],C[k][j],NC[k][j],QC[k][j],FC[k][j],seg_index[k][j]) for j in range(len(segs[k]))],axis = 1)[:,seg_to_ind[k]] for k in range(len(segs))]) # shape [nseg,ndata,nfilt]

    if np.any(np.isnan(gsm_resp)) or np.any(np.isinf(gsm_resp)):
        print("gsm_resp")
        exit()

        
    result = np.sum(np.expand_dims(np.expand_dims(probs,-1),-1) * gsm_resp,axis = 0)

    return result


def CC_MGSM_att_g(F1,F2,CC,CCS,CS,NCC,NCCS,NCS,QCC,QCCS,QCS,FCC,FCCS,FCS,P):
    
    '''
    Description: computes the expected g of the center filter set in the noisy MGSM
    '''

    #first we need to get the assignment probabilities:
    #[[],..,[]]

    probs = np.transpose(get_att_CC_seg_weight(F1,F2,CC,CCS,CS,NCC,NCCS,NCS,QCC,QCCS,QCS,FCC,FCCS,FCS,P))
    
    gv = np.array([[att_gexp(n,np.transpose([get_CC_seg_x(F1,i)[0],get_CC_seg_x(F2,i)[0]],[1,0,2]),CC if i == 0 else CCS[i-1],NCC if i == 0 else NCCS[i-1],QCC if i == 0 else QCCS[i-1],FCC if i == 0 else FCCS[i-1]) for n in range(8)] for i in range(len(P))])

    return np.transpose((np.reshape(probs,(len(P),1,-1))*gv).sum(axis = 0),(1,0))
    

def att_PShared_nonoise(d,segs,C,QC,FF,P,fq_shared = False,f_ID = False):

    '''
    Description: given filters and covariances computes P[x|cov,shared]
    inputs: 
    - x - filter values - [n_data,n_site,n_ori,n_phase]
    - c - covariance - [nsite*n_ori*n_phase,nsite*n_ori*n_phase]
    '''

    if f_ID:
        FC = [[np.eye(len(vtoS(C[f][m])))*sigmoid(FF[f][m]) for m in range(len(FF[f]))] for f in range(len(FF))]
    else:
        FC = FF
        
    ll = np.array([np.sum(np.array([att_LogLikelihood(d[:,:,segs[s][c]],C[s][c],QC[s][c],FC[s][c],mean = False,fq_shared = fq_shared) for c in range(len(segs[s]))]),axis = 0) + np.log(P[s]) for s in range(len(segs))])
    
    #ll -> [len(segs),ndat]

    b = np.amax(ll,0,keepdims=True)
    
    return np.transpose(np.exp(ll-b)/np.sum(np.exp(ll-b),axis = 0,keepdims = True))
    
def att_MGSM_loglik(d,segs,C,QC,F,P,fq_shared = False,f_ID = False):

    '''
    Description: given filters and covariances computes P[x|cov,shared]
    inputs: 
    - x - filter values - [n_data,n_site,n_ori,n_phase]
    - c - covariance - [nsite*n_ori*n_phase,nsite*n_ori*n_phase]
    '''

    if f_ID:
        FC = [[sigmoid(F)*np.eye(len(m)) for m in s] for s in segs]
    else:
        FC = F
        
    ll = np.array([np.sum(np.array([att_LogLikelihood(d[:,:,segs[s][c]],C[s][c],QC[s][c],FC[s][c],mean = False,fq_shared = fq_shared) for c in range(len(segs[s]))]),axis = 0) + np.log(P[s]) for s in range(len(segs))])

    lm = np.max(ll,axis = 0,keepdims = True)

    ll = np.log(np.sum(np.exp(ll - lm),axis = 0)) + lm
    #ll -> [len(segs),ndat]
    return np.mean(ll)

def att_LogLikelihood(X,vc,vq,F,mean = True,fq_shared = False):

    C = vtoS(vc)

    if fq_shared:
        Q = Q_self_con(C,F)
    else:
        Q = vtoS(vq)

    lam,_ = att_LAM(C,Q,F,X)
    lam = rectify(lam)

    n = X.shape[-1]
    m = X.shape[-2]-1

    _,ldc = np.linalg.slogdet(C)
    _,ldq = np.linalg.slogdet(Q)

    coef = -((n + n*m)*np.log(2*np.pi) + ldc + m*ldq)/2

    val = np.float32(mplog(mpbesselk(1 - (m+1.)*n/2.,np.sqrt(lam) + EPS))) + (1 - (m+1.)*n/2.)*np.log(lam)/2
    tval = lam

    aa = np.any(np.isnan(val))

    if aa:
#        print(np.concatenate([np.expand_dims(val,-1),np.expand_dims(tval,-1)],-1))
        print(np.any(np.isnan(vc)),np.any(np.isnan(vc)),np.any(np.isnan(vc)))
        print(C,Q,F)

        print("logdets:",ldc,ldq)
        exit()
        
    if mean:
        return np.mean(coef + val)
    else:
        return coef + val
    
def f_LP_att_grad(X,vc,vq,f_mat,fq_shared = False):

    '''
    This computes the gradient of PShared w.r.t. cov

    if fq_shared then f and q are considered to be related by the self-consistency equation S = F.S.F. + Q for S the covariance.

    we will compute F gradients, and keep Q fixed accordingly

    in particular Q = S - F.S.F -> dQ/dF ~ - S.F
    '''
    tt = time.time()
    
    x = X

    c_cho = vtoCH(vc)
    c_mat = vtoS(vc)

    N = int(c_mat.shape[0])
    M = int(X.shape[1]) - 1

    DC = np.linalg.det(c_mat) #DT used to be det(C)
    
    CI = np.linalg.inv(c_mat)

    if fq_shared == False:
        q_cho = vtoCH(vq)
        q_mat = vtoS(vq)
    else:
        q_mat = vq

    DQ = np.linalg.det(q_mat)

    QI = np.linalg.inv(q_mat)
#    TIT = np.transpose(TI)

    #DdetT = np.reshape(np_DDS(cov),[1,N,N])#derivative of determinant w.r.t matric

    xTx,q_x = att_LAM(c_mat,q_mat,f_mat,x)

    #deriv. of x.C^-1.x w.r.t. C .DxTx = np_DXDX(cov,x)

    cDxTx = np_DXDX(c_mat,x[:,0,:])
    qDxTx = np_DXDX_att(q_mat,q_x)

    temp1 = np.tensordot(x[:,1:],QI,axes = [2,0])
    temp2 = np.tensordot(np.tensordot(x[:,:-1],f_mat,axes = [2,1]),QI,axes = [2,1])
        
    fDxTx = np.sum(2*x[:,:-1,None,:]*(temp2[:,:,:,None]-temp1[:,:,:,None]),axis = 1)

    #    denom = np.sqrt(DC)*np.power(np.sqrt(DQ),m)*scipy.special.kvmp.(1 - (N*(M+1.)/2),np.sqrt(xTx))

    ###
    #The log likelihood consists of three terms: Log[Det[c]] and Log[Det[q]] (with coefficients) and lastly a term that looks like Log[l^a BesselK[b,Sqrt[l]]]
    #the derivative of teh third terms w.r.t. L is needed for all three of C Q and F.
    ###

    cDterm = - .5 * 1 * np.expand_dims(CI,axis = 0) #D[det] / det = inv_transpose
    qDterm = - .5 * M * np.expand_dims(QI,axis = 0)

    b = (1. - N*(M+1.)/2)
    a = b/2

    DlogtermDL = - (-2 * a + b + np.sqrt(xTx) * np.float32(mpbesselk(b-1.,np.sqrt(xTx)+EPS)/mpbesselk(b,np.sqrt(xTx)+EPS))) / (2 * xTx)

    DlogtermDL = np.reshape(DlogtermDL,[-1,1,1])

    ###
    #now I have the bits and pieces. Next I need DL/DC, DL/DQ adn DL/DF.

    DLDC = cDterm + DlogtermDL*cDxTx
    DLDQ = qDterm + DlogtermDL*qDxTx
    DLDF = DlogtermDL*fDxTx
   
    t1 = time.time()
    ctemp = chsum(DLDC,c_cho)
    if fq_shared == False:
        qtemp = chsum(DLDQ,q_cho)
    ftemp = DLDF
    t2 = time.time()

    cfin = np.array([CHtov(ch) for ch in ctemp])
    if fq_shared == False:
        qfin = np.array([CHtov(ch) for ch in qtemp])
    else:
        qfin = DLDQ
        
    t3 = time.time()

    aa = np.any(np.array([np.any(np.isnan(cfin)),np.any(np.isnan(qfin)),np.any(np.isnan(ftemp))]))

    if aa:
        print(DlogtermDL)
        print([np.any(np.isnan(cfin)),np.any(np.isnan(qfin)),np.any(np.isnan(ftemp))])
        print(np.min(xTx),np.max(xTx))

#        print("logdets:",ldc,ldq)
        exit()


    return cfin,qfin,ftemp

def f_LP_att_grad_FID(X,vc,f):

    '''
    This computes the gradient of PShared w.r.t. cov

    if fq_shared then f and q are considered to be related by the self-consistency equation S = F.S.F. + Q for S the covariance.

    we will compute F gradients, and keep Q fixed accordingly

    in particular Q = S - F.S.F -> dQ/dF ~ - S.F
    '''
    
    fval = sigmoid(f)
    
    f_mat = fval * np.eye(X.shape[2])

    vq = [x*np.sqrt(1. - fval*fval) for x in vc]
    
    tt = time.time()
    
    x = X

    c_cho = vtoCH(vc)
    c_mat = vtoS(vc)
    q_cho = vtoCH(vq)
    q_mat = vtoS(vq)

    N = int(c_mat.shape[0])
    M = int(X.shape[1]) - 1

    DC = np.linalg.det(c_mat) #DT used to be det(C)
    
    CI = np.linalg.inv(c_mat)

    DQ = np.linalg.det(q_mat)

    QI = np.linalg.inv(q_mat)
#    TIT = np.transpose(TI)

    #DdetT = np.reshape(np_DDS(cov),[1,N,N])#derivative of determinant w.r.t matric

    xTx,q_x = att_LAM(c_mat,q_mat,f_mat,x)

    #deriv. of x.C^-1.x w.r.t. C .DxTx = np_DXDX(cov,x)

    cDxTx = np_DXDX(c_mat,x[:,0,:])
    qDxTx = np_DXDX_att(q_mat,q_x)

    temp1 = np.tensordot(x[:,1:],QI,axes = [2,0])
    temp2 = np.tensordot(np.tensordot(x[:,:-1],f_mat,axes = [2,1]),QI,axes = [2,1])
        
    fDxTx = np.sum(2*x[:,:-1,None,:]*(temp2[:,:,:,None]-temp1[:,:,:,None]),axis = 1)

    #    denom = np.sqrt(DC)*np.power(np.sqrt(DQ),m)*scipy.special.kvmp.(1 - (N*(M+1.)/2),np.sqrt(xTx))

    ###
    #The log likelihood consists of three terms: Log[Det[c]] and Log[Det[q]] (with coefficients) and lastly a term that looks like Log[l^a BesselK[b,Sqrt[l]]]
    #the derivative of teh third terms w.r.t. L is needed for all three of C Q and F.
    ###

    cDterm = - .5 * 1 * np.expand_dims(CI,axis = 0) #D[det] / det = inv_transpose
    qDterm = - .5 * M * np.expand_dims(QI,axis = 0)

    b = (1. - N*(M+1.)/2)
    a = b/2

    DlogtermDL = - (-2 * a + b + np.sqrt(xTx) * np.float32(mpbesselk(b-1.,np.sqrt(xTx)+EPS)/mpbesselk(b,np.sqrt(xTx)+EPS))) / (2 * xTx)

    DlogtermDL = np.reshape(DlogtermDL,[-1,1,1])

    ###
    #now I have the bits and pieces. Next I need DL/DC, DL/DQ adn DL/DF.

    DLDC = cDterm + DlogtermDL*cDxTx
    DLDQ = qDterm + DlogtermDL*qDxTx
    DLDF = DlogtermDL*fDxTx
   
    t1 = time.time()
    
    ctemp = chsum(DLDC,c_cho)

    qtemp = chsum(DLDQ,q_cho)

    ftemp = DLDF
    t2 = time.time()

    dqcholfin = np.array([CHtov(ch) for ch in qtemp])
    
    cfin = np.array([CHtov(ch) for ch in ctemp]) + dqcholfin*(np.sqrt(1. - fval*fval))

    ffin = ((np.sum(np.diagonal(ftemp,axis1=1,axis2=2),axis = 1)) - 2 * fval * np.sum(DLDQ*np.expand_dims(c_mat,0),axis = (1,2)))*dsigmoid(f)
    
    t3 = time.time()

    return cfin,ffin

def att_PShared(I,cov,ncov,qcov,F,log = False):
    '''
    this computes the UNNORMALIZED posterior segmentation probs.
    '''

    '''
    n = len(cov)
    ic = np.linalg.inv(cov)

    def pfunc(i1,i2,a):
        chi = np.linalg.inv(ic + np.linalg.inv(ncov/a/a))
        m = np.dot(chi,np.dot(ic,i1))

        S1 = np.linalg.inv(cov + (ncov/a/a))
        S2 = np.linalg.inv(chi + qcov + (ncov/a/a))

        s,ldS1 = np.linalg.slogdet(S1)
        s,ldS2 = np.linalg.slogdet(S2)
        
        out = (1. - 2.*n)*np.log(a)
        out += -(a*a)/2
        out += -(1./2)*IP(i1/a,S1,i1/a) + .5*ldS1 - (n/2.)*np.log(2* np.pi)
        out += -(1./2)*IP((m + (i2/a)),S2,m + (i2/a)) + .5*ldS2 - (n/2.)*np.log(2* np.pi)

        return np.exp(out)

    '''

    '''
    out =  np.array([INT(lambda x:pfunc(np.reshape(I1[f],[-1]),np.reshape(I2[f],[-1]),x),0,np.inf)[0] for f in range(len(I1))])
    '''
    
    out =  np.array([INT(lambda a:att_PIA_iter(i,a,cov,ncov,qcov,F,log = False)*Pa(a),0,np.inf,limit = 100)[0] for i in I])


        
    if np.any(np.isinf(out)) or np.any(np.isnan(out)):
        print("PShared inf/nan")
        print(out)
        exit()
        
    return out

def get_att_CC_seg_weight(X1,X2,CC,CCS,CS,NCC,NCCS,NCS,QCC,QCCS,QCS,FCC,FCCS,FCS,P):

    '''
    Description: computes the posterior segmentation probabilities

    '''
    seg_x1 = get_CC_seg_x(X1,0)
    seg_x2 = get_CC_seg_x(X2,0)
    
    Pns = np.reshape(P[0]*np.prod(np.concatenate([np.reshape(att_PShared(np.transpose([seg_x1[0],seg_x2[0]],[1,0,2]),CC,NCC,QCC,FCC),[-1,1])] + [np.reshape(att_PShared(np.transpose([seg_x1[i+1],seg_x2[i+1]],[1,0,2]),CS[i],NCS[i],QCS[i],FCS[i]),[-1,1]) for i in range(4)],axis = 1),axis = 1),[X1.shape[0],1])

    Pseg = []

    for k in range(4):
        seg_x1 = get_CC_seg_x(X1,k+1)
        seg_x2 = get_CC_seg_x(X2,k+1)

        Pseg.append(np.reshape(P[k+1]*np.prod(np.concatenate([np.reshape(att_PShared(np.transpose([seg_x1[0],seg_x1[0]],[1,0,2]),CCS[k],NCCS[k],QCCS[k],FCCS[k]),[-1,1])] + [np.reshape(att_PShared(np.transpose([seg_x1[i+1],seg_x2[i+1]],[1,0,2]),CS[i],NCS[i],QCS[i],FCS[i]),[-1,1])  for i in range(4)  if i != k],axis = 1),axis = 1),[X1.shape[0],1]))

    Pseg = np.transpose(np.squeeze(np.array(Pseg)))

    PROB = np.concatenate([Pns,Pseg],1) 
    NORM = np.sum(PROB,1,keepdims = True)

    return PROB/NORM

def mat_sq(m):
    return np.dot(m,m.transpose())

if __name__ == "__main__":
    I = np.random.randn(2,10)
    a = 1.
    cov = mat_sq(np.random.randn(10,10))
    ncov = mat_sq(np.random.randn(10,10))
    qcov = mat_sq(np.random.randn(10,10))
    F = .9*np.eye(10)
    n = 0
    
    res = att_egia(n,I,a,cov,ncov,qcov,F)

    print(res)

    
