from MGSM_inference import *

def nc_att_egia(n,I,a,cov,ncov,qcov,ucov,F,G,getall = False,getP = False):
    #we're basically going to perform 2 steps of kalman update equations starting with the prior
    #I can be a list of images, and I will write it recursively to handle any number

    g = np.zeros(len(cov) + len(ncov))
    p = scipy.linalg.block_diag(cov,ncov)
    
    out = [g]
    pout = [p]
    
    QQ = scipy.linalg.block_diag(qcov,ucov)
    FF = scipy.linalg.block_diag(F,G)
    H = np.concatenate([a*np.eye(len(cov)),np.zeros_like(ncov)],axis = 1)
    R = np.zeros([I.shape[-1],I.shape[-1]])
    
    for i in I:
        g,p = nc_att_egia_update(a,i,g,p,QQ,FF,H,R)
        out.append(g)
        pout.append(p)
        p = dot([FF,p,FF.transpose()]) + QQ
        g = dot([FF,g])

        
    if getP:
        return np.array(out),np.array(pout)
    
    if getall:
        return np.array(out)
    
    return out[-1][n]

def nc_att_egia_update(a,I,g,p,Q,F,H,R):
    meas_err = np.linalg.inv(dot([H,p,H.transpose()]) + R)
    
    K = dot([p,H.transpose(),meas_err])

    go = g + np.dot(K,I - dot([H,g]))
    
    P = np.dot(np.identity(len(go)) - dot([K,H]),p)

    return go,P

def get_new_mc_cornoise(xi,mu,II,SQGUFi,SU,FmGi,B,sigma):

    A = sigma + xi
    a = dot([SQGUFi,(mu + dot([SU,II]))])
    b = dot([FmGi,II])

    xio = np.linalg.inv(np.linalg.inv(A) + np.linalg.inv(B))
    muo = dot([xio,dot([A,b]) + dot([B,a])])

    x = (a - b)
    S = A + B

    _,Slogdet = np.linalg.slogdet(S)

    logterm = (-len(II) * np.log(2 * np.pi) - Slogdet - dot([x,np.linalg.inv(S),x]))/2

    return logterm,xio,muo    

def nc_att_PIA_iter(I,a,cov,ncov,qcov,ucov,F,G):
    n = len(cov)
    m = len(I)

    #repeatedly used, independent of I
    FmGi = np.linalg.inv(F-G)
    QpU = (a*a*qcov + ucov)
    QGUF = dot([a*a*qcov,G]) + dot([ucov,F])
    sigma = np.linalg.inv(np.linalg.inv(a*a*qcov) + np.linalg.inv(ucov))
    SQGUFi = np.linalg.inv(dot([sigma,QGUF]))
    B = dot([FmGi,QpU,FmGi.transpose()])
    SU = dot([sigma,ucov])

    _,ldSQGUFi = np.linalg.slogdet(SQGUFi)

    _,ldcov = np.linalg.slogdet(a*a*cov + ncov)
    
    l2p = m * n * np.log(2*np.pi)
    
    #also repeatedly used
    _,ldFmGi = np.linalg.slogdet(FmGi)
    _,ldQpU = np.linalg.slogdet(QpU)

    ldcoef = ldSQGUFi + ldFmGi

    #get teh diff of subsequent inputs
    ImI = np.tensordot(I[:-1],F,axes = [[1],[1]]) - I[1:]


    #this is actually the constribution from the first image (same even when many frames are present)
    result = (- l2p - ldcov - dot([I[0],np.linalg.inv(a*a*cov + ncov),I[0]]))/2.

    #if there is just one frame, then this is the result:
    if len(I) == 1:
        return result
    
    #otherwise, go for it
    xt = B
    mt = dot([FmGi,ImI[-1]])

    result += ldFmGi# * 2 #I think this is just suppposed to be a single det[F-G]

    for i in reversed(ImI[:-1]):
        logterm,xt,mt = get_new_mc_cornoise(xt,mt,i,SQGUFi,SU,FmGi,B,sigma)

        result += logterm + ldcoef# * 2#same with this factor of 2

    fcov = np.linalg.inv(np.linalg.inv(a*a*cov) + np.linalg.inv(ncov)) + xt # this changed from inv(a^-1 + b^-1 + c^-1)
    
    _,ldxcov = np.linalg.slogdet(fcov)
    
    fI = mt - dot([(np.linalg.inv(a*a*cov) + np.linalg.inv(ncov)),ncov,I[0]])
    
    result += (- l2p - ldxcov - dot([fI,fcov,fI])) / 2

    return result

def nc_att_integrand(n,I,a,cov,ncov,qcov,ucov,F,G):

    pia = nc_att_PIA_iter(I,a,cov,ncov,qcov,ucov,F,G)
    gia = nc_att_egia(n,I,a,cov,ncov,qcov,ucov,F,G)
    
    return pia*gia*Pa(a)

def nc_att_norm(I,a,cov,ncov,qcov,ucov,F,G):

    pia = nc_att_PIA_iter(I,a,cov,ncov,qcov,ucov,F,G)
    
    return pia*Pa(a)


def nc_att_gexp(n,x,cov,ncov,qcov,ucov,F,G):
    
    TOP = [INT(lambda a: nc_att_integrand(n,x[k],a,cov,ncov,qcov,ucov,F,G),0,np.inf)[0] for k in range(len(x))]
    
    NORM = [INT(lambda a: nc_att_norm(x[k],a,cov,ncov,qcov,ucov,F,G),0,np.inf)[0] for k in range(len(x))]
    
    OUT = [TOP[k]/NORM[k] if NORM[k] != 0 else 0 for k in range(len(TOP))]
    
    return np.array(OUT)

def nc_att_pexp(n,x,cov,ncov,qcov,ucov,F,G):
    
    TOP = [INT(lambda a: nc_att_P_integrand(n,x[k],a,cov,ncov,qcov,ucov,F,G),0,np.inf)[0] for k in range(len(x))]
    
    NORM = [INT(lambda a: nc_att_norm(x[k],a,cov,ncov,qcov,ucov,F,G),0,np.inf)[0] for k in range(len(x))]
    
    OUT = [TOP[k]/NORM[k] if NORM[k] != 0 else 0 for k in range(len(TOP))]
    
    return np.array(OUT)

def general_nc_att_gexp(x,cov,ncov,qcov,ucov,F,G,ind):
    
    TOP = [np.array([INT(lambda a: nc_att_integrand(j,x[k],a,cov,ncov,qcov,F),0,np.inf)[0] for j in ind]) for k in range(len(x))]
    
    NORM = [INT(lambda a: nc_att_norm(x[k],a,cov,ncov,qcov,F),0,np.inf)[0] for k in range(len(x))]
    
    OUT = [TOP[k]/NORM[k] if NORM[k] != 0 else np.zeros_like(TOP[k]) for k in range(len(TOP))]

    return np.array(OUT)

def general_nc_att_pexp(x,cov,ncov,qcov,F,ind):
    
    TOP = [np.array([INT(lambda a: nc_att_P_integrand(j,x[k],a,cov,ncov,qcov,F),0,np.inf)[0] for j in ind]) for k in range(len(x))]
    
    NORM = [INT(lambda a: nc_att_norm(x[k],a,cov,ncov,qcov,F),0,np.inf)[0] for k in range(len(x))]
    
    OUT = [TOP[k]/NORM[k] if NORM[k] != 0 else np.zeros_like(TOP[k]) for k in range(len(TOP))]

    return np.array(OUT)

def nc_att_gexp_full(x,cov,ncov,qcov,F):
    return general_nc_att_gexp(x,cov,ncov,qcov,F,range(x.shape[2]))

def nc_att_pexp_full(x,cov,ncov,qcov,F):
    return general_nc_att_pexp(x,cov,ncov,qcov,F,range(x.shape[2]))

def get_MGSM_nc_att_weights(F,segs,C,NC,QC,FC,P):
    #(I,cov,ncov,qcov,F)
    gsm_resp = np.array([np.log(P[k]) + np.sum([np.log(nc_att_PShared(F[:,:,segs[k][j]],C[k][j],NC[k][j],QC[k][j],FC[k][j])) for j in range(len(segs[k]))],axis = 0) for k in range(len(segs))]) # shape [nseg,ndata]

    b = np.amax(gsm_resp,axis = 0,keepdims = True)
    
    out = np.transpose(np.squeeze(np.exp(gsm_resp-b)/np.sum(np.exp(gsm_resp-b),axis = 0,keepdims = True)))

    if np.any(np.isnan(out)):
        print("wtf MGSM_att_weights")
        print(gsm_resp)
        exit()

    return out


def find_GSM_nc_pia_max(I,C,NC,QC,FC,low,high,start,eps):
    out = find_f_max(lambda a:Pa(a,log = True) + nc_att_PIA_iter(I,a,C,NC,QC,FC,log = True),low=low,high=high,start=start,eps = eps)
    return out
    
def stable_get_MGSM_nc_att_weights(F,segs,C,NC,QC,FC,P,ind,rev_ind,ifrac = 100,npnt = 1000,op = True,calc_p = False):
    #(I,cov,ncov,qcov,F)

    if op == True:
        PIA_max = np.array([[[find_GSM_nc_pia_max(F[i][:,segs[k][j]],C[k][j],NC[k][j],QC[k][j],FC[k][j],low=0,high=np.inf,start=10.,eps = 1.) for i in range(F.shape[0])] for j in range(len(segs[k]))] for k in range(len(segs))]) # shape [nseg,ndata]
        
        logints = np.array([[[logint.integrate_log(lambda a:Pa(a,log = True) + nc_att_PIA_iter(F[i][:,segs[k][j]],a,C[k][j],NC[k][j],QC[k][j],FC[k][j],log = True),PIA_max[k][j][i][0]/ifrac,ifrac*PIA_max[k][j][i][0],[PIA_max[k][j][i][0]]) for i in range(len(PIA_max[k][j]))] for j in range(len(PIA_max[k]))] for k in range(len(PIA_max))])
    else:
        logints = np.array([[[logint.integrate_log(lambda a:Pa(a,log = True) + nc_att_PIA_iter(F[i][:,segs[k][j]],a,C[k][j],NC[k][j],QC[k][j],FC[k][j],log = True),.0001,5000,[]) for i in range(F.shape[0])] for j in range(len(segs[k]))] for k in range(len(segs))])
        
    if calc_p:
        GNlogints = np.array([np.concatenate([np.array([expectation_log(lambda a:nc_att_egia(ind[k][j],F[i][:,segs[k][j]],a,C[k][j],NC[k][j],QC[k][j],FC[k][j],getP = True)[1][-1],lambda a: Pa(a,log = True) + nc_att_PIA_iter(F[i][:,segs[k][j]],a,C[k][j],NC[k][j],QC[k][j],FC[k][j],log = True),logints[k][j][i][1]["points"],n_d_exp = 2) if len(ind[k][j])>0 else [] for i in range(F.shape[0])]) for j in range(len(segs[k]))],axis = 1) for k in range(len(segs))])
        
    else:
        GNlogints = np.array([np.concatenate([np.array([expectation_log(lambda a:nc_att_egia(ind[k][j],F[i][:,segs[k][j]],a,C[k][j],NC[k][j],QC[k][j],FC[k][j]),lambda a: Pa(a,log = True) + nc_att_PIA_iter(F[i][:,segs[k][j]],a,C[k][j],NC[k][j],QC[k][j],FC[k][j],log = True),logints[k][j][i][1]["points"]) if len(ind[k][j])>0 else [] for i in range(F.shape[0])]) for j in range(len(segs[k]))],axis = 1)[:,rev_ind[k]] for k in range(len(segs))])
    
    mgsm_resp = np.array([np.log(P[k]) + np.array([np.sum([logints[k][j][i][0] for j in range(len(segs[k]))]) for i in range(F.shape[0])]) for k in range(len(segs))]) # shape [nseg,ndata]

    b = np.amax(mgsm_resp,axis = 0,keepdims = True)
    
    out = np.squeeze(np.exp(mgsm_resp-b)/np.sum(np.exp(mgsm_resp-b),axis = 0,keepdims = True))

    if np.any(np.isnan(out)):
        print("wtf MGSM_att_weights")
        print(gsm_resp)
        exit()


    return out,GNlogints

def general_MGSM_g_nc_att(F,segs,C,NC,QC,FC,P,ind,ifrac = 1.5,npnt = 1000,stable = True,op=True):

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
        probs,gsm_resp = stable_get_MGSM_nc_att_weights(F,segs,C,NC,QC,FC,P,seg_index,seg_to_ind,ifrac=ifrac,npnt=npnt,op=op,calc_p=False)#[nseg,ndata]
    else:
        probs = np.transpose(get_MGSM_nc_att_weights(F,segs,C,NC,QC,FC,P))#[nseg,ndata]

    if np.any(np.isnan(probs)):
        print("probs")
        exit()
        
    #att_gexp(n,x,cov,ncov,qcov,F)
    if stable == False:
        gsm_resp = np.array([np.concatenate([general_nc_att_gexp(F[:,:,segs[k][j]],C[k][j],NC[k][j],QC[k][j],FC[k][j],seg_index[k][j]) for j in range(len(segs[k]))],axis = 1)[:,seg_to_ind[k]] for k in range(len(segs))]) # shape [nseg,ndata,nfilt]

    if np.any(np.isnan(gsm_resp)) or np.any(np.isinf(gsm_resp)):
        print("gsm_resp")
        exit()

        
    result = np.sum(np.expand_dims(probs,-1) * gsm_resp,axis = 0)

    return result

def general_MGSM_p_nc_att(F,segs,C,NC,QC,FC,P,ind,ifrac = 1.5,npnt = 1000,stable = True,op=True):

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
        probs,gsm_resp = stable_get_MGSM_nc_att_weights(F,segs,C,NC,QC,FC,P,seg_index,seg_to_ind,ifrac=ifrac,npnt=npnt,op=op,calc_p = True)#[nseg,ndata]
    else:
        probs = np.transpose(get_MGSM_nc_att_weights(F,segs,C,NC,QC,FC,P))#[nseg,ndata]

    if np.any(np.isnan(probs)):
        print("probs")
        exit()
        
    #att_gexp(n,x,cov,ncov,qcov,F)
    if stable == False:
        gsm_resp = np.array([np.concatenate([general_nc_att_pexp(F[:,:,segs[k][j]],C[k][j],NC[k][j],QC[k][j],FC[k][j],seg_index[k][j]) for j in range(len(segs[k]))],axis = 1)[:,seg_to_ind[k]] for k in range(len(segs))]) # shape [nseg,ndata,nfilt]

    if np.any(np.isnan(gsm_resp)) or np.any(np.isinf(gsm_resp)):
        print("gsm_resp")
        exit()

        
    result = np.sum(np.expand_dims(np.expand_dims(probs,-1),-1) * gsm_resp,axis = 0)

    return result


def CC_MGSM_nc_att_g(F1,F2,CC,CCS,CS,NCC,NCCS,NCS,QCC,QCCS,QCS,FCC,FCCS,FCS,P):
    
    '''
    Description: computes the expected g of the center filter set in the noisy MGSM
    '''

    #first we need to get the assignment probabilities:
    #[[],..,[]]

    probs = np.transpose(get_nc_att_CC_seg_weight(F1,F2,CC,CCS,CS,NCC,NCCS,NCS,QCC,QCCS,QCS,FCC,FCCS,FCS,P))
    
    gv = np.array([[nc_att_gexp(n,np.transpose([get_CC_seg_x(F1,i)[0],get_CC_seg_x(F2,i)[0]],[1,0,2]),CC if i == 0 else CCS[i-1],NCC if i == 0 else NCCS[i-1],QCC if i == 0 else QCCS[i-1],FCC if i == 0 else FCCS[i-1]) for n in range(8)] for i in range(len(P))])

    return np.transpose((np.reshape(probs,(len(P),1,-1))*gv).sum(axis = 0),(1,0))
    
def nc_att_PShared_nonoise(d,segs,C,QC,FF,P,fq_shared = False,f_ID = False):

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
        
    ll = np.array([np.sum(np.array([nc_att_LogLikelihood(d[:,:,segs[s][c]],C[s][c],QC[s][c],FC[s][c],mean = False,fq_shared = fq_shared) for c in range(len(segs[s]))]),axis = 0) + np.log(P[s]) for s in range(len(segs))])
    
    #ll -> [len(segs),ndat]

    b = np.amax(ll,0,keepdims=True)
    
    return np.transpose(np.exp(ll-b)/np.sum(np.exp(ll-b),axis = 0,keepdims = True))
    
def nc_att_MGSM_loglik(d,segs,C,QC,F,P,fq_shared = False,f_ID = False):

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
        
    ll = np.array([np.sum(np.array([nc_att_LogLikelihood(d[:,:,segs[s][c]],C[s][c],QC[s][c],FC[s][c],mean = False,fq_shared = fq_shared) for c in range(len(segs[s]))]),axis = 0) + np.log(P[s]) for s in range(len(segs))])

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
    
    out =  np.array([INT(lambda a:att_PIA_iter(i,a,cov,ncov,qcov,F)*Pa(a),0,np.inf,limit = 100)[0] for i in I])


        
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


if __name__ == "__main__":

    import ATT_GSM_inference as att
    np.random.seed(0)

    def mat_sq(m):
        return np.dot(m,m.transpose())
    
    I = np.random.randn(2,1,10)
    I = np.ones([2,2,10])
    
    a = 2.
    cov = mat_sq(np.random.randn(10,10))
    ncov = mat_sq(np.random.randn(10,10))

    F = .9*np.eye(10)
    G = 0*np.eye(10)

    qcov = Q_self_con(cov,F)#mat_sq(np.random.randn(10,10))
    ucov = Q_self_con(ncov,G)#)ncov#.1*mat_sq(np.random.randn(10,10))

    print("noise cov:",np.sum((ucov - ncov)**2))

    n = 0
    
    print("Testing")
    
    g1 = nc_att_egia(n,I[0],a,cov,ncov,qcov,ucov,F,G,getall = True)
    g2 = att.att_egia(n,I[0],a,cov,ncov,qcov,F,getall = True)
    
    print(((g1[:,:10] - g2)**2).sum())
    #exit()
    
    print("Testing PIA")
    
    p1 = nc_att_PIA_iter(I[0],a,cov,ncov,qcov,ucov,F,G)
    p2 = att.att_PIA_iter(I[0],a,cov,ncov,qcov,F,log = True)
    p3 = att.att_PIA_iter_OLD(I[0],a,cov,ncov,qcov,F,log = True)
    print(p1)
    print(p2)
    print(p3)
    exit()

   
