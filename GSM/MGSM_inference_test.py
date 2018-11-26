import MGSM_inference as inf
import numpy as np

def test_att_grad():

    data = np.random.normal(0,1,[2,5,6])

    cchol = inf.IDchol(6)
    qchol = inf.IDchol(6)
    F = np.eye(6) + np.random.randn(6,6)/5.

    L1 = inf.att_LogLikelihood(data,cchol,qchol,F)
    for k in range(10):
        L1 = inf.att_LogLikelihood(data,cchol,qchol,F)
        dc,dq,df = inf.f_LP_att_grad(data,cchol,qchol,F)
        cchol += dc.mean(axis = 0) * .01
        qchol += dq.mean(axis = 0) * .01
        F += df.mean(axis = 0) * .01
        if k%1 == 0:
            print(L1)

    res = []

    dd = .00001

    for k in range(len(cchol)):
        L1 = inf.att_LogLikelihood(data,cchol,qchol,F)
        dc,dq,df = inf.f_LP_att_grad(data,cchol,qchol,F)

        cchol[k] += dd
        L2 = inf.att_LogLikelihood(data,cchol,qchol,F)

        res.append(((L2 - L1) - np.mean(dc,axis = 0)[k]*dd)/(L2 - L1))
    print(res)
    res = []

    for k in range(len(qchol)):
        L1 = inf.att_LogLikelihood(data,cchol,qchol,F)
        dc,dq,df = inf.f_LP_att_grad(data,cchol,qchol,F)

        qchol[k] += dd
        L2 = inf.att_LogLikelihood(data,cchol,qchol,F)

        res.append(((L2 - L1) - np.mean(dq,axis = 0)[k]*dd)/(L2 - L1))
        
    print(res)
    res = []

    for k in range(len(F)):
        for j in range(len(F[k])):
            L1 = inf.att_LogLikelihood(data,cchol,qchol,F)
            dc,dq,df = inf.f_LP_att_grad(data,cchol,qchol,F)
            
            F[k,j] += dd
            L2 = inf.att_LogLikelihood(data,cchol,qchol,F)

            res.append(((L2 - L1) - np.mean(df,axis = 0)[k,j]*dd)/(L2 - L1))

    print(res)
    
if __name__ == "__main__":
    test_att_grad()
