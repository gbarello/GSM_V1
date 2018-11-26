import numpy as np
import GSM.MGSM_inference as inference

def cov(c,a = 1.):
    return a*np.array([[1.,c],[c,1.]])

def S1(t,MASK,TARGET):

    out = np.array([0.,0.])

    if np.mod(t,.175) < .125 and TARGET:
        out += np.array([1.,.5])
    elif np.mod(t,.175) > .125 and MASK:
        out += np.array([.5,1.])

    return out

def S2(t,MASK,TARGET):

    out = np.array([0.,0.])

    if t > .250:
        return out
    
    elif np.mod(t,.175) < .125 and TARGET:
        out += np.array([1.,.5])
    elif np.mod(t,.175) > .125 and MASK:
        out += np.array([.5,1.])

    return out

def S3(t,MASK,TARGET):

    out = np.array([0.,0.])

    if t > .250:
        return out    
    elif t < 1.75 and TARGET:
        out += np.array([1.,.5])
    elif t > .125  and MASK:
        out += np.array([.5,1.])

    return out

def stimfunc(STIM,t,nt,MASK = True,TARGET = True):
    if STIM == 1:
        return np.array([S1(T,MASK,TARGET) for T in np.linspace(0,t,nt)])
    elif STIM == 2:
        return np.array([S2(T,MASK,TARGET) for T in np.linspace(0,t,nt)])
    elif STIM == 3:
        return np.array([S3(T,MASK,TARGET) for T in np.linspace(0,t,nt)])

nta = 80
tot = .4
ta = tot/(nta)
Fa = np.exp(-ta)*np.identity(2)    

stima = np.array([10*stimfunc(s,.4,nta + 1,MASK = M[1],TARGET = M[0])[:k] for k in range(1,81,5) for s in [1,2,3] for M in [[True,False],[False,True],[True,True]]])
stimp = np.array([10*stimfunc(s,.4,nta + 1,MASK = M[1],TARGET = M[0]) for s in [1,2,3] for M in [[True,False],[False,True],[True,True]]])

cor = [.1]

resp = np.array([inference.att_gexp(0,np.array([s]),cov(c),cov(0),inference.Q_self_con(cov(c),Fa),Fa) for c in cor for s in stima])

np.savetxt("./att_2d_resp.csv",resp)
np.savetxt("./att_2d_stim.csv",np.reshape(stimp,[-1,2]))
