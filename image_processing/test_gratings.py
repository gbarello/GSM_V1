from PIL import Image
import numpy as np
import math
from scipy import signal

##these functions make gratings (s and c)
def GRATS(con,a,k,s,t,surr = "mean",l = .25,A = lambda c:1,offset = 1):
    
    c = float(con)

    if surr == "mean":
        sur = A(c)*offset
    else:
        sur = 0.

    v1 = np.sin(a)
    v2 = np.cos(a)
    t2 = float(t-1)/2
    if l == 0:
        sin = np.array([[A(c)*(offset + c*np.sin(2*math.pi*((v1*(float(x)-t2)) + (v2*(float(y)-t2)))/k)) if (float(x) - t2)**2 + (float(y) - t2)**2 < s**2 else sur for x in range(t)] for y in range(t)])
    else:
        mask = np.sqrt(np.array([[(float(x) - t2)**2 + (float(y) - t2)**2 for x in range(t)] for y in range(t)]))
        
        mask = 1./(1 + np.exp((mask - s)/l))

        sur = np.array([[sur for x in range(t)] for y in range(t)])
        sin = np.array([[A(c)*(offset + c*np.sin(2*math.pi*((v1*(float(x)-t2)) + (v2*(float(y)-t2)))/k)) for x in range(t)] for y in range(t)])

        sin = sin*mask + sur *(1. - mask)
        

#    sin = sin/sin.max()

    return sin

def GRATC(con,a,k,s,t,surr = "mean",l = .25,A = lambda c:1,offset = 1):
    
    c = float(con)

    if surr == "mean":
        sur = A(c)*offset
    else:
        sur = surr

    v1 = np.sin(a)
    v2 = np.cos(a)
    t2 = float(t-1)/2

    if l == 0:
        sin = np.array([[A(c)*(offset + c*np.cos(2*math.pi*((v1*(float(x)-t2)) + (v2*(float(y)-t2)))/k)) if (float(x) - t2)**2 + (float(y) - t2)**2 < s**2 else sur for x in range(t)] for y in range(t)])
    else:
        mask = np.sqrt(np.array([[(float(x) - t2)**2 + (float(y) - t2)**2 for x in range(t)] for y in range(t)]))
        
        mask = 1./(1 + np.exp((mask - s)/l))

        sur = np.array([[sur for x in range(t)] for y in range(t)])
        sin = np.array([[A(c)*(offset + c*np.cos(2*math.pi*((v1*(float(x)-t2)) + (v2*(float(y)-t2)))/k)) for x in range(t)] for y in range(t)])

        sin = sin*mask + sur *(1. - mask)
        

#    sin = sin/sin.max()

    return sin

#these function make "surround" gratings

def s_GRATS(con,a,k,s,t,surr = "mean",l = .25,A = lambda c:1,offset = 1):
    
    c = float(con)

    if surr == "mean":
        sur = A(c)*offset
    else:
        sur = surr

    v1 = np.sin(a)
    v2 = np.cos(a)

    t2 = float(t-1)/2
    if l == 0:
        sin = np.array([[A(c)*(offset + c*np.sin(2*math.pi*((v1*(float(x)-t2)) + (v2*(float(y)-t2)))/k)) if (float(x) - t2)**2 + (float(y) - t2)**2 >= s**2 else sur for x in range(t)] for y in range(t)])
    else:
        mask = np.sqrt(np.array([[(float(x) - t2)**2 + (float(y) - t2)**2 for x in range(t)] for y in range(t)]))
        
        mask = 1./(1 + np.exp((mask - s)/l))

        sur = np.array([[sur for x in range(t)] for y in range(t)])
        sin = np.array([[A(c)*(offset + c*np.sin(2*math.pi*((v1*(float(x)-t2)) + (v2*(float(y)-t2)))/k)) for x in range(t)] for y in range(t)])

        sin = sur*mask + sin*(1. - mask)
        

#    sin = sin/sin.max()

    return sin

def s_GRATC(con,a,k,s,t,surr = "mean",l = .25,A = lambda c:1,offset = 1):
    
    c = float(con)

    if surr == "mean":
        sur = A(c)*offset
    else:
        sur = surr

    v1 = np.sin(a)
    v2 = np.cos(a)
    t2 = float(t-1)/2
    if l == 0:
        sin = np.array([[A(c)*(offset + c*np.cos(2*math.pi*((v1*(float(x)-t2)) + (v2*(float(y)-t2)))/k)) if (float(x) - t2)**2 + (float(y) - t2)**2 >= s**2 else sur for x in range(t)] for y in range(t)])
    else:
        mask = np.sqrt(np.array([[(float(x) - t2)**2 + (float(y) - t2)**2 for x in range(t)] for y in range(t)]))
        
        mask = 1./(1 + np.exp((mask - s)/l))

        sur = np.array([[sur for x in range(t)] for y in range(t)])
        sin = np.array([[A(c)*(offset + c*np.cos(2*math.pi*((v1*(float(x)-t2)) + (v2*(float(y)-t2)))/k)) for x in range(t)] for y in range(t)])

        sin = sur*mask + sin *(1. - mask)        

#    sin = sin/sin.max()

    return sin

##
if __name__ == "__main__":
    for l in [0,.1,.25,.5,1]:
        for A in [0,1]:
            print(l)
            test = GRATC(.5,0,8/(2*np.pi),7,24,surr = "mean",l = l) + A*s_GRATC(.5,0,8/(2*np.pi),7,24,surr = "mean",l = l,offset = 0)
            np.savetxt("./test_grating_{}_{}.csv".format(l,A),test)
