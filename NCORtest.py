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

    IN = np.array([[1.,.1,.1]])
    
    out = []

    for s1 in np.linspace(.1,5,10):
        for s2 in np.linspace(.1,5,10):
            CC = s1*cov(3,.1)
            NC = s2*cov(3,.1)
            
            out.append([GSM.gexp(0,IN,CC,NC,precom = False),np.linalg.det(CC),np.linalg.det(NC),np.linalg.norm(CC),np.linalg.norm(NC)])

    np.savetxt("./NCORtest.csv",out)

if __name__ == "__main__":
    run()
