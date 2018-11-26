deimport GSM.MGSM_inference as inference
import numpy as np

def cov(c):
    return np.array([[1.,c],[c,1.]])

XXl = np.array([[1,0],[0,1],[1,1]])
XXh = np.array([[10,0],[0,10],[10,10]])

cc = np.linspace(-.9,.9,40)
ncov = np.array([[1.,0],[0,1.]])

gl = np.array([inference.gexp(0,XXl,cov(c),ncov) for c in cc])
gh = np.array([inference.gexp(0,XXh,cov(c),ncov) for c in cc])

gal = np.array([[inference.att_gexp(0,XXl,XXl,cov(c),ncov,q*q*cov(c)) for c in cc] for q in [0.,1.,10.]])
gah = np.array([[inference.att_gexp(0,XXh,XXh,cov(c),ncov,q*q*cov(c)) for c in cc] for q in [0.,1.,10.]])

import matplotlib as mpl
mpl.use('Agg')

import matplotlib.pyplot as plt

ext = ".jpg"

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 25
plt.rcParams['lines.linewidth'] = 2.

color1 = (1,1,1)
color2 = (.5,.5,.5)
color3 = (.25,.25,.25)

plt.plot(np.linspace(-1,1,2),[1,1],"k--")
plt.plot(cc,gl[:,2]/(gl[:,0]),"k--")
plt.plot(cc,gh[:,2]/(gh[:,0]),"r--")

plt.plot(cc,gal[0,:,2]/(gal[0,:,0]),"k")
plt.plot(cc,gah[0,:,2]/(gah[0,:,0]),"r")

plt.xlim(-1,1)
plt.ylim(0,2)
plt.xticks([-.5,0,.5])
plt.tight_layout()    
plt.savefig("./coratttest_0.jpg")
plt.clf()

plt.plot(np.linspace(-1,1,2),[1,1],"k--")
plt.plot(cc,gl[:,2]/(gl[:,0]),"k--")
plt.plot(cc,gh[:,2]/(gh[:,0]),"r--")
 
plt.plot(cc,gal[1,:,2]/(gal[1,:,0]),"k")
plt.plot(cc,gah[1,:,2]/(gah[1,:,0]),"r")

plt.xlim(-1,1)
plt.ylim(0,2)
plt.xticks([-.5,0,.5])
plt.tight_layout()    
plt.savefig("./coratttest_1.jpg")
plt.clf()

plt.plot(np.linspace(-1,1,2),[1,1],"k--")
plt.plot(cc,gl[:,2]/(gl[:,0]),"k--")
plt.plot(cc,gh[:,2]/(gh[:,0]),"r--")
 
plt.plot(cc,gal[2,:,2]/(gal[2,:,0]),"k")
plt.plot(cc,gah[2,:,2]/(gah[2,:,0]),"r")

plt.xlim(-1,1)
plt.ylim(0,2)
plt.xticks([-.5,0,.5])
plt.tight_layout()    
plt.savefig("./coratttest_2.jpg")
