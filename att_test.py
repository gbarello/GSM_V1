import numpy as np
import GSM.MGSM_inference as MGSM

cov = np.array([[1,.1],[.1,1]])
ncov = np.array([[1,0],[0,1]])
qcov = np.array([[.01,0],[0,.01]])

print("cov",cov)
print("ncov",ncov)
print("qcov",qcov)

ff = np.array([[1,0],[1,1]])

res = MGSM.att_gexp(0,ff,ff,cov,ncov,qcov)
print(res)
