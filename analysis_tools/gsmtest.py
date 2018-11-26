import GSM.MGSM_train as fit
import GSM.MGSM_inference as MGSM

import numpy as np
n = 100000
m = 8

a = np.random.rayleigh(1,[n,1])

#S = np.random.rand(m,m)
#S = np.dot(S,S.transpose())

S = np.array([[np.exp(1 + np.cos(x - y)) for y in np.linspace(0,2*np.pi - (2.*np.pi/m),m)] for x in np.linspace(0,2*np.pi - (2.*np.pi/m),m)])
print(S.shape)


b = np.random.multivariate_normal(np.zeros(m),S,n)

dd = a*b

chtest = MGSM.CHtov(np.fliplr(np.flipud(np.transpose(np.linalg.cholesky(np.cov(np.transpose(dd))/2)))))

print(MGSM.log_likelihood_center(dd,S))
out,test = fit.fit_center(dd,GRAD=True,init = chtest)
print("TRUE",MGSM.log_likelihood_center(dd,S))

print(np.mean(np.abs(S)))
print(np.mean(np.abs(S-out)))
#for k in range(len(test)):
#    print(np.mean(np.abs(S-test[k])))

np.savetxt("./truecov.csv",S)
np.savetxt("./fitcov.csv",out)
