import numpy as np
from svd_tmv import computeLDLInverse as chol
import time
import pylab as plt

### linalg.inv vs tmv chol
np.random.seed(1)

N = np.linspace(2,1000,100).astype(int)

time_numpy = np.zeros(100)
time_tmv = np.zeros(100)

for i in range(100):
    print i
    
    m = np.random.rand(N[i],N[i])
    m = m + m.T  
    m += N[i]*np.identity(N[i])

    A = []
    for j in range(10):
        t0 = time.time()
        m_inv = np.linalg.inv(m)
        t1 = time.time()
        A.append(t1-t0)
        
    time_numpy[i] = np.mean(A)
    
    A = []
    for j in range(10):
        t0 = time.time()
        m_inv = chol(m)
        t1 = time.time()
        A.append(t1-t0)
        
    time_tmv[i] = np.mean(A)

plt.plot(N,time_numpy,'k',linewidth=3)
plt.plot(N,time_tmv,'b',linewidth=3)
plt.show()
