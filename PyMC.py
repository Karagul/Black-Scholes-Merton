import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import exp, sqrt, log
from random import gauss, seed
import time
import math


S0 = 100.0
K = 105.0
T = 1.0
r = 0.05
sigma = 0.2


# number of time steps
M = 100

# time interval
dt = T/M

# number of iterations
I = 100000



#--------------METHOD 1------------------------------------------

t0 = time.time()

S = []
for i in range(I):
  path = []
  for t in range(M+1):
      if t == 0:
        path.append(S0)
      else:
        z = gauss(0.0,1.0)
        St = path[t-1]*exp((r-0.5*sigma*sigma)*dt + sigma*sqrt(dt)*z)
        path.append(St)   
  S.append(path)   

C0 = exp(-r*T)*sum([max(path[-1]-K,0) for path in S])/I


t1 = time.time()

time_to_compute = t1 - t0

print "European Option Value: %7.3f" % C0
print "Duration in Seconds (Method 1): %7.3f" % time_to_compute


#--------------METHOD 2------------------------------------------

t0 = time.time()

S = np.zeros((M+1,I))
S[0] = S0

for t in range(1, M+1):
  z = np.random.standard_normal(I)
  S[t] = S[t-1]*np.exp((r-0.5*sigma*sigma)*dt + sigma*math.sqrt(dt)*z)

C0 = math.exp(-r*T)*np.sum(np.maximum(S[-1]-K,0))/I

t1 = time.time()

time_to_compute = t1 - t0

print "European Option Value: %7.3f" % C0
print "Duration in Seconds (Method 2): %7.3f" % time_to_compute



#------------------------------------------------------------------

plt.plot(S[:,:10])
plt.grid(True)
plt.xlabel('time step')
plt.ylabel('index level')
plt.show()

plt.hist(S[-1],bins=50)
plt.grid(True)
plt.xlabel('index level')
plt.ylabel('frequency')
plt.show()

#------------------------------------------------------------------

