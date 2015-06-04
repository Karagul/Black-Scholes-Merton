# European call option - Black-Scholes-Merton

import numpy as np
import pandas as pd
from math import log, sqrt, exp
from scipy import stats
import matplotlib.pyplot as plt



def call_value(S0,K,T,r,sigma):
  S0 = float(S0) 
  d1 = (log(S0/K) + (r + sigma*sigma/2)*T)/(sigma*sqrt(T))
  d2 = (log(S0/K) + (r - sigma*sigma/2)*T)/(sigma*sqrt(T))
  Nd1 = stats.norm.cdf(d1,0.0,1.0)
  Nd2 = stats.norm.cdf(d2,0.0,1.0)   
  C = S0*Nd1 - exp(-r*T)*K*Nd2
  return C


def vega(S0,K,T,r,sigma):
  S0 = float(S0)
  d1 = (log(S0/K) + (r + sigma*sigma/2)*T)/(sigma*sqrt(T))
  Nd1 = stats.norm.cdf(d1,0.0,1.0)
  v = S0*Nd1*sqrt(T)
  return v


def imp_vol(S0,K,T,r,C0,sigma_est,it=100):
  S0 = float(S0)
  for i in range(it):
    c1 = call_value(S0,K,T,r,sigma_est)
    v = vega(S0,K,T,r,sigma_est) 
    sigma_est -= (c1-C0)/v
  return sigma_est



V0 = 17.6639
r = 0.01
h5 = pd.HDFStore('./vstoxx_data_31032014.h5','r')
futures_data = h5['futures_data']
options_data = h5['options_data']
h5.close()


print futures_data
print options_data.info()
print options_data[['DATE','MATURITY','TTM','STRIKE','PRICE']].head()



# implied volatility

options_data['IMP_VOL'] = 0.0
tol = 0.5

for option in options_data.index:
  forward = futures_data[futures_data['MATURITY'] == options_data.loc[option]['MATURITY']]['PRICE'].values[0]
  if(forward*(1-tol) < options_data.loc[option]['STRIKE'] < forward*(1+tol)):
    K = options_data.loc[option]['STRIKE']
    T = options_data.loc[option]['TTM']
    C0 = options_data.loc[option]['PRICE']
    iv = imp_vol(V0,K,T,r,C0,sigma_est=2.0,it=100)
    options_data['IMP_VOL'].loc[option] = iv

print futures_data['MATURITY']
print options_data.loc[46170]
print options_data.loc[46170]['STRIKE']

plot_data = options_data[options_data['IMP_VOL'] > 0]
maturities = sorted(set(options_data['MATURITY']))


# output results

plt.figure(figsize=(8,6))
for maturity in maturities:
  data = plot_data[options_data.MATURITY == maturity]
  plt.plot(data['STRIKE'],data['IMP_VOL'],label=maturity.date(),lw=1.5)
  plt.plot(data['STRIKE'],data['IMP_VOL'],'r.')
  plt.grid(True)
  plt.xlabel('strike')
  plt.ylabel('implied volatility')
  plt.legend()
  plt.show()


group_data = plot_data.groupby(['MATURITY','STRIKE'])['PRICE','IMP_VOL']
print group_data

group_data = group_data.sum()
print group_data.head()
print group_data.index.levels


  

