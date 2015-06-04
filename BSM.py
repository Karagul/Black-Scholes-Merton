# Monte Carlo evaluation of European Call Option using Black-Scholes-Merton model

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


S0 = 100.0 # initial index level
K = 105.0 # strike price
T = 1.0 # time to maturity
r = 0.05 # riskless short rate
sigma = 0.2 # volatility

I = 100000 # number of simulations

z = np.random.randn(I)

# Values at maturity
ST = S0*np.exp((r - 0.5*sigma*sigma)*T + sigma*np.sqrt(T)*z)
hT = np.maximum(ST-K,0.0)

# Monte Carlo estimator
C0 = np.exp(-r*T)*np.sum(hT)/I

print "Value of the European Call Option %5.3f" % C0



