from __future__ import print_function

import numpy as np
import pickle,gzip
from scipy import stats
import matplotlib.pyplot as plt

numParticipants = 40

def interests():
    global numParticipants
    m = numParticipants
    # 11 papers accepted in fatrec-2017
    n = 11 
    cites = [4,3,14,3,1,11,1,18,10,16,15]
    V = np.zeros((m,n))
    max_cite = max(cites) * 1.0
    for talk in range(n):
        # truncated_norm: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.truncnorm.html
        my_mean = cites[talk] / max_cite
        my_std = my_mean / 4.0
        myclip_a = 0
        myclip_b = 1
        V[:,talk] = stats.truncnorm.rvs((myclip_a-my_mean)/my_std, (myclip_b-my_mean)/my_std,
                                      loc=my_mean, scale=my_std, size = m)

    np.savetxt("data/fatrecContInterest_V.csv", V, delimiter=",") 
    print ("data/fatrecContInterest_V.csv Saved!")  

interests()