import numpy as np
import json

from matplotlib import pyplot as plt
'''
from IF1 import InterleavedFilter
from RMED import RMED
from BTM import BeatTheMean
from doubler import Doubler
from savage import SAVAGE
from rucb import run_rucb
from rcs import run_rcs '''

from borda_exp3 import BordaEXP3, BordaEXP3Con, BordaTS

# dataset = "5_real"
# best_arm = 0

dataset = "5_art"
best_arm = 0

#dataset = "10_real"
#best_arm = 7

#dataset = "10_art"
#best_arm = 0

#dataset = "cars10"
#best_arm = 2


with open("./datasets/" + dataset + ".npy", 'rb') as f:
    pref_mat = np.load(f)    

consumption = np.array([0.9,0.9,0.1,0.8,0.8,0.8]) #synthetic5
#consumption = np.array([0.1,0.2,0.3,0.4,0.5,0.6])
consumption = np.array([0.6,0.5,0.4,0.3,0.2,0.1])
consumption = np.array([0,0,0,0,0,0])

#consumption = np.array([0.9,0.9,0.9,0.8,0.01,0.1,0.2,0.1,0.7,0.8]) #synthetic10

consumption = np.array([0.9,0.9,0.8,0.9,0.02,0.01,0.1,0.8,0.7,0.8]) #real
#consumption = np.array([0.7,0.9,0.9,0.8,0.6,0.1,0.4,0.3,0.5,0.2]) #2 3 0 4 8 6 7 1 9 5
#consumption = np.array([0,0,0,0,0,0,0,0,0,0])
def generator_fnc(i, j):
    return np.random.binomial(n=1, p=pref_mat[int(i)][int(j)], size=1)[0]
    
def consumption_gen(i, j):


    v1 = min(max(consumption[i] + np.random.normal(0, 0.05, 1)[0],0),1)
    v2 = min(max(consumption[j] + np.random.normal(0, 0.05, 1)[0],0),1)

    return v1, v2

    return max(min((consumption[i] + np.random.normal(0, 1, 1)),1),0) , max(min((consumption[j] + np.random.normal(0, 1, 1)),1),0)


def regret_fn(i, j):
    return pref_mat[best_arm][int(i)] + pref_mat[best_arm][int(j)] - 1

def rew_fn(i,j):
    return pref_mat[int(i)][best_arm] + pref_mat[int(j)][best_arm]


def gen():

    rew_all = []
    samples =50
    B = 1000 # 2000 #1000
    opt = 2000# 4000 #2000
    Z = opt/B
    horizon = 2000
    for j in range(samples):
        
        x = BordaEXP3Con(len(pref_mat), horizon, 1, B,opt,Z,  generator_fnc, consumption_gen, rew_fn)
        
        #y = BordaEXP3(len(pref_mat), horizon, 1, B,opt,Z,  generator_fnc, consumption_gen, rew_fn)
        
        cnsmp = 0
        rew = []
        for i in range(horizon):
        
        
            if cnsmp > B:
                rew.append(rew[-1])
        
            else:
                rew, cnsmp = x.algo()
        
            
        print(rew[-1])
            
        rew_all.append(rew)
    np.save('./results/bordaexp3_2_cnsmp_{}.npy'.format(dataset), rew_all)
     
     
    
    rew_all = []
    
    for j in range(samples):
        
        x = BordaEXP3(len(pref_mat), horizon, 1, B,opt,Z,  generator_fnc, consumption_gen, rew_fn)
        
        #y = BordaEXP3(len(pref_mat), horizon, 1, B,opt,Z,  generator_fnc, consumption_gen, rew_fn)
        
        cnsmp = 0
        rew = []
        for i in range(horizon):
        
        
            if cnsmp > B:
                rew.append(rew[-1])
        
            else:
                rew, cnsmp = x.algo()
        
            
        print(rew[-1])
            
        rew_all.append(rew)
    np.save('./results/bordaexp3_2_{}.npy'.format(dataset), rew_all)
    
    rew_all = []
    
    for j in range(samples):
        
        x = BordaTS(len(pref_mat), horizon, 1, B,opt,Z,  generator_fnc, consumption_gen, rew_fn)
        
        #y = BordaEXP3(len(pref_mat), horizon, 1, B,opt,Z,  generator_fnc, consumption_gen, rew_fn)
        
        cnsmp = 0
        rew = []
        for i in range(horizon):
        
        
            if cnsmp > B:
                rew.append(rew[-1])
        
            else:
                rew, cnsmp = x.algo()
        
            
        print(rew[-1])
            
        rew_all.append(rew)
    np.save('./results/bordaTS_2_{}.npy'.format(dataset), rew_all) 
          
    
    
    
    
print(pref_mat)
print(np.sum(pref_mat,axis=1)/5)
gen()
#print(pref_mat)
# plot()


    
'''
    #for i in range(samples):
    for i in range(horizon):

        #print("On sample number", i)

        if cnsmp > B:
            rew.append(rew[-1])
        
        else:
            rew, cnsmp = x.algo()
        
        #print(rew,cnsmp)
        
            #break
            
    print(rew[-1])
    
    
    
            
    np.save('./results/bordaexp3_cnsmp_{}.npy'.format(dataset), rew)
    
    rew = []
    cnsmp = 0
    
    for i in range(horizon):

        #print("On sample number", i)
        
        
        if cnsmp > B:
            rew.append(rew[-1])
        
        else:
            rew, cnsmp = y.algo()
        
        #rew, cnsmp = y.algo()
        
        
    print(rew[-1])
            
    np.save('./results/bordaexp3_{}.npy'.format(dataset), rew)'''
