#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 15:07:45 2024

@author: lunamahn
"""

###########################################################
### Imports
import time
from datetime import timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import random
import sys
import statsmodels.api as sm
from scipy.stats import probplot
import seaborn as sns
from scipy.optimize import minimize

def aa(df):
    # Log returns
    y = df.iloc[:,0]/100
    plt.figure(figsize=(12, 6))
    plt.plot(y - np.mean(y), c = 'black')
    plt.show()
    
    mean = np.mean(y)
    variance = np.var(y)
    skewness = np.mean((y - mean) ** 3) / np.power(variance, 3/2)
    kurtosis = np.mean((y - mean) ** 4) / np.power(variance, 2) - 3
    
    print(f"Mean: {mean}")
    print(f"Variance: {variance}")
    print(f"Skewness: {skewness}")
    print(f"Kurtosis: {kurtosis}")
    
    return y
    
def b(y):
    #Apply data transformation 
    mu = np.mean(y)
    x = np.log(pow((y - mu),2))
    
    plt.figure(figsize=(12, 6))
    plt.scatter(range(len(x)),x, c = 'black', s=9)
    plt.show()
         
    plt.figure(figsize=(12, 6))
    plt.plot(x, c = 'black')
    plt.show()
    
    return x

def kalmanFilter(y ,P1,kappa, sigma2_eta, psi):
    v = np.zeros(len(y))
    a = np.zeros(len(y))
    att = np.zeros(len(y))
    P = np.zeros(len(y))
    Ptt = np.zeros(len(y))
    F = np.zeros(len(y))
    K = np.zeros(len(y))
    
    P1 = sigma2_eta / (1 - math.pow(psi, 2))
    P[0] = P1
    a[0] = random.gauss(0,np.sqrt(P[0]))
    
    for i in range(len(y)):
        v[i] = y[i] - a[i] - kappa
        F[i] = P[i] + 4.93
        K[i] = P[i] / F[i]
        
        if i >= len(y) - 1:
            break
        
        att[i] = a[i] + P[i]*v[i]/F[i]
        a[i+1] = psi*att[i]
        
        Ptt[i] = P[i] - (P[i]**2)/F[i]
        P[i+1] = (psi**2)*Ptt[i] + sigma2_eta
    
    return P,a,v,F,K


def llllm(params, y,P1):
    kappa, sigma2_eta, psi = [param for param in params]
    
    v = np.zeros(len(y))
    P = np.zeros(len(y))
    F = np.zeros(len(y))
    K = np.zeros(len(y))
    
    P[0] = P1
    
    P,a,v,F,K = kalmanFilter(y, P1, kappa, sigma2_eta, psi)
    
        
    ll = len(y)*0.5*np.log(2*np.pi) + 0.5*sum((np.log(F) + ((v**2))/(F)))
    return ll

def estimate(y,P1):
    xt = y[:len(y)-1]
    xt_1 = y[1:]
    
    #psi_0 = (np.cov(xt_1,xt))[0][1]/(np.var(y) - (np.pi)**2/2
    psi_0 = 0.95
    sigma2_eta_0 = (1-psi_0**2)*(np.var(y)- (np.pi)**2/2)
    params = {
        "kappa": {"x0": np.mean(y), "bounds": [-100, 100]},
        "sigma2_eta": {"x0": sigma2_eta_0, "bounds": [0.001, 1000]},
        "psi": {"x0": psi_0, "bounds": [0, 1]},
    }
    
    x0 = [param["x0"] for key, param in params.items()]
    result = minimize(llllm,  x0,args=(y,P1), tol = 1e-6,method='Nelder-Mead', options = {'disp': True})
    print('ll',result)
    estParams = result.x
    
    return estParams
    

def kalmanSmoothing(y,P,a,v,F,K):
    r = np.zeros(len(P)+1)
    N = np.zeros(len(P)+1)
    
    alpha = np.zeros(len(P))
    V = np.zeros(len(P))
    
    for i in reversed(range(len(P))):
        
        r[i] =  (pow(F[i],-1) * v[i]) + (1-K[i]) * r[i+1]
        N[i] =  pow(F[i],-1) + ((1-K[i])**2) * N[i+1]
        
        alpha[i] = a[i] + (P[i] * r[i])
        V[i] = P[i] - ((P[i]**2) * N[i])
    #
    return alpha,V,r[1:],N[1:]

def main():
    df = pd.read_excel('C:/Users/rafam/Downloads/EOR/Master/P4/TS/Assignment 2/sv.xlsx')
    # Log returns multiplied by 100
    
    y = aa(df)
    x = b(y)

    P1 = 1
    estParams = estimate(x, P1)
    print(estParams)
    
    kappa, sigma2_eta, psi = [param for param in estParams]
    P,a,v,F,K = kalmanFilter(x, P1, kappa, sigma2_eta, psi)
    plt.figure(figsize = (12,6))
    plt.plot(range(len(a)),a)
    
    alpha,V,r,N = kalmanSmoothing(x,P,a,v,F,K)
    plt.plot(range(len(alpha)),alpha)

    plt.show()
    
    sys.exit()
    
    plt.figure(figsize=(12, 6))
    plt.plot(x, c = 'black')
    plt.plot(alpha)
    plt.show()

    
###########################################################
### call main
if __name__ == "__main__":
    start_time = time.monotonic()
    main()
    end_time = time.monotonic()
    print("\nExecution took:",timedelta(seconds=end_time - start_time))