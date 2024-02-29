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

def a(df):
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

    
def kalmanFilter(y,P1,kappa,sigma2_eta,psi):
    z = 1.645
    v = np.zeros(len(y))
    a = np.zeros(len(y))
    P = np.zeros(len(y))
    F = np.zeros(len(y))
    K = np.zeros(len(y))
    xi = np.random.normal(0, np.sqrt(4.93), len(y))
    eta = np.random.normal(0, np.sqrt(sigma2_eta), len(y))

    P[0] = 0
    for i in range(len(y)):
        # v[i] = y[i] - a[i]
        y[i] = kappa + a[i] + xi[i]
        F[i] = P[i] + xi[i]
        K[i] = P[i]/F[i]
        
        if i >= len(y)-1:
            break 
        
        a[i+1] = psi*a[i] + eta[i]
        P[i+1] = P[i] * (1 - K[i]) + sigma2_eta
        
    LB,UB = CI(a,P,z)
    
    return P,a,v,F,K    

def CI(a,P,z):
    LB = a - (z * np.sqrt(P))
    UB = a +  (z * np.sqrt(P))
    
    return LB,UB

def estimate(y,P1):
    params = {
        "kappa": {"x0": 10, "lbub": [-1000,1000]},
        "sigma2_eta": {"x0": 100, "lbub": [-1000,1000]},
        "psi": {"x0": 10, "lbub": [-1000,1000]},
          }
    x0 = [param["x0"] for key, param in params.items()]
    bnds = [param["lbub"] for key, param in params.items()]
    result = minimize(llllm,  x0,args=(y,P1), tol = 1e-6,method='Nelder-Mead', options = {'disp': True})
    print('ll',result)
    print(result.x)

def llllm(params, y,P1):
    kappa, sigma2_eta, psi = [param for param in params]
    
    v = np.zeros(len(y))
    P = np.zeros(len(y))
    F = np.zeros(len(y))
    K = np.zeros(len(y))
    
    P[0] = P1
    #ll = 0
    
    P,a,v,F,K = kalmanFilter(y,P1,kappa,sigma2_eta,psi)
    
    ll = len(y)*0.5*np.log(2*np.pi) + 0.5*sum((np.log(F) + ((v**2))/(F)))
    
    return ll
    
def main():
    df = pd.read_excel('C:/Users/rafam/Downloads/EOR/Master/P4/TS/Assignment 2/sv.xlsx')
    # Log returns multiplied by 100
    P1 = 0
    
    y = a(df)
    x = b(y)
    estimate(x,P1)

    
###########################################################
### call main
if __name__ == "__main__":
    start_time = time.monotonic()
    main()
    end_time = time.monotonic()
    print("\nExecution took:",timedelta(seconds=end_time - start_time))