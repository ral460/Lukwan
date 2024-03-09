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
from scipy.stats import t


def aa(df):
    # Log returns
    y = df.iloc[:,0]/100
    plt.figure(figsize=(10, 6))
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
    x = np.log((y - mu)**2)
    
    #plt.figure(figsize=(12, 6))
    plt.scatter(range(len(x)),x, c = 'black', s=9)
    plt.show()
         
    plt.figure(figsize=(12, 6))
    plt.plot(x, c = 'black')
    plt.show()
    
    return x

def kalmanFilter(y ,kappa, sigma2_eta, psi):
    v = np.zeros(len(y))
    a = np.zeros(len(y))
    att = np.zeros(len(y))
    P = np.zeros(len(y))
    Ptt = np.zeros(len(y))
    F = np.zeros(len(y))
    K = np.zeros(len(y))
    
    
    P1 = sigma2_eta / (1 - (psi**2))
    P[0] = P1
    a[0] = 0
    
    # print(a[0])

    for i in range(len(y)):
        v[i] = y[i] - a[i] - kappa
        F[i] = P[i] + 4.93
        K[i] = P[i] / F[i]
        
        if i >= len(y) - 1:
            break
        
        att[i] = a[i] + (P[i]*v[i])/F[i]
        a[i+1] = psi*att[i]
        
        Ptt[i] = P[i] - (P[i]**2)/F[i]
        P[i+1] = (psi**2)*Ptt[i] + sigma2_eta
       
    #print('test',F)
    return P,a,v,F,K


def llllm(params, y):
    kappa, sigma2_eta, psi = [param for param in params]
    
    
    P,a,v,F,K = kalmanFilter(y, kappa, sigma2_eta, psi)
    
    
    ll = len(y)*0.5*np.log(2*np.pi) + 0.5*sum((np.log(F) + ((v**2))/(F)))
    return ll

def estimate(y):
    xt = y[:len(y)-1]
    xt_1 = y[1:]
    
    #psi_0 = (np.cov(xt_1,xt))[0][1]/(np.var(y) - (np.pi)**2/2
    psi_0 = 0.99
    sigma2_eta_0 = (1-psi_0**2)*(np.var(y)- ((np.pi)**2)/2)
    #sigma2_eta_0 =  0.8**2
    params = {
        "kappa": {"x0": np.mean(y), "bounds": [-100, 100]},
        "sigma2_eta": {"x0": sigma2_eta_0, "bounds": [0.001, 1000]},
        "psi": {"x0": psi_0, "bounds": [0, 1]},
    }
    
    x0 = [param["x0"] for key, param in params.items()]
    result = minimize(llllm,  x0,args=(y), tol = 1e-6, method='SLSQP', options={'maxiter': 500, 'disp': True})
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

def RVdata():
    RV = pd.read_csv('C:/Users/rafam/Downloads/EOR/Master/P4/TS/Assignment 2/realized_volatility.csv')
    print(RV)
    RV = RV[RV['Symbol'] == '.SPX']
    RV['date'] = RV['date'].str.split(' ').str[0]
    RV['date'] = pd.to_datetime(RV['date'], format='%Y-%m-%d')
    
    start_date = '2015-02-10'
    end_date = '2021-02-10'
    RV = RV[(RV['date'] >= start_date) & (RV['date'] <= end_date)]
    RV = RV.reset_index(drop=True)
    
    RV['Log Returns'] = np.log(RV['close_price'] / RV['close_price'].shift(1))

    RV = RV.drop(index=0)

    # Reset the index
    RV = RV.reset_index(drop=True)

    print(RV)

    y = RV['Log Returns'].values
    plt.figure(figsize=(10, 6))
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
    
    plt.plot(RV['close_price'])
    plt.show()
    
    return RV, y

def results(estParams, x):
    kappa, sigma2_eta, psi = [param for param in estParams]
    print('kappa is', kappa)
    print('sigma  is ', np.sqrt(np.exp(kappa + 1.27)))
    print('sigma  eta is', np.sqrt(sigma2_eta))
    print('phi is ',psi)
    P,a,v,F,K = kalmanFilter(x, kappa, sigma2_eta, psi)
    
    #plt.figure(figsize = (12,6))
    plt.plot(range(len(a)),a)
    alpha,V,r,N = kalmanSmoothing(x,P,a,v,F,K)
    
    plt.plot(range(len(alpha)),alpha)

    plt.show()
        
    plt.figure(figsize=(12, 6))
    plt.scatter(range(len(x)),x, color = 'black')
    plt.plot(kappa + alpha, c = 'red', linewidth = 5)
    plt.show()
    
    plt.plot(np.exp(alpha/2))
    plt.show()

def main():
   
    # Log returns multiplied by 100
    df = pd.read_excel('C:/Users/rafam/Downloads/EOR/Master/P4/TS/Assignment 2/sv.xlsx')
    
    # Part A
    y = aa(df)

    # Part B
    x = b(y)
    
    # Part C & D
    estParams = estimate(x)
    results(estParams, x)
        
    # Part E: revisit A-D with S&P 500 data
    RV, ySP = RVdata()
    x = b(ySP)
    estParams = estimate(x)
    results(estParams, x)
    
    # Part E: implement Beta*log(RV) into the model
    ...
    
    

    
###########################################################
### call main
if __name__ == "__main__":
    start_time = time.monotonic()
    main()
    end_time = time.monotonic()
    print("\nExecution took:",timedelta(seconds=end_time - start_time))