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
from scipy.stats import pearsonr
from scipy.stats import ttest_ind


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
    

def b(y, unique_years, RV):
    #Apply data transformation 
    mu = np.mean(y)
    x = np.log((y - mu)**2)
    
    plt.figure(figsize=(12, 6))
    if len(unique_years) > 0:
        plt.scatter(RV['date'], x, c = 'black', s=9)
        plt.xticks([pd.Timestamp(f'{year}-01-01') for year in unique_years], unique_years)
    else:
        plt.scatter(range(len(x)),x, c = 'black', s=9)
    plt.show()
         
    plt.figure(figsize=(12, 6))
    if len(unique_years) > 0:
        plt.plot(RV['date'], x, c = 'black')
        plt.xticks([pd.Timestamp(f'{year}-01-01') for year in unique_years], unique_years)
    else:
        plt.plot(x, c = 'black')
    plt.show()
    
    return x

def kalmanFilter(y ,kappa, sigma2_eta, psi, Beta, logRV):
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
        if Beta is not None and logRV is not None:
            v[i] = y[i] - a[i] - kappa - Beta*logRV[i]
        F[i] = P[i] + 4.93
        K[i] = P[i] / F[i]
        
        if i >= len(y) - 1:
            break
        
        att[i] = a[i] + (P[i]*v[i])/F[i]
        a[i+1] = psi*att[i]
        # Beta[i+1] = Beta[i]
        
        Ptt[i] = P[i] - (P[i]**2)/F[i]
        P[i+1] = (psi**2)*Ptt[i] + sigma2_eta
    
    return P,a,v,F,K


def llllm(params, y, Beta, logRV):
    kappa, sigma2_eta, psi = [param for param in params]
    
    
    P,a,v,F,K = kalmanFilter(y, kappa, sigma2_eta, psi, Beta, logRV)
    
    
    ll = len(y)*0.5*np.log(2*np.pi) + 0.5*sum((np.log(F) + ((v**2))/(F)))
    return ll


def estimate(y, Beta, logRV):
    xt = y[:len(y)-1]
    xt_1 = y[1:]
    
    #psi_0 = (np.cov(xt_1,xt))[0][1]/(np.var(y) - (np.pi)**2/2
    psi_0 = 0.99
    sigma2_eta_0 = (1-psi_0**2)*(np.var(y)- ((np.pi)**2)/2)
    #sigma2_eta_0 =  0.8**2
    # print(np.mean(y))
    params = {
        "kappa": {"x0": np.mean(y), "bounds": [-100, 100]},
        "sigma2_eta": {"x0": sigma2_eta_0, "bounds": [0.001, 1000]},
        "psi": {"x0": psi_0, "bounds": [0, 1]},
    }
    
    x0 = [param["x0"] for key, param in params.items()]
    result = minimize(llllm,  x0,args=(y, Beta, logRV), tol = 1e-6, method='SLSQP', options={'maxiter': 500, 'disp': True})
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
    RV = RV[RV['Symbol'] == '.SPX']
    RV['date'] = RV['date'].str.split(' ').str[0]
    RV['date'] = pd.to_datetime(RV['date'], format='%Y-%m-%d')
    
    start_date = '2015-02-10'
    end_date = '2021-02-10'
    RV = RV[(RV['date'] >= start_date) & (RV['date'] <= end_date)]
    RV = RV.reset_index(drop=True)
    
    # Close to Close
    RV['Log Returns'] = np.log(RV['close_price'] / RV['close_price'].shift(1))

    RV = RV.drop(index=0)

    # Reset the index
    RV = RV.reset_index(drop=True)

    # Open to Close
    y = RV['Log Returns'].values
    
    y = np.log(RV['open_price'] / RV['close_price'])
    
    unique_years = RV['date'].dt.year.unique()
    
    plt.figure(figsize=(12, 6))
    plt.plot(RV['date'], y - np.mean(y), c = 'black')
    plt.xticks([pd.Timestamp(f'{year}-01-01') for year in unique_years], unique_years)
    plt.show()
    
    mean = np.mean(y)
    variance = np.var(y)
    skewness = np.mean((y - mean) ** 3) / np.power(variance, 3/2)
    kurtosis = np.mean((y - mean) ** 4) / np.power(variance, 2) - 3
    
    print(f"Mean: {mean}")
    print(f"Variance: {variance}")
    print(f"Skewness: {skewness}")
    print(f"Kurtosis: {kurtosis}")
    
    plt.figure(figsize=(12, 6))
    plt.plot(RV['date'], RV['close_price'], c = 'black')
    plt.xticks([pd.Timestamp(f'{year}-01-01') for year in unique_years], unique_years)
    plt.show()
        
    return RV, y, unique_years


def results(estParams, x, Beta, logRV, unique_years, RV):
    kappa, sigma2_eta, psi = [param for param in estParams]
    print('kappa is', kappa)
    print('sigma  is ', np.sqrt(np.exp(kappa + 1.27)))
    print('sigma  eta is', np.sqrt(sigma2_eta))
    print('phi is ',psi)
    
    P,a,v,F,K = kalmanFilter(x, kappa, sigma2_eta, psi, Beta, logRV)
    alpha,V,r,N = kalmanSmoothing(x,P,a,v,F,K)

    # Plot Alpha vs a
    plt.figure(figsize = (12,6))
    if len(unique_years) > 0:
        plt.plot(RV['date'],a)
        plt.plot(RV['date'],alpha)
        plt.xticks([pd.Timestamp(f'{year}-01-01') for year in unique_years], unique_years)
    else:
        plt.plot(range(len(a)),a)
        plt.plot(range(len(alpha)),alpha)
    plt.show()
        
    # Plot kappa + alpha vs xt
    plt.figure(figsize=(12, 6))
    if len(unique_years) > 0:
        plt.scatter(RV['date'],x, color = 'black')
        plt.plot(RV['date'], kappa + alpha, c = 'red', linewidth = 5)
        plt.xticks([pd.Timestamp(f'{year}-01-01') for year in unique_years], unique_years)
    else:
        plt.scatter(range(len(x)),x, color = 'black')
        plt.plot(kappa + alpha, c = 'red', linewidth = 5)
    plt.show()

    # Plot exp(alpha/2)
    plt.figure(figsize=(12, 6))    
    if len(unique_years) > 0:
        plt.plot(RV['date'], np.exp(alpha/2), c = 'black')
        plt.xticks([pd.Timestamp(f'{year}-01-01') for year in unique_years], unique_years)
    else:
        plt.plot(np.exp(alpha/2), c = 'black')
    plt.show()
    
    return a, v, F, kappa, sigma2_eta, psi


def tests(v, vNew, F, FNew, vUpdate, FUpdate, logRV, ySP, unique_years, RV):
    # Tests etc.
    
    t_statistic, p_value = ttest_ind(v, vNew)
    print(t_statistic, p_value)
    t_statistic, p_value = ttest_ind(F, FNew)
    print(t_statistic, p_value)
    alpha = 0.05  # Significance level
    if p_value < alpha:
        print("The means are significantly different.")
    else:
        print("The means are not significantly different.")
    
    plt.figure(figsize=(12, 6))
    plt.plot(RV['date'], v, c = 'red')
    plt.plot(RV['date'], vNew, c = 'black')
    plt.xticks([pd.Timestamp(f'{year}-01-01') for year in unique_years], unique_years)
    plt.show()
    
    plt.figure(figsize=(12, 6))
    plt.plot(RV['date'], F, c = 'red')
    plt.plot(RV['date'], FNew, c = 'black')
    plt.xticks([pd.Timestamp(f'{year}-01-01') for year in unique_years], unique_years)
    plt.show()
    
    plt.figure(figsize=(12, 6))
    plt.plot(RV['date'], v, c = 'red')
    plt.plot(RV['date'], vUpdate, c = 'black')
    plt.xticks([pd.Timestamp(f'{year}-01-01') for year in unique_years], unique_years)
    plt.show()
    
    plt.figure(figsize=(12, 6))
    plt.plot(RV['date'], F, c = 'red')
    plt.plot(RV['date'], FUpdate, c = 'black')
    plt.xticks([pd.Timestamp(f'{year}-01-01') for year in unique_years], unique_years)
    plt.show()
    
    print("Standard deviation of v:", np.std(v))
    print("Mean of v:", np.mean(v))
    print("Standard deviation of vNew:", np.std(vNew))
    print("Mean of vNew:", np.mean(vNew))
    print("Standard deviation of f:", np.std(F))
    print("Mean of f:", np.mean(F))
    print("Standard deviation of fNew:", np.std(FNew))
    print("Mean of fNew:", np.mean(FNew))

    Ycorr = ySP[:-1]
    RVcorr = logRV[1:]
    
    correlation_coef, p_value = pearsonr(logRV, ySP)

    print("Correlation coefficient:", correlation_coef)
    print("P-value:", p_value)
    
    # Check if correlation is significant at a 5% significance level
    if p_value < 0.05:
        print("The correlation is significant at the 5% level.")
    else:
        print("The correlation is not significant at the 5% level.")
    
    correlation_coef, p_value = pearsonr(RVcorr, Ycorr)

    print("Correlation coefficient:", correlation_coef)
    print("P-value:", p_value)
    
    # Check if correlation is significant at a 5% significance level
    if p_value < 0.05:
        print("The correlation is significant at the 5% level.")
    else:
        print("The correlation is not significant at the 5% level.")
    
    print(np.mean(logRV))
    print(np.std(logRV))
    

def bootstrapFilter(sigma,mu,psi,sigma2_eta,y,a):
    a1 = 0
    P1 = sigma2_eta/ (1 - (psi**2))
    
    alpha = np.random.normal(a1,np.sqrt(P1),10000)
   
    x = np.zeros(len(y))
    
    for i in range(len(y)):
        
        sigma2_t = sigma**2 * np.exp(alpha)
        w =  (1/(np.sqrt(2*math.pi*sigma2_t)))*np.exp(((y[i] - mu)**2)/ (-2*sigma2_t))
        norm_w = w / np.sum(w)
        
        x[i] = np.sum(norm_w * alpha)
        
        alpha = np.random.choice(alpha,len(alpha),p=norm_w)
        alpha = np.random.normal(psi* alpha, np.sqrt(sigma2_eta),10000)
        
    plt.figure(figsize=(12, 6))
    plt.plot(a)
    plt.plot(x, c='red')
    plt.show()
     
    return x

   
def main():
    
    beta_hat = None
    logRV = None
    unique_years = []
    RV = []
    
    # Log returns multiplied by 100
    df = pd.read_excel('C:/Users/rafam/Downloads/EOR/Master/P4/TS/Assignment 2/sv.xlsx')
    
    # Part A
    y = aa(df)

    # Part B
    x = b(y, unique_years, RV)
    
    # Part C & D
    estParams = estimate(x, beta_hat, logRV)
    a, v, F, kappa, sigma2_eta, psi = results(estParams, x, beta_hat, logRV, unique_years, RV)
        
    # Part E: revisit A-D with S&P 500 data
    RV, ySP, unique_years = RVdata()
    x = b(ySP, unique_years, RV)
    estParams = estimate(x, beta_hat, logRV)
    a_sp, v_sp, F_sp, kappa_sp, sigma2_eta_sp, psi_sp = results(estParams, x, beta_hat, logRV, unique_years, RV)
    
    # Part E: implement Beta*log(RV) into the model
    vol = RV['rv5_ss']
    volat = np.log(vol)
    X = volat + kappa_sp
    
    P,a,X_star,F1,K = kalmanFilter(X ,kappa_sp, sigma2_eta_sp, psi_sp, beta_hat, logRV)
    
    logRV = volat
    beta_hat = np.multiply(1/(np.sum(np.multiply(np.multiply(np.transpose(X_star),1/(F_sp)),X_star))),np.sum(np.multiply(np.multiply(np.transpose(X_star),1/(F_sp)),v_sp)))
    print(beta_hat)
    
    # Without re estimating paramteres    
    aNew, vNew, FNew, kappaNew, sigma2_etaNew, psiNew = results(estParams, x, beta_hat, logRV, unique_years, RV)
    
    # With re estimating paramteres
    estParams = estimate(x, beta_hat, logRV)
    aUpdate, vUpdate, FUpdate, kappaUpdate, sigma2_etaUpdate, psiUpdate = results(estParams, x, beta_hat, logRV, unique_years, RV)
    
    tests(v_sp, vNew, F_sp, FNew, vUpdate, FUpdate, logRV, ySP, unique_years, RV)
    
    # Part F  SV Data
    sigma = np.sqrt(np.exp(kappa + 1.27))
    mu = np.mean(y)
    
    bootstrapFilter(sigma,mu,psi,sigma2_eta,y,a)
    
    # Part F  S&P500
    sigma = np.sqrt(np.exp(kappa_sp + 1.27))
    mu = np.mean(ySP)

    bootstrapFilter(sigma,mu,psi_sp,sigma2_eta_sp,ySP,a_sp)

###########################################################
### call main
if __name__ == "__main__":
    start_time = time.monotonic()
    main()
    end_time = time.monotonic()
    print("\nExecution took:",timedelta(seconds=end_time - start_time))