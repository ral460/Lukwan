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



###########################################################

def plotKF(y,year,P,a,v,F,UB,LB):
   
    
    plt.scatter(year,y,s=10, c='black',linewidth=1)
    
    
    plt.yticks(np.arange(500,1500,250))
    plt.plot(year[1:],a[1:],c='black',linewidth=1)
    plt.plot(year[1:],UB[1:], c='grey',linewidth=1)
    plt.plot(year[1:],LB[1:], c='grey',linewidth=1)
    plt.text(0.02, 0.98, "(i)", fontsize=12, transform=plt.gca().transAxes, verticalalignment='top', horizontalalignment='left', color='black')
    plt.savefig('11.png')
    plt.show()
    
    plt.plot(year[1:], P[1:],c='black',linewidth=1)
    plt.yticks(np.arange(7500,18500,2500))
    plt.text(0.02, 0.98, "(ii)", fontsize=12, transform=plt.gca().transAxes, verticalalignment='top', horizontalalignment='left', color='black')
    plt.savefig('12.png')
    plt.show()
    
    plt.plot(year[1:],v[1:],c='black',linewidth=1)
    plt.axhline(y=0,c='black',linewidth=1)
    plt.yticks(np.arange(-250,350,250))
    plt.text(0.02, 0.98, "(iii)", fontsize=12, transform=plt.gca().transAxes, verticalalignment='top', horizontalalignment='left', color='black')
    plt.savefig('13.png')
    plt.show()
    
    plt.plot(year[1:],F[1:],c='black',linewidth=1)
    plt.yticks(np.arange(22500,33500,2500))
    plt.text(0.02, 0.98, "(iv)", fontsize=12, transform=plt.gca().transAxes, verticalalignment='top', horizontalalignment='left', color='black')
    plt.savefig('14.png')
    plt.show()
    
def plotKS(y, year,V,alpha,r,N,UB,LB):
    
    
    plt.scatter(year,y,s=10, c='black',linewidth=1)
    
    plt.yticks(np.arange(500,1500,250))
    plt.plot(year,alpha,c='black',linewidth=1)
    plt.plot(year,UB, c='grey',linewidth=1)
    plt.plot(year,LB, c='grey',linewidth=1)
    plt.text(0.02, 0.98, "(i)", fontsize=12, transform=plt.gca().transAxes, verticalalignment='top', horizontalalignment='left', color='black')
    plt.savefig('21.png')
    plt.show()
    
    plt.plot(year, V,c='black',linewidth=1)
    plt.yticks(np.arange(2500,4100,500))
    plt.text(0.02, 0.98, "(ii)", fontsize=12, transform=plt.gca().transAxes, verticalalignment='top', horizontalalignment='left', color='black')
    plt.savefig('22.png')
    plt.show()
    
    plt.plot(year[1:],r[:len(y)-1],c='black',linewidth=1)
    plt.axhline(y=0,c='black',linewidth=1)
    plt.yticks(np.arange(-0.02,0.03,0.02))
    plt.text(0.02, 0.98, "(iii)", fontsize=12, transform=plt.gca().transAxes, verticalalignment='top', horizontalalignment='left', color='black')
    plt.savefig('23.png')
    plt.show()
    
    plt.plot(year[1:],N[:len(y)-1],c='black',linewidth=1)
    plt.yticks(np.arange(6e-5,1.1e-4,2e-5))
    plt.text(0.02, 0.98, "(iv)", fontsize=12, transform=plt.gca().transAxes, verticalalignment='top', horizontalalignment='left', color='black')
    plt.savefig('24.png')
    plt.show()
    
def CI(a,P,z):
    LB = a - (z * np.sqrt(P))
    UB = a +  (z * np.sqrt(P))
    
    return LB,UB


def kalmanSmoothing(y, year,P,a,v,F,K):
    z = 1.645
    r = np.zeros(len(P)+1)
    N = np.zeros(len(P)+1)
    
    alpha = np.zeros(len(P))
    V = np.zeros(len(P))
    for i in reversed(range(len(P))):
        
        r[i] =  (pow(F[i],-1) * v[i]) + (1-K[i]) * r[i+1]
        N[i] =  pow(F[i],-1) + ((1-K[i])**2) * N[i+1]
        
        alpha[i] = a[i] + (P[i] * r[i])
        V[i] = P[i] - ((P[i]**2) * N[i])
        
    
    LB,UB = CI(alpha,V,z)
    #
    plotKS(y, year,V,alpha,r[1:],N[1:],UB,LB)
    return alpha,V,r[1:],N[1:]

def kalmanFilter(y, year,P1,sigma2_epsilon,sigma2_eta):
    z = 1.645
    v = np.zeros(len(y))
    a = np.zeros(len(y))
    P = np.zeros(len(y))
    F = np.zeros(len(y))
    K = np.zeros(len(y))
    P[0] = P1
    for i in range(len(y)):
        v[i] = y[i] - a[i]
        F[i] = P[i] + sigma2_epsilon
        K[i] = P[i]/F[i]
        
        if i >= len(y)-1:
            break 
        
        a[i+1] = a[i] + (K[i]* v[i])
        P[i+1] = P[i] * (1 - K[i]) + sigma2_eta
        
    LB,UB = CI(a,P,z)
    plotKF(y,year,P,a,v,F,UB,LB)
    
    return P,a,v,F,K
    
def disturbances(y, year,alpha,sigma2_epsilon,sigma2_eta,F,K,N):
   
    epsilon = y - alpha
    
    
    D =  pow(F,-1) + (K **2) * N

    variance_epsilon = sigma2_epsilon - (sigma2_epsilon**2) * D
    
    plt.plot(year,epsilon,c='black',linewidth=1)
    plt.axhline(y=0,c='black',linewidth=1)
    plt.yticks(np.arange(-200,300,200))
    plt.text(0.02, 0.98, "(i)", fontsize=12, transform=plt.gca().transAxes, verticalalignment='top', horizontalalignment='left', color='black')
    plt.savefig('31.png')
    plt.show()
    
    plt.plot(year,np.sqrt(variance_epsilon),c='black',linewidth=1)
    plt.yticks(np.arange(50,61,5))
    plt.text(0.02, 0.98, "(ii)", fontsize=12, transform=plt.gca().transAxes, verticalalignment='top', horizontalalignment='left', color='black')
    plt.savefig('32.png')
    plt.show()
    
    eta = alpha[1:] - alpha[:len(alpha)-1]
    variance_eta = sigma2_eta - (sigma2_eta**2) * N
    
    plt.plot(year[:len(eta)],eta,c='black',linewidth=1)
    plt.axhline(y=0,c='black',linewidth=1)
    plt.yticks(np.arange(-25,30,25))
    plt.text(0.02, 0.98, "(iii)", fontsize=12, transform=plt.gca().transAxes, verticalalignment='top', horizontalalignment='left', color='black')
    plt.savefig('33.png')
    plt.show()
    
    plt.plot(year,np.sqrt(variance_eta),c='black',linewidth=1)
    plt.yticks(np.arange(36,38.5,1))
    plt.text(0.02, 0.98, "(iv)", fontsize=12, transform=plt.gca().transAxes, verticalalignment='top', horizontalalignment='left', color='black')
    plt.savefig('34.png')
    plt.show()
    
    return epsilon, eta

def simulation(y, year,sigma2_epsilon,sigma2_eta,epsilon, alpha,eta,P1):  
    mu = 0
    
    np.random.seed(0)
    epsilon_plus = np.random.normal(mu , np.sqrt(sigma2_epsilon), len(y))
    eta_plus =  np.random.normal(mu , np.sqrt(sigma2_eta), len(y))
    
    
    alpha_plus = np.zeros(len(y))
    alpha_plus[0] = alpha[0]
    for i in range(len(y)-1):
        alpha_plus[i+1] = alpha_plus[i] + eta_plus[i]
    
    y_plus = alpha_plus + epsilon_plus
    
    # Hier krijg je dus nog 4 extra plots voor yplus, maar die boeien niet
    P,a,v,F,K= kalmanFilter(y_plus, year,P1,sigma2_epsilon,sigma2_eta)
    
    alpha_hat = kalmanSmoothing(y_plus, year,P,a,v,F,K)[0]
    
    epsilon_plushat = y_plus - alpha_hat
    
    
    epsilon_tilde = epsilon_plus - epsilon_plushat + epsilon
    
    alpha_tilde = y- epsilon_tilde
   
    
    eta_tilde = alpha_tilde[1:] - alpha_tilde[:(len(alpha_tilde)-1)]

    plt.scatter(year,alpha_plus,s=10, c='black')
    plt.plot(year,alpha,c='black',linewidth=1)
    plt.text(0.02, 0.98, "(i)", fontsize=12, transform=plt.gca().transAxes, verticalalignment='top', horizontalalignment='left', color='black')
    plt.savefig('41.png')
    plt.show()
    
    plt.scatter(year,alpha_tilde,s=10, c='black')
    plt.plot(year,alpha,c='black',linewidth=1)
    plt.text(0.02, 0.98, "(ii)", fontsize=12, transform=plt.gca().transAxes, verticalalignment='top', horizontalalignment='left', color='black')
    plt.savefig('42.png')
    plt.show()
    
    plt.scatter(year,epsilon_tilde,s=10, c='black')
    plt.axhline(y=0,c='black',linewidth=1)
    plt.plot(year,epsilon,c='black',linewidth=1)
    plt.text(0.02, 0.98, "(iii)", fontsize=12, transform=plt.gca().transAxes, verticalalignment='top', horizontalalignment='left', color='black')
    plt.savefig('43.png')
    plt.show()
    
    plt.scatter(year[:len(eta_tilde)],eta_tilde,s=10, c='black')
    plt.axhline(y=0,c='black',linewidth=1)
    plt.plot(year[:len(eta)],eta,c='black',linewidth=1)
    plt.text(0.02, 0.98, "(iv)", fontsize=12, transform=plt.gca().transAxes, verticalalignment='top', horizontalalignment='left', color='black')
    plt.savefig('44.png')
    plt.show()
    
def plotMissingFilter(df,P,a,v):
    year = df.iloc[1:,0]
        
    plt.yticks(np.arange(500,1500,250))
    plt.plot(year,df.iloc[1:,1],c='black',linewidth=1)
    plt.plot(year,a[1:],c='black',linewidth=1)
    plt.text(0.02, 0.98, "(i)", fontsize=12, transform=plt.gca().transAxes, verticalalignment='top', horizontalalignment='left', color='black')
    plt.savefig('51.png')
    plt.show()
    
    plt.plot(year, P[1:],c='black',linewidth=1)
    plt.yticks(np.arange(2500,35001,10000))
    plt.text(0.02, 0.98, "(ii)", fontsize=12, transform=plt.gca().transAxes, verticalalignment='top', horizontalalignment='left', color='black')
    plt.savefig('52.png')
    plt.show()
        
def missingFilter(df,P1,sigma2_epsilon,sigma2_eta):
    df.loc[df.index[20:40].append(df.index[60:80]), 'Nile'] = np.nan
    
    v = np.zeros(len(df))
    a = np.zeros(len(df))
    P = np.zeros(len(df))
    F = np.zeros(len(df))
    K = np.zeros(len(df))
    x = np.zeros(len(df))
    P[0] = P1
    
    numb_missing = 0
    for i in range(len(df)):
        if pd.isna(df['Nile'][i]):
            numb_missing += 1
            # df['Nile'][i] = df['Nile'][i-1]
            v[i] = df['Nile'][i - numb_missing] - a[i]
            F[i] = P[i] + sigma2_epsilon
            K[i] = 0
            
        else:
            numb_missing = 0
        
            v[i] = df['Nile'][i] - a[i]
            F[i] = P[i] + sigma2_epsilon
            K[i] = P[i]/F[i]
            
        if i >= len(df)-1:
            break 
                    
        P[i+1] = P[i] * (1 - K[i]) + sigma2_eta
       
        a[i+1] = a[i] + (K[i]* v[i])
    
    plotMissingFilter(df,P,a,v)
    
    return P,a,v,F,K

    
def missingSmoothing(df,P,a,v,F,K):
    df.loc[df.index[20:40].append(df.index[60:80]), 'Nile'] = np.nan
    
    z = 0.674
    r = np.zeros(len(P)+1)
    N = np.zeros(len(P)+1)
    
    alpha = np.zeros(len(P))
    V = np.zeros(len(P))
    
    numb_missing = 0
    for i in reversed(range(len(P))):
        if (pd.isna(df['Nile'][i])):
            numb_missing += 1
            r[i] = r[i+1]
            K[i] = 0
            N[i] = ((1-K[i])**2) * N[i+1]

        else:
            numb_missing = 0
            r[i] =  (pow(F[i],-1) * v[i]) + (1-K[i]) * r[i+1]
            N[i] =  pow(F[i],-1) + ((1-K[i])**2) * N[i+1]
        
        alpha[i] = a[i] + (P[i] * r[i])
        V[i] = P[i] - ((P[i]**2) * N[i])
        
    
    LB,UB = CI(alpha,V,z)
    #
    plotMissingSmoothing(df,V,alpha,r[1:],N[1:],UB,LB)
    return alpha,V,r[1:],N[1:]

def plotMissingSmoothing(df,V,alpha,r,N,UB,LB):
    year = df.iloc[1:,0]
        
    plt.yticks(np.arange(500,1500,250))
    plt.plot(year,df.iloc[1:,1],c='black',linewidth=1)
    plt.plot(year,alpha[1:],c='black',linewidth=1)
    plt.text(0.02, 0.98, "(iii)", fontsize=12, transform=plt.gca().transAxes, verticalalignment='top', horizontalalignment='left', color='black')
    plt.savefig('53.png')
    plt.show()
    
    plt.plot(df.iloc[:,0], V, c='black',linewidth=1)
    plt.yticks(np.arange(2500,10001,2500))
    plt.text(0.02, 0.98, "(iv)", fontsize=12, transform=plt.gca().transAxes, verticalalignment='top', horizontalalignment='left', color='black')
    plt.savefig('54.png')
    plt.show()


def plotForecast(df,P,a,v,F,UB,LB):
    year = df.iloc[1:,0]
    
    plt.scatter(df.iloc[:,0],df['Nile'],s=10, c='black',linewidth=1)
    
    UB = UB[-30:]
    LB = LB[-30:]
    
    plt.yticks(np.arange(500,1500,250))
    plt.plot(year,a[1:],c='black',linewidth=1)
    plt.plot(year[-30:],UB, c='grey',linewidth=1)
    plt.plot(year[-30:],LB, c='grey',linewidth=1)
    plt.text(0.02, 0.98, "(i)", fontsize=12, transform=plt.gca().transAxes, verticalalignment='top', horizontalalignment='left', color='black')
    plt.savefig('61.png')
    plt.show()
    
    plt.plot(year, P[1:],c='black',linewidth=1)
    plt.yticks(np.arange(5000,50001,10000))
    plt.text(0.02, 0.98, "(ii)", fontsize=12, transform=plt.gca().transAxes, verticalalignment='top', horizontalalignment='left', color='black')
    plt.savefig('62.png')
    plt.show()
    
    plt.plot(year,a[1:],c='black',linewidth=1)
    plt.yticks(np.arange(700,1200,100))
    plt.text(0.02, 0.98, "(iii)", fontsize=12, transform=plt.gca().transAxes, verticalalignment='top', horizontalalignment='left', color='black')
    plt.savefig('63.png')
    plt.show()
    
    plt.plot(year,F[1:],c='black',linewidth=1)
    plt.yticks(np.arange(20000,65000,10000))
    plt.text(0.02, 0.98, "(iv)", fontsize=12, transform=plt.gca().transAxes, verticalalignment='top', horizontalalignment='left', color='black')
    plt.savefig('64.png')
    plt.show()
    
def forecast(df,P1,sigma2_epsilon,sigma2_eta):
    z = 0.674
    v = np.zeros(len(df))
    a = np.zeros(len(df))
    P = np.zeros(len(df))
    F = np.zeros(len(df))
    K = np.zeros(len(df))

    P[0] = P1
    numb_missing = 0
    for i in range(len(df)):
        if pd.isna(df['Nile'][i]):
            numb_missing += 1
            v[i] = df['Nile'][i - numb_missing] - a[i]
            F[i] = P[i] + sigma2_epsilon
            K[i] = 0
            
        else:
            numb_missing = 0
            
            v[i] = df['Nile'][i] - a[i]
            F[i] = P[i] + sigma2_epsilon
            K[i] = P[i]/F[i]
            
        if i >= len(df)-1:
            break
        

        P[i+1] = P[i] * (1 - K[i]) + sigma2_eta
                
        a[i+1] = a[i] + (K[i]* v[i])
        
    LB,UB = CI(a,P,z)
    plotForecast(df,P,a,v,F,UB,LB)
    
    return P,a,v,F,K

def check(nu_t, F, df, year):
    
    et = nu_t/np.sqrt(F)
    
    plt.plot(year[1:], et[1:], c='black',linewidth=1)
    plt.axhline(0, color='black', linestyle='-', linewidth=1, alpha = 0.5)
    plt.text(0.02, 0.98, "(i)", fontsize=12, transform=plt.gca().transAxes, verticalalignment='top', horizontalalignment='left', color='black')
    plt.savefig('71.png')
    plt.show()
    
    plt.hist(et[1:], bins=13, density=True, edgecolor='black', facecolor='white', linewidth=1)
    sns.kdeplot(et[1:], color='black', linewidth=1)
    plt.xlim(-4, 3.75)
    plt.xticks(np.arange(-3, 4, 1))
    plt.text(0.02, 0.98, "(ii)", fontsize=12, transform=plt.gca().transAxes, verticalalignment='top', horizontalalignment='left', color='black')
    plt.savefig('72.png')
    plt.show()
    
    sm.qqplot(et[1:], fit=True, markerfacecolor='none', markeredgecolor='none', color='black', linestyle='solid')
    plt.text(0.02, 0.98, "(iii)", fontsize=12, transform=plt.gca().transAxes, verticalalignment='top', horizontalalignment='left', color='black')
    plt.legend(['normal'], loc='upper left')
    plt.axline((0, 0), slope=1, color='black', linestyle='solid')
    plt.savefig('73.png')
    plt.show()
    
    acf_values = sm.tsa.acf(et[1:], nlags=10, alpha=None, fft=True)

    # Adjust the width of the bars
    bar_width = 0.6

    # Create a bar plot with specified width
    plt.bar(np.arange(1, 11) - bar_width/2, acf_values[1:], width=bar_width, color='black', alpha = 0.8)

    plt.ylim(-1, 1)
    plt.yticks(np.arange(-1, 1.1, 0.5))
    plt.xlim(0, 11)
    plt.xticks(np.arange(0, 11, 5))
    plt.axhline(0, color='black', linestyle='-', linewidth=1, alpha = 0.5)
    plt.text(0.02, 0.98, "(iv)", fontsize=12, transform=plt.gca().transAxes, verticalalignment='top', horizontalalignment='left', color='black')
    plt.savefig('74.png')
    plt.show()
    
def aux(y,df,year,alpha):
    v =y-alpha
    e_star = v/np.sqrt(np.var(v))
    alpha_hat = alpha[:len(alpha)-1]
    nu_star = alpha[1:] - alpha_hat
    r_star = nu_star/np.sqrt(np.var(nu_star))
    
    plt.plot(year[1:], e_star[1:], c='black',linewidth=1)
    plt.axhline(0, color='black', linestyle='-', linewidth=1, alpha = 0.5)
    plt.text(0.02, 0.98, "(i)", fontsize=12, transform=plt.gca().transAxes, verticalalignment='top', horizontalalignment='left', color='black')
    plt.savefig('81.png')
    plt.show()
    
    plt.hist(e_star[1:], bins=13, density=True, edgecolor='black', facecolor='white', linewidth=1)
    sns.kdeplot(e_star[1:], color='black', linewidth=1)
    plt.xlim(-4, 3.75)
    plt.xticks(np.arange(-3, 4, 1))
    plt.text(0.02, 0.98, "(ii)", fontsize=12, transform=plt.gca().transAxes, verticalalignment='top', horizontalalignment='left', color='black')
    plt.savefig('82.png')
    plt.show()
    
    plt.plot(year[1:], r_star, c='black',linewidth=1)
    plt.axhline(0, color='black', linestyle='-', linewidth=1, alpha = 0.5)
    plt.text(0.02, 0.98, "(iii)", fontsize=12, transform=plt.gca().transAxes, verticalalignment='top', horizontalalignment='left', color='black')
    plt.savefig('83.png')
    plt.show()
    
    plt.hist(r_star, bins=13, density=True, edgecolor='black', facecolor='white', linewidth=1)
    sns.kdeplot(r_star, color='black', linewidth=1)
    plt.xlim(-4, 3.75)
    plt.xticks(np.arange(-3, 4, 1))
    plt.text(0.02, 0.98, "(iv)", fontsize=12, transform=plt.gca().transAxes, verticalalignment='top', horizontalalignment='left', color='black')
    plt.savefig('84.png')
    plt.show()
    
def estimate(y,P1,a):
   
    x0 = 1
    result = minimize(llRealizedGarch,  x0,args=(y,P1), tol = 1e-6,method='SLSQP', options = {'disp': True})
    print('ll',result)
    print(result.x)

def llRealizedGarch(x0, y,P1):
    
    q = x0
    
    v = np.zeros(len(y))
    P = np.zeros(len(y))
    F = np.zeros(len(y))
    K = np.zeros(len(y))
    a = np.zeros(len(y))
    P[0] = P1
    
    
    for i in range(len(y)):
        v[i] = y[i] - a[i]
        F[i] = P[i] + 1
        K[i] = P[i]/F[i]
    
        if i >= len(y)-1:
             break 
        
        a[i+1] = a[i] + (K[i]* v[i])
        P[i+1] = P[i] * (1 - K[i]) + q
        
    sigma_e_hat = (1/(len(y)-1)) * sum ((v[1:] **2)/F[1:])
    ll = (len(y)-1)*0.5 +len(y)*0.5 *np.log(2 *np.pi) + 0.5 * sum(np.log(F[1:])) + (len(y)-1)*0.5 *np.log(sigma_e_hat)
    
    print(sigma_e_hat)
    sigma_eta_hat = q* sigma_e_hat
    print(sigma_eta_hat)
    print(type(ll))
    
#     return ll
    
def main():
    df = pd.read_excel('C:/Users/rafam/Downloads/EOR/Master/P4/TS/Assignment 1/Nile.xlsx')
    year = df.iloc[:,0]
    y = df['Nile'].to_numpy()
    #initialization
    P1 = 10e6
    sigma2_epsilon = 15099
    sigma2_eta = 1469.1
    
    df1 = df
    original_years = df1.iloc[:, 0].tolist()
    new_years = range(max(original_years) + 1, max(original_years) + 31)
    new_data = {'Unnamed: 0': new_years, 'Nile': [np.nan] * 30}
    new_df = pd.DataFrame(new_data)
    result_df = pd.concat([df1, new_df], ignore_index=True)
    
    #2.1
    P,a,v,F,K = kalmanFilter(y, year,P1,sigma2_epsilon,sigma2_eta)
    #2.2
    alpha,V,r,N = kalmanSmoothing(y, year,P,a,v,F,K)
    #2.3
    epsilon, eta = disturbances(y, year,alpha,sigma2_epsilon,sigma2_eta,F,K,N)
    #2.4
    simulation(y,year,sigma2_epsilon,sigma2_eta,epsilon, alpha,eta,P1)
    #2.5
    P,a,v,F,K = missingFilter(df, P1, sigma2_epsilon, sigma2_eta)
    missingSmoothing(df, P, a, v, F, K)
    #2.6
    P,a,v,F,K = kalmanFilter(y, year,P1,sigma2_epsilon,sigma2_eta)
    forecast(result_df,P1,sigma2_epsilon,sigma2_eta)
    #2.7
    check(v, F, df, year)
    #2.8
    aux(y,df,year,alpha)
    
    # Likelihood
    # estimate(y,P1,a)
    

###########################################################
### call main
if __name__ == "__main__":
    start_time = time.monotonic()
    main()
    end_time = time.monotonic()
    print("\nExecution took:",timedelta(seconds=end_time - start_time))