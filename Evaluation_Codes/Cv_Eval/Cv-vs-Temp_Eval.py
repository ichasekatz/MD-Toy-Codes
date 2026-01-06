#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 05:31:41 2025

@author: chasekatz
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statistics import stdev
from sympy import *

# Inputs
file_name = ''

i = 2
avg = []
temp =[]
error = []
while i <= 12:
    temp += [i*100]
    filename = f'{file_name}/MD/Temp_change/Al-melting/Outputs_NPT/samp_{temp[i-2]}.out'
    log_file = filename
    df = pd.read_csv(log_file, skiprows=1, delim_whitespace=True, names=['step', 'ke', 'pe', 'te', 'enthalpy', 'vol', 'pres', 'pxx', 'pyy', 'pzz', 'pxy', 'vol_duplicate', 'lx'])
    E = 0
    Elist = []
    E2 = 0
    E2list = []
    vol = 0
    volist = []
    steps = 0
    for idx, row in df.iterrows():
        steps += 1
        E += (row['vol']*row['enthalpy'])
        Elist += [(row['vol']*row['enthalpy'])]
        E2 += (row['enthalpy'])
        E2list += [(row['enthalpy'])]
        vol += row['vol']
        volist += [row['vol']]
    KbT2 = ((temp[i-2]**2)*8.617333262145*(10**(-5)))
    avg += [((E/steps)-((E2/steps)*(vol/steps)))/(KbT2*(vol/steps))]
    error += [((((stdev(Elist))**2)+((np.mean(volist)**2)*((stdev(E2list))**2))+(((np.mean(E2list))**2)*((stdev(volist))**2)))**(1/2))/KbT2]
    i += 1

x = temp
y = avg

def swapRows(v,i,j):
    if len(v.shape) == 1:
        v[i],v[j] = v[j],v[i]
    else:
        v[[i,j],:] = v[[j,i],:]

def swapCols(v,i,j):
    v[:,[i,j]] = v[:,[j,i]]

def gaussPivot(a,b,tol=1.0e-12):
    n = len(b)
    s = np.zeros(n)
    for i in range(n):
        s[i] = max(np.abs(a[i,:]))
    for k in range(0,n-1):
        p = np.argmax(np.abs(a[k:n,k])/s[k:n]) + k
        if p != k:
            swapRows(b,k,p)
            swapRows(s,k,p)
            swapRows(a,k,p)
        for i in range(k+1,n):
            if a[i,k] != 0.0:
                lam = a[i,k]/a[k,k]
                a[i,k+1:n] = a[i,k+1:n] - lam*a[k,k+1:n]
                b[i] = b[i] - lam*b[k]
    b[n-1] = b[n-1]/a[n-1,n-1]
    for k in range(n-2,-1,-1):
        b[k] = (b[k] - np.dot(a[k,k+1:n],b[k+1:n]))/a[k,k]
    return b

def polyFit(xData,yData,m):
    a = np.zeros((m+1,m+1))
    b = np.zeros(m+1)
    s = np.zeros(2*m+1)
    for i in range(len(xData)):
        temp = yData[i]
        for j in range(m+1):
            b[j] = b[j] + temp
            temp = temp*xData[i]
        temp = 1.0
        for j in range(2*m+1):
            s[j] = s[j] + temp
            temp = temp*xData[i]
    for i in range(m+1):
        for j in range(m+1):
            a[i,j] = s[i+j]
    return gaussPivot(a,b)

def stdDev(c,xData,yData):
    def evalPoly(c,x):
        m = len(c) - 1
        p = c[m]
        for j in range(m):
            p = p*x + c[m-j-1]
        return p
    n = len(xData) - 1
    m = len(c) - 1
    sigma = 0.0
    for i in range(n+1):
        p = evalPoly(c,xData[i])
        sigma = sigma + (yData[i] - p)**2
    sigma = (sigma/(n - m))**(1/2)
    return sigma

def plotPoly(xData,yData,coeff,xlab='x',ylab='y'):
    m = len(coeff)
    x1 = min(xData)
    x2 = max(xData)
    dx = (x2 - x1)/20.0
    x = np.arange(x1,x2 + dx/10.0,dx)
    y = np.zeros((len(x)))*1.0
    for i in range(m):
        y = y + coeff[i]*x**i
    plt.plot(xData,yData,'o',x,y,'-')
    plt.xlabel(xlab); plt.ylabel(ylab)
    plt.grid (True)
    plt.show()
    
c = polyFit(x,y,1)
quad = stdDev(c,x,y)
print('\nThe linear line of regression is y =',c[1],'x +',c[0])
    
x1 = symbols('x')
func = (c[1]*x1)+c[0]
f = lambdify(x1, func, 'numpy')
xval = np.arange(200, 1200, 1)
func = f(xval)

c = polyFit(x,y,2)
quad = stdDev(c,x,y)
print('The quadratic line of regression is y =',c[2],'x^2 +',c[1],'x +',c[0])

x2 = symbols('x')
func2 = (c[2]*x2*x2)+(c[1]*x2)+c[0]
f = lambdify(x2, func2, 'numpy')
xval2 = np.arange(200, 1200, 1)
func2 = f(xval2)

plt.figure(figsize=(8, 6))
plt.plot(temp, avg, label='Thermal expansion', marker='o', color='b')
plt.plot(xval, func, '--', label='Linear fit', color='r')
plt.plot(xval2, func2, label='Quadratic fit', color='g')
plt.xlabel('Temp (K)')
plt.ylabel('Thermal expansion (1/K)')
plt.title('Thermal expansion vs. Temp')
plt.grid(True)
plt.legend()
plt.show()

print()
for i in range(len(temp)):
    print('|Temp:',temp[i],'|Thermal expansion:',avg[i],'(1/K)|Error:+-',error[i],'|')

c = polyFit(x,y,1)
quad = stdDev(c,x,y)
x1 = symbols('x')
func = (c[1]*x1)+c[0]
f = lambdify(x1, func, 'numpy')
xval = np.arange(200, 1300, 100)
func = f(xval)
total, residual = 0, 0
for i in range(len(func)):
    total += ((avg[i] - np.mean(avg)) ** 2)
    residual += ((avg[i] - func[i]) ** 2)
r2 = 1 - (residual/total)
print('\nR^2 value for linear fit:',r2)

c = polyFit(x,y,2)
quad = stdDev(c,x,y)
x1 = symbols('x')
func = (c[2]*x1*x1)+(c[1]*x1)+c[0]
f = lambdify(x1, func, 'numpy')
xval = np.arange(200, 1300, 100)
func = f(xval)
total, residual = 0, 0
for i in range(len(func)):
    total += ((avg[i] - np.mean(avg)) ** 2)
    residual += ((avg[i] - func[i]) ** 2)
r2 = 1 - (residual/total)
print('R^2 value for quadratic fit:',r2)
