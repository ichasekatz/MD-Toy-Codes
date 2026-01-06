#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 05:20:02 2025

@author: chasekatz
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statistics import stdev
from sympy import *

# Inputs
file_name = '/Users/chasekatz/Desktop/'

i = 2
avg = []
temp =[]
error = []
while i <= 22:
    temp += [i*100]
    filename = f'{file_name}Research/MD/Temp_change/Al-melting/Outputs_NVTtoNPTheating/samp/samp_{temp[i-2]}.out'
    log_file = filename
    df = pd.read_csv(log_file, skiprows=1, delim_whitespace=True, names=['step', 'ke', 'pe', 'te', 'enthalpy', 'vol', 'pres', 'pxx', 'pyy', 'pzz', 'pxy', 'vol_duplicate', 'lx'])
    steps = 0
    E = 0
    data = []
    for idx, row in df.iterrows():
        steps += 1
        E += row['vol']
        data += [row['vol']]
    avg += [E/steps]
    error += [stdev(data, (E/steps))]
    i += 1

x = temp[0:11]
y = avg[0:11]

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
xval = np.arange(0, 2200, 1)
func = f(xval)

c = polyFit(x,y,2)
quad = stdDev(c,x,y)
print('The quadratic line of regression is y =',c[2],'x^2 +',c[1],'x +',c[0])

x2 = symbols('x')
func2 = (c[2]*x2*x2)+(c[1]*x2)+c[0]
f = lambdify(x2, func2, 'numpy')
xval2 = np.arange(0, 2200, 1)
func2 = f(xval2)

plt.figure(figsize=(8, 6))
plt.plot(temp, avg, label='Volume', marker='o', color='b')
plt.plot(xval, func, '--', label='Linear fit', color='r')
plt.plot(xval2, func2, label='Quadratic fit', color='g')
plt.xlabel('Temp (K)')
plt.ylabel('Volume (A^3)')
plt.title('Volume vs. Temp')
plt.grid(True)
plt.legend()
plt.show()

print()
for i in range(len(temp)):
    print('|Temp:',temp[i],'|Volume:',avg[i],'|Error:+-',error[i],'|')
 
c = polyFit(x,y,1)
quad = stdDev(c,x,y)   
x1 = symbols('x')
func = (c[1]*x1)+c[0]
f = lambdify(x1, func, 'numpy')
xval = np.arange(200, 1400, 100)
func = f(xval)
total, residual = 0, 0
for i in range(len(func)):
    total += ((avg[i] - np.mean(avg[0:11])) ** 2)
    residual += ((avg[i] - func[i]) ** 2)
r2 = 1 - (residual/total)

print()
coeff = []
for i in range(len(temp)):
    coeff += [(c[1]/avg[i])]
    print('|Temp:',temp[i],'|Volumetric Thermal Expansion Coeff:',coeff[i],'|')

print()
print('Difference between 300K and 600K:', coeff[4]-coeff[1])

print('\nR^2 value for linear fit:',r2)

c = polyFit(x,y,2)
quad = stdDev(c,x,y)
x1 = symbols('x')
func = (c[2]*x1*x1)+(c[1]*x1)+c[0]
f = lambdify(x1, func, 'numpy')
xval = np.arange(200, 1400, 100)
func = f(xval)
total, residual = 0, 0
for i in range(len(func)):
    total += ((avg[i] - np.mean(avg[0:11])) ** 2)
    residual += ((avg[i] - func[i]) ** 2)
r2 = 1 - (residual/total)

print()
coeff = []
for i in range(len(temp)):
    x1 = temp[i]
    func = (2*c[2]*x1)+c[1]
    coeff += [(func/avg[i])]
    print('|Temp:',temp[i],'|Volumetric Thermal Expansion Coeff:',coeff[i],'|')

print()
print('Difference between 300K and 600K:', coeff[4]-coeff[1])

print('R^2 value for quadratic fit:',r2)
