#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 02:01:33 2025

@author: chasekatz
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statistics import stdev
from sympy import *

# Inputs
folder_directory = ''
min_temp = 0
max_temp = 5000

filename = f'{folder_directory}/MD/Temp_change/Ni-Melting/Outputs/thermo_output.txt'
df = pd.read_csv(filename, skiprows=2, delim_whitespace=True, names=['step', 'temp', 'press', 'etotal', 'vol'])

df['temp'] = pd.to_numeric(df['temp'], errors='coerce')
df['etotal'] = pd.to_numeric(df['etotal'], errors='coerce')

x = df['temp'].tolist()
y = df['etotal'].tolist()

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
    
x1 = symbols('x')
func = (c[1]*x1)+c[0]
f = lambdify(x1, func, 'numpy')
xval = np.arange(min_temp, max_temp, 1)
func = f(xval)


c = polyFit(x,y,2)
quad = stdDev(c,x,y)

x2 = symbols('x')
func2 = (c[2]*x2*x2)+(c[1]*x2)+c[0]
f = lambdify(x2, func2, 'numpy')
xval2 = np.arange(min_temp, max_temp, 1)
func2 = f(xval2)

plt.figure(figsize=(8, 6))
plt.plot(x, y, label='Internal Energy', marker='o', color='b')
plt.plot(xval, func, '--', label='Linear fit', color='r')
plt.plot(xval2, func2, label='Quadratic fit', color='g')
plt.xlabel('Temp (K)')
plt.ylabel('Internal Energy (eV/atom)')
plt.title('Internal Energy vs. Temp')
plt.grid(True)
plt.legend()
plt.show()

with open(filename, 'r') as file:
    line_count = sum(1 for line in file)-1

lines = int(round((line_count/(line_count/10)), -1))
print()
for j in range(lines):
    i = j*int(line_count/10)
    print('|Temp:',x[i],'|Energy:',y[i],'|')


