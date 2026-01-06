#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 22:45:07 2024

@author: chasekatz
"""

import pandas as pd
import matplotlib.pyplot as plt

# Inputs
folder_directory = ''

filename = f'{folder_directory}/MD/XRD/Ni-melting/Outputs/ni.xray'

df = pd.read_csv(filename, skiprows=3, delim_whitespace=True, names=['N','X','Y'])

df['N'] = pd.to_numeric(df['N'], errors='coerce')
df['X'] = pd.to_numeric(df['X'], errors='coerce')
df['Y'] = pd.to_numeric(df['Y'], errors='coerce')

rows = []
for i in range(11):
    rows += [i*14347]

for i in range(len(rows)-1):
    plt.figure(figsize=(8, 6))
    start_idx = (rows[i])+1
    end_idx = (rows[i+1])
    plt.scatter(df['X'].iloc[start_idx:end_idx], df['Y'].iloc[start_idx:end_idx], label=f'XRD {i+1}')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'XRD {i+1}')
    plt.grid(True)
    plt.legend()
    plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(df['X'], df['Y'], label=f'XRD')
plt.xlabel('X')
plt.ylabel('Y')
plt.title(f'XRD')
plt.grid(True)
plt.legend()
plt.show()
