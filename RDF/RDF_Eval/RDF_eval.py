#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 01:27:12 2024

@author: chasekatz
"""
import pandas as pd
import matplotlib.pyplot as plt

# Inputs
folder_directory = '/Users/chasekatz/Desktop/'

filename = f'{folder_directory}Research/MD/RDF/Ni-melting/Outputs/ni.rdf'

df = pd.read_csv(filename, skiprows=3, delim_whitespace=True, names=['N','X','Y','Z'])

df['Y'] = pd.to_numeric(df['Y'], errors='coerce')
df['N'] = pd.to_numeric(df['N'], errors='coerce')
df_filtered = df[(df['N'] >= 1) & (df['N'] <= 200)]
df_filtered = df_filtered[(df_filtered['Y'] >= 0.0001) & (df_filtered['Y'] <= 200)]

j = 0
i = 0
k = [0]
for idx, row in df_filtered.iterrows():
    if row['N'] != 200:
        j += 1
    if row['N'] == 200:
        k += [j+k[i]]
        j = 0
        i += 1
        j += 1
rows = k

df['X'] = pd.to_numeric(df_filtered['X'], errors='coerce')
df['Y'] = pd.to_numeric(df_filtered['Y'], errors='coerce')

for i in range(len(rows)-1):
    plt.figure(figsize=(8, 6))
    start_idx = (rows[i])+1
    end_idx = (rows[i+1])
    plt.plot(df_filtered['X'].iloc[start_idx:end_idx], df_filtered['Y'].iloc[start_idx:end_idx], label=f'RDF {i+1}')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'RDF {i+1}')
    plt.grid(True)
    plt.legend()
    plt.show()

plt.figure(figsize=(8, 6))
start_idx = rows[0]
end_idx = rows[len(rows)-1]
plt.scatter(df_filtered['X'].iloc[start_idx:end_idx], df_filtered['Y'].iloc[start_idx:end_idx], label=f'RDF {i+1}')
plt.xlabel('X')
plt.ylabel('Y')
plt.title(f'RDF {i+1}')
plt.grid(True)
plt.legend()
plt.show()