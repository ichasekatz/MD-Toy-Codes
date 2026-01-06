#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 22:22:29 2024

@author: chasekatz
"""

import pandas as pd
import matplotlib.pyplot as plt


# Inputs
folder_directory = ''

filename = f'{folder_directory}/MD/Temp_change/Ni-Melting/Outputs/thermo_output.txt'
filename = f'{folder_directory}/MD/H2O/2phase/Output/log.lammps'
df = pd.read_csv(filename, skiprows=91, delim_whitespace=True, names=['step', 'temp', 'press', 'etotal', 'vol'])

df['step'] = pd.to_numeric(df['step'], errors='coerce')
df['temp'] = pd.to_numeric(df['temp'], errors='coerce')

i = 0
Var = []
temps = 0
steps = []
for idx, row in df.iterrows():
    if i >= 0:
        if row['temp'] >= 0:
            temps += row['temp']
            variance = row['temp']
            if i >= 1:
                tempav = temps/(i-1)
                Var += [(((row['temp'])-(tempav))**2)/(i-1)]
                steps += [row['step']]
    i += 1

print(len(steps))
print(len(Var))
print(Var)
plt.figure(figsize=(10, 6))
#plt.scatter(steps, Var, label='Variance')
plt.scatter(df['step'], df['temp'], label='Variance')
plt.xlabel('Timestep')
plt.ylabel('Variance')
plt.xlim(240,250)
plt.title('Variance vs. Timestep')
plt.grid(True)
plt.legend()
plt.show()
