# -*- coding: utf-8 -*-
"""
Created on Tue May 30 19:08:11 2023

@author: Shawn Skelton
"""

import pickle 
import matplotlib.pyplot as plt
import numpy as np

###grab all the initial data
filename='H_net_data.pkl'
E_fci = -1.136189454088
with open(filename, 'rb') as manage_file:
     data=pickle.load(manage_file)
##data has entries like:
#{'n':[], 'angle':[], 'energy':[], 't': [], 
#'s': [], 'ns':[], 'sangle':[], 'senergy':[], 'ts':[]}

###grab all the data
sarray=np.linspace(0, 1, data['s'])
senergy=data['senergy']
sangle=data['sangle']
ns=data['ns']


###Plots
fig=plt.figure()
fig.set_figheight(5)
fig.set_figwidth(18)
##subplot commands has arguement (row index, column index, position)

# Add energy plot on column 1
ax1 = fig.add_subplot(131)
ax1.plot(sarray, senergy, "go", ls="dashed")
#ax1.plot(sarray, np.full(ssteps, E_fci), color="red")
ax1.set_xlabel("Adiobatic step s", fontsize=13)
ax1.set_ylabel("Energy (Hartree)", fontsize=13)
ax1.text(0.0, -1.1176, r"$E_\mathrm{HF}$", fontsize=15)
ax1.text(0.2, -1.1357, r"$E_\mathrm{FCI}$", fontsize=15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

#Add angle plot on column 2
ax2 = fig.add_subplot(132)
ax2.plot(sarray, sangle, "go", ls="dashed")
ax2.set_xlabel("Adiobatic step s", fontsize=13)
ax2.set_ylabel("Gate parameter $\\theta$ (rad)", fontsize=13)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

ax3 = fig.add_subplot(133)
ax3.plot(sarray, ns, "go", ls="solid")
ax3.plot(sarray, data['n']*np.ones(len(sarray)), ls="dashed")
ax3.set_xlabel("Adiabatic step s", fontsize=13)
ax3.set_ylabel("Number of iterations, n", fontsize=13)
ax3.text(0.0, 20, r"$n_{VQE}$", fontsize=15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.subplots_adjust(wspace=0.3, bottom=0.2)
plt.show()