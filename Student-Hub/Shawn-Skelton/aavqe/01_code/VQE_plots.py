# -*- coding: utf-8 -*-
"""
Created on Tue May 30 19:08:11 2023

@author: Shawn Skelton
"""

import pickle 
import matplotlib.pyplot as plt
import numpy as np
import os.path

#filename='H_net_data.pkl'
numpoints=5
file_name='kandala_H2_'+str(numpoints)+'_iads'   


####Path info: CHANGE TO PATH LOCATION THE AAVQE FOLDER IS SAVED

data_path = 'C:/Users/Shawn Skelton/Documents/AAVQE/03_data'
figure_path= 'C:/Users/Shawn Skelton/Documents/AAVQE/02_figures'

data_name=os.path.join(data_path, file_name+'.pkl')   
figure_name = os.path.join(figure_path, file_name+'.pdf')    

with open(data_name, 'rb') as manage_file:
     data=pickle.load(manage_file)
     
##data has entries like:
#{'n':[], 'angle':[], 'energy':[], 't': [], 
#'s': [], 'ns':[], 'sangle':[], 'senergy':[], 'ts':[]}

def HEVCE_TO_VQE_PLOTS(data, ifsave=False, figname=figure_name):
    kenergy=data['kenergy']
    kits=data['kits']
    ktimes=data['ktimes']
    kls="dashed"
    kms="."
    ck='tab:blue'
    
    energy=data['energy']
    its=data['its']
    times=data['times']
    vls="solid"
    vms="+"
    cv='tab:orange'
    
    GSE=data['GSE']
    iad_array=data['interatom_d']
    ctol=data['conv_tol']
    
    fig=plt.figure()
    fig.set_figheight(5)
    fig.set_figwidth(18)
    
    ax1=fig.add_subplot(131)
    
    ax1.plot(iad_array, GSE, marker="p",color='k', ls='dotted', label='GSE')
    
    ax1.plot(iad_array, kenergy, ls=kls, marker=kms,color=ck, label='HE_ansatz_E')
    ax1.errorbar(iad_array, kenergy, yerr=ctol*np.ones([5]), color=ck, xerr=None)
    
    ax1.plot(iad_array, energy,ls=vls,marker=vms,color=cv, label='VQE_E')
    ax1.errorbar(iad_array, energy, yerr=ctol*np.ones([5]), xerr=None, color=cv)
    
    ax1.legend()
    ax1.set_xlabel('inter atomic distance (A)')
    ax1.set_ylabel('Energy (Hartrees)')
    
    ax2=fig.add_subplot(132)
    ax2.plot(iad_array, kits,ls=kls, marker=kms, color=ck,label='HE_ansatz_E')
    ax2.plot(iad_array, its, ls=vls,marker=vms, color=cv,label='VQE_E')
    ax2.legend()
    ax2.set_xlabel('inter atomic distance (A)')
    ax2.set_ylabel('Its to solution')
    
    ax3=fig.add_subplot(133)
    ax3.plot(iad_array, ktimes, ls=kls, marker=kms,color=ck,label='HE_ansatz_E')
    ax3.plot(iad_array, times,ls=vls,marker=vms,color=cv, label='VQE_E')
    ax3.legend()
    ax3.set_xlabel('inter atomic distance (A)')
    ax3.set_ylabel('Time to solution')
    
    if ifsave==True:
        plt.savefig(figname)
    return
    
    

def AAVQE_to_VQE_plots(data):
    ###grab all the data
    E_fci = -1.136189454088
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
    return
    

HEVCE_TO_VQE_PLOTS(data, ifsave=True, figname=figure_name)