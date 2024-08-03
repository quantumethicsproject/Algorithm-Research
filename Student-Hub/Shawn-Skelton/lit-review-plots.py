import matplotlib.pyplot as plt
import numpy as np

'''DEFINE THE FIGURE AND DOMAIN'''
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)

import pickle
import os.path

yearlist=['2014', '2017' , '2018', '2019', '2020', '2021', '2022', '2023']
vqalist=[2, 4, 6, 13, 14, 31, 69, 106]
maxreallist=[2, 0, 0, 4, 4, 15, 26, 72]
means={'noiseless simulation': [0,10.5, 12,22, 5.875, 8.222222222, 10.81967213, 14.66666667],
       'noisy simulation': [0, 4.5, 6, 24.33333333, 5.25, 8.125, 6.8, 9.766666667],
       'quantum hardware':[2,0,0,2.8,3, 7.285714286, 7.066666667, 10]}


def NUMPUBLISHED_VS_YEAR_PLOT(yr, publist,ifsave=False):
    plt.figure()
    plt.bar(yr,publist, label='logical qubits')
    plt.ylabel('Logical qubits')
    plt.xlabel('Year')
    plt.legend()
    plt.title('Number of VQA Implementations vs Year')
    if ifsave==True:
        SAVE_PLOT('numpub_year.pdf')
    else:
        plt.show()

def MAXREAL_VS_YEAR_PLOT(yr, reallist,ifsave=False):
    plt.figure()
    plt.bar(yr,reallist, label='logical qubits')
    plt.ylabel('Logical qubits')
    plt.xlabel('Year')
    plt.legend()
    plt.title('Largest Quantum Hardware Implementation vs Year')
    if ifsave==True:
        SAVE_PLOT('maxreal_year.pdf')
    else:
        plt.show()

patterns = [ "\\" , "/", "",  "x", "o", "O", ".", "*" ]

def AVG_QUBITS_VS_YEAR_PLOT(yr, means, bar_colors = ['tab:blue', 'mediumseagreen', 'tab:orange'], ifsave=False):
    x=np.arange(len(yr))
    width=0.25
    multiplier=0

    fig, ax=plt.subplots(layout='constrained')
    
    for attribute, measure in means.items():
        offset=width*multiplier
        ax.bar(x+offset,measure, width, label= attribute, color=bar_colors[multiplier], hatch=patterns[multiplier])
        multiplier +=1
    ax.set_ylabel('Mean logical qubits')
    ax.set_xlabel('Year')
    ax.set_title('Mean Implementation size vs Year')
    ax.set_xticks(x + width, yr)
    plt.legend()
    if ifsave==True:
        SAVE_PLOT('meanimplementation_year.pdf')
    else:
        plt.show()

def SAVE_PLOT(filename, dev='mac'):
    script_path = os.path.abspath(__file__)
    if dev=='mac':
        save_path=script_path.replace(".py", "")
    else:
        save_path=script_path.replace(".py", "")
    completename = os.path.join(save_path, filename) 
    
    plt.savefig(completename)
    return
AVG_QUBITS_VS_YEAR_PLOT(yearlist, means, ifsave=True)
