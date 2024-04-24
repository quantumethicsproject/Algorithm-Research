import numpy as np
import matplotlib.pyplot as plt
'''FANCY PREAMBLE TO MAKE BRAKET PACKAGE WORK NICELY'''
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{braket}')
import pickle
import os.path

numpoints=6

qubitrange=np.array([3, 5, 7])
models=["bitflipcirq=0.05", "depolcirq=0.05", "bitflipibm=0.05"]
bdl_array=np.linspace(-1, 1, numpoints)
c_tol=1.6*10**(-3)

def DATA_EXTRACT(NMODEL="bitflipcirq=0.05",HNAME='XX3', numpoints=6):
    filename='AAVQE_w_'+NMODEL+'_'+HNAME+'_'+str(numpoints)+'_iads.pkl'
    script_path = os.path.abspath(__file__)
    save_path=script_path.replace("01_code\AAVQE_plots.py", "03_data")
    completename = os.path.join(save_path, filename) 
    
    with open(completename,'rb') as file:
        DATA=pickle.load(file)

    return DATA

def CONV_TEST(E,ctol=c_tol, token_val=1):
    token=0
    if abs(E[-1]-E[-2])<=ctol:
        token=token_val
    return token

def SUCC_TEST(E,gsE, ctol=c_tol, token_val=1):
    token=0
    if abs(E[-1]-gsE)<=ctol:
        token=token_val
    return token

def EXTRACT_ENERGIES(data, bdl, noisy=True):
    bdictname='b_'+str(np.around(bdl))+'_data'
    bdict=data[bdictname]
    gsE=bdict['gsE']
    kE=bdict['kdata']['energies']
    kn=bdict['kdata']['its']
    
    AAE=bdict['sdata']['fullenergy']
    AAn=bdict['sdata']['fulln']

    if noisy==False:
        return gsE, np.ones(len(kE)), kE,  np.zeros(len(AAE)),AAE, kn, AAn

    kNE=bdict['Nkdata']['energies']
    NAAE=bdict['Nsdata']['fullenergy']

    return gsE, kE, kNE, AAE, NAAE, kn, AAn

def GET_b_DATA(DATA, bdl):
    bdictname='b_'+str(np.around(bdl))+'_data'
    bdict=DATA[bdictname]
    return bdict


def GET_PLOT1(guessqubits, barray=bdl_array, NMODELS=["bitflipcirq=0.05", 'nonoise', "bitflippenny=0.05"],  numpoints=6, HPREF='XX',ifsave=False):
    best_instances=np.zeros([len(guessqubits)])
    noiseparam=[True, False, True]
    bar_labels = ['red', 'blue', '_red']
    bar_colors = ['tab:red', 'tab:blue', 'tab:red']

    for m, qubits in enumerate(guessqubits):
        Z=Z_FCN_BEST(np.array([qubits]),barray, NMODELS[m], numpoints, noisy=noiseparam[m])
        print(Z)
        if len(Z[np.where(Z==1)])==0:
            print('Warning, no AAVQE best solution found for this noise model and'+str(qubits)+' qubits')
    
    plt.bar(NMODELS, guessqubits, label=bar_labels, color=bar_colors)       
    plt.ylabel('number of qubits')
    plt.xlabel('noise models')
    
    if ifsave==True:
        SAVE_PLOT('AAVQE_figure1.pdf')
    plt.show()


def GET_PLOT2(guessqubit, barray=bdl_array, NMODEL="bitflipcirq=0.05",  numpoints=6, HPREF='XX',ifsave=False):
    Z=Z_FCN_BEST(np.array([guessqubit]),barray, NMODEL, numpoints)
    if len(Z[np.where(Z==1)])==0:
        print('Warning, no AAVQE best solution found for this noise model and'+str(qubits)+' qubits')
    else:
      qinds, binds=(np.where(Z==1))
      qind=qinds[-1]
      bind=binds[-1]
    DATA=DATA_EXTRACT("bitflipcirq=0.05",HPREF+str(guessqubit), numpoints)
    
    binstdict=GET_b_DATA(DATA, barray[bind])
    
    NSDATA=binstdict['Nsdata']
    n=NSDATA['fulln']
    E=NSDATA['fullenergy']
    x=np.linspace(0, n, n)
   
    gsE=binstdict['gsE']
    Edist=gsE*np.ones(len(E))-E
    plt.plot(x,E , label=r'AAVQE $\braket{E}$')
    plt.plot(x, Edist, label=r'error in $\braket{E}$', linestyle='dashed')
    plt.axhline(y=gsE,xmin=0,xmax=3,c="blue",linewidth=1,zorder=0, label=r"$\braket{E}_{real}$")
    plt.legend()
    plt.xlabel('AAVQE iteration')
    plt.ylabel('Energy (Hartrees)')
    plt.title('AAVQE convergence for '+str(guessqubit)+' qubits and bond length '+str(barray[bind]))

    if ifsave==True:
        SAVE_PLOT('AAVQE_figure2_'+NMODEL+'.pdf')
    plt.show()

def SAVE_PLOT(filename):
    script_path = os.path.abspath(__file__)
    save_path=script_path.replace("01_code\AAVQE_plots.py", "02_figures")
    completename = os.path.join(save_path, filename) 
    plt.savefig(completename)
    print(completename)


def Z_FCN(qubitrange,barray, NMODEL, numpoints, HPREF='XX'):
    Z=np.zeros([  len(qubitrange), len(barray)])
    for q, qubit in enumerate(qubitrange):
        HNAME=HPREF+str(int(qubit))
        
        data=DATA_EXTRACT(NMODEL,HNAME, numpoints)
        for b, bdl in enumerate(barray):
            gsE, k, nk, aa, naa, kn, sn=EXTRACT_ENERGIES(data, bdl)
            
            Z[ q, b]=SUCC_TEST(nk,gsE, token_val=2)
            #if CONV_TEST(naa)==1: 
            Z[q, b]=Z[q, b]+SUCC_TEST(naa, gsE, token_val=1)

    return Z
def Z_FCN_BEST(qubitrange,barray,  NMODEL, numpoints, HPREF='XX', noisy=True):
    Z=np.zeros([  len(qubitrange), len(barray)])
    for q, qubit in enumerate(qubitrange):
        HNAME=HPREF+str(int(qubit))
        
        data=DATA_EXTRACT(NMODEL,HNAME, numpoints)
        for b, bdl in enumerate(barray):
            gsE, k, nk, aa, naa, kn, sn=EXTRACT_ENERGIES(data, bdl, noisy)
            
            ksucc=SUCC_TEST(nk,gsE, token_val=2)
            AAVQEsucc=SUCC_TEST(naa,gsE, token_val=1)
            
            if ksucc==2 and kn<=sn:
                Z[q, b]=ksucc
            elif ksucc==2 and AAVQEsucc==1:
                Z[q, b]=AAVQEsucc
                print('both but AAVQE better')
            elif AAVQEsucc==1:
                Z[q, b]=AAVQEsucc

    return Z
def CONTOUR_PLOT(qubitrange, barray=bdl_array,ctol=c_tol, NMODEL="bitflipcirq=0.05", numpoints=6):
    Z=Z_FCN(qubitrange, barray, ctol, NMODEL, numpoints)
    X, Y = np.meshgrid(barray, qubitrange)
    
    fig, ax = plt.subplots()
    cs = ax.contourf(X, Y, Z, levels=[0,  1, 2, 3], colors=['seagreen', 'gold', 'blue','lightblue'], extend='both')
    
    #cs.cmap.set_over('red')
    cs.cmap.set_under('tan')
    cs.changed()
    #ax.clabel(cs, inline=True, fontsize=10)
    ax.set_title('Contour plot of noisy VQE success')
    fig.colorbar(cs, ax=ax, pad=0.1, label='Name [units]')
    plt.show()

    print(Z)

def CONTOUR_PLOT_BEST(qubitrange, barray=bdl_array,ctol=c_tol, NMODEL="bitflipcirq=0.05", numpoints=6):
    Z=Z_FCN_BEST(qubitrange, barray, ctol, NMODEL, numpoints)
    X, Y = np.meshgrid(barray, qubitrange)
    
    fig, ax = plt.subplots()
    cs = ax.contourf(X, Y, Z, levels=[0,  1, 2], colors=['seagreen', 'gold', 'blue'], extend='both')
    
    #cs.cmap.set_over('red')
    cs.cmap.set_under('tan')
    cs.changed()
    #ax.clabel(cs, inline=True, fontsize=10)
    ax.set_title('Contour plot of best VQE success')
    fig.colorbar(cs, ax=ax, pad=0.1, label='Name [units]')
    plt.show()

    print(Z)

def CONTOUR_PLOT_AVG_BEST(qubitrange, barray=bdl_array,ctol=c_tol, NMODEL="bitflipcirq=0.05", numpoints=6):
    Z=AVG_BEST_Z(qubitrange, barray, ctol, NMODEL, numpoints)
    X, Y = np.meshgrid(barray, qubitrange)
    
    fig, ax = plt.subplots()
    print(Z)
    cs = ax.contourf(X, Y, Z, levels=[0,  1, 2], colors=['seagreen', 'gold', 'blue'], extend='both')
    
    #cs.cmap.set_over('red')
    cs.cmap.set_under('tan')
    cs.changed()
    #ax.clabel(cs, inline=True, fontsize=10)
    ax.set_title('Contour plot of best VQE success')
    fig.colorbar(cs, ax=ax, pad=0.1, label='Name [units]')
    plt.show()

def CHECK_CONSIST(qubitrange=np.array([3, 5]), barray=bdl_array,ctol=c_tol, NMODEL="bitflipcirq=0.05", numpoints=6):
    Hreflist=['XX', '1XX', '2XX', '3XX']
    for xind, X in enumerate(Hreflist):
        print(Z_FCN_BEST(qubitrange,barray,ctol, NMODEL, numpoints, HPREF=X))
    

def AVG_BEST_Z(qubitrange=np.array([3, 5]), barray=bdl_array,ctol=c_tol, NMODEL="bitflipcirq=0.05", numpoints=6):
    Hreflist=['XX', '1XX', '2XX', '3XX']
    AVG=np.zeros([len(qubitrange), len(barray)])
    for xind, X in enumerate(Hreflist):
        AVG=AVG+Z_FCN_BEST(qubitrange,barray, NMODEL, numpoints, HPREF=X)
    AVG=AVG/len(Hreflist)
    
    return AVG

#CONTOUR_PLOT_AVG_BEST(qubitrange=np.array([3, 5]), NMODEL="bitflippenny=0.05")
#CHECK_CONSIST()
GET_PLOT2(8, ifsave=True)