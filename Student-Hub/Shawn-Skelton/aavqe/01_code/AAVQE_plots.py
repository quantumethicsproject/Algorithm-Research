import numpy as np
import matplotlib.pyplot as plt

'''DEFINE THE FIGURE AND DOMAIN'''
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
'''FANCY PREAMBLE TO MAKE BRAKET PACKAGE WORK NICELY'''
#plt.rc('text', usetex=True)
#plt.rc('text.latex', preamble=r'\usepackage{braket}')
import pickle
import os.path

numpoints=8
qubitrange=np.array([3, 4, 5, 6, 7, 8])
models=["bitflippenny=0.05", 'nonoise', "FakeManila"]
bdl_array=np.linspace(-1, 1, numpoints) #should remove
c_tol=1.6*10**(-3)
dev='mac'

def DATA_EXTRACT(NMODEL="bitflipcirq=0.05",HNAME='XX3', numpoints=6):
    filename='AAVQE_w_'+NMODEL+'_'+HNAME+'_'+str(numpoints)+'_iads.pkl'
    script_path = os.path.abspath(__file__)
    if dev=='mac':
        save_path=script_path.replace("01_code/AAVQE_plots.py", "03_data")
    else:
        save_path=script_path.replace("01_code\AAVQE_plots.py", "03_data")
    completename = os.path.join(save_path, filename)
    
    with open(completename,'rb') as file:
        DATA=pickle.load(file)

    return DATA

def CONV_TEST(E,c_tol=1.6*10**(-3), token_val=1):
    token=0
    nlastind=np.int64(len(E))
    
    if abs(np.diff(E))[-1]<=1.6*10**(-3):
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
    
    AAE=bdict['sdata']['fullenergy']
    AAn=bdict['sdata']['fulln']

    if noisy==False:
        #kE=bdict['kdata']['energies']
        #kn=bdict['kdata']['its']
        return gsE,  AAE,  AAn

    kNE=bdict['Nkdata']['energies']
    kNn=bdict['Nkdata']['its']
    NAAE=bdict['Nsdata']['fullenergy']
    NAAn=bdict['Nsdata']['fulln']

    return gsE, kNE, kNn, AAE, NAAE, NAAn

def EXTRACT_ITERATIONS(data, bdl, noisy=True,getnoisykandala=False):
    bdictname='b_'+str(np.around(bdl))+'_data'
    bdict=data[bdictname]
    gsE=bdict['gsE']
    
    if getnoisykandala==True:
        AAE=bdict['Nkdata']['energies']
        AAn=bdict['Nkdata']['its']
    else:
        AAE=bdict['sdata']['fullenergy']
        AAn=bdict['sdata']['fulln']

    if noisy==False:
        return gsE, np.ones(len(kE)), kE,  np.zeros(len(AAE)),AAE, kn, AAn

    NAAE=bdict['Nsdata']['fullenergy']
    NAAn=bdict['Nsdata']['fulln']

    return gsE,  AAE,  AAn, NAAE, NAAn


def GET_b_DATA(DATA, bdl):
    bdictname='b_'+str(np.around(bdl))+'_data'
    bdict=DATA[bdictname]
    return bdict

#data=DATA_EXTRACT('bitflippenny=0.05')
#bdata=GET_b_DATA(data, 1.0)
#print(bdata['Nsdata'].keys())
def GET_PLOT1(guessqubits, barray=bdl_array, NMODELS=["bitflippenny=0.05", 'nonoise', "FakeManila"],  numpoints=6, HPREF='XX',ifsave=False, bar_labels = ['red'], bar_colors = ['tab:red']):
    best_instances=np.zeros([len(guessqubits)])

    for m, qubits in enumerate(guessqubits):
        if NMODELS[m]=='nonoise':
            Z=Z_FCN(np.array([qubits]),barray, NMODELS[m], numpoints=numpoints,HPREF=HPREF, noisy=False)
        else:
            Z=Z_FCN_BEST(np.array([qubits]),barray, NMODELS[m], numpoints,HPREF=HPREF, noisy=True)
        
        if len(Z[np.where(Z==1)])==0:
            print('Warning, no AAVQE best solution found for this noise model and'+str(qubits)+' qubits')
    
    plt.bar(NMODELS, guessqubits, label=bar_labels, color=bar_colors)       
    plt.ylabel('number of qubits')
    plt.xlabel('noise models')
    
    if ifsave==True:
        SAVE_PLOT('AAVQE_max_implementation.pdf')
    plt.show()

def GET_PLOT2(guessqubit, barray=bdl_array, NMODEL="bitflipcirq=0.05",  numpoints=6, HPREF='XX',ifsave=False):
    bar_colors = ['tab:blue', 'mediumseagreen', 'tab:orange']
    Z=Z_FCN_BEST(np.array([guessqubit]),barray, NMODEL, numpoints, HPREF)
    if len(Z[np.where(Z==1)])==0:
        print('Warning, no AAVQE best solution found for this noise model and'+str(guessqubit)+' qubits')
    else:
      qinds, binds=(np.where(Z==1))
      qind=qinds[-1]
      bind=binds[-1]
    DATA=DATA_EXTRACT(NMODEL,HPREF+str(guessqubit), numpoints)
    
    binstdict=GET_b_DATA(DATA, barray[bind])
    
    NSDATA=binstdict['Nsdata']
    n=NSDATA['fulln']
    E=NSDATA['fullenergy']
    x=np.linspace(0, n, n)
    
    gsE=binstdict['gsE']
    Edist=gsE*np.ones(len(E))-E

    print('GSE', gsE)
    print('energy array', E)
    fig, ax=plt.subplots(layout='constrained')
    ax.plot(x, E, label=r'$C(\theta_j)$', color=bar_colors[0])
    #ax.plot(x,E , label=r'AAVQE $\braket{E}$')
    ax.plot(x, Edist, label=r'Error in $C(\theta_j)$', linestyle='dashed', color=bar_colors[1])
    plt.axhline(y=gsE,xmin=0,xmax=3,c=bar_colors[2],linewidth=1,zorder=0, label=r"$C(\theta_{ideal})$")
    #plt.axhline(y=gsE,xmin=0,xmax=3,c="blue",linewidth=1,zorder=0, label=r"$\braket{E}_{real}$")
    
    #plt.xlabel('AAVQE iteration')
    #plt.ylabel('Energy (Hartrees)')
    ax.set_ylabel(r'Cost function $C(\theta_j)$')
    ax.set_xlabel('AAVQE'+' iteration step j')
    #plt.title('AAVQE convergence for '+str(guessqubit)+' qubits and bond length '+str(barray[bind]))
    plt.title('AAVQE'+' convergence with FakeManila noise')
    ax.legend()
    plt.title('AAVQE'+' convergence with FakeManila noise')
    if ifsave==True:
        SAVE_PLOT('AAVQE_fakemanila_convergence.pdf')
    plt.show()

def SAVE_PLOT(filename):
    script_path = os.path.abspath(__file__)
    if dev=='mac':
        save_path=script_path.replace("01_code/aavqe_plots.py", "02_figures")
    else:
        save_path=script_path.replace("01_code\AAVQE_plots.py", "02_figures")
    completename = os.path.join(save_path, filename) 
    plt.savefig(completename)
    print(completename)

def Z_FCN(qubitrange,barray, NMODEL, numpoints, HPREF='XX',  noisy=True):
    Z=np.zeros([  len(qubitrange), len(barray)])
    for q, qubit in enumerate(qubitrange):
        HNAME=HPREF+str(int(qubit))
        
        data=DATA_EXTRACT(NMODEL,HNAME, numpoints)
        for b, bdl in enumerate(barray):
            args=EXTRACT_ENERGIES(data, bdl, noisy=False)
            
            Z[ q, b]=SUCC_TEST(args[1],args[0], token_val=1)
            print(args[1][-1],args[0])
            #print(args[2], SUCC_TEST(args[1],args[0], token_val=1))

    return Z

def Z_FCN_BEST(qubitrange,barray,  NMODEL, numpoints, HPREF='XX', noisy=True):
    Z=np.ones([ len(qubitrange), len(barray)])
    for q, qubit in enumerate(qubitrange):
        HNAME=HPREF+str(int(qubit))
        
        data=DATA_EXTRACT(NMODEL,HNAME, numpoints)
        for b, bdl in enumerate(barray):
            if noisy==True:
                gsE, nk, nkn, aa, naa, sn=EXTRACT_ENERGIES(data, bdl, noisy)
            else:
                gsE,  naa, sn=EXTRACT_ENERGIES(data, bdl, noisy)
                nk=np.zeros(len(naa))
                nkn=0
                
            ksucc=SUCC_TEST(nk,gsE, token_val=-1)
            
            AAVQEsucc=SUCC_TEST(naa,gsE, token_val=1)
            #print(sn, AAVQEsucc)
            if ksucc==-1 and nkn<=sn:
                Z[q, b]=ksucc
            elif ksucc==-1 and AAVQEsucc==1:
                Z[q, b]=AAVQEsucc
                print('both but AAVQE better')
            elif AAVQEsucc==1:
                Z[q, b]=AAVQEsucc
            
    print(Z)
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


def CONTOUR_PLOT_BEST(qubitrange, barray=bdl_array,ctol=c_tol, NMODEL="bitflipcirq=0.05", numpoints=6, saveplot=False):
    Z=Z_FCN_BEST(qubitrange, barray, ctol, NMODEL, numpoints)
    X, Y = np.meshgrid(barray, qubitrange)
    
    fig, ax = plt.subplots()
    cs = ax.contourf(X, Y, Z, levels=[-1,  0, 1], colors=['seagreen', 'gold', 'blue'], extend='both')
    
    #cs.cmap.set_over('red')
    cs.cmap.set_under('tan')
    cs.changed()
    #ax.clabel(cs, inline=True, fontsize=10)
    ax.set_title('Contour plot of best VQE success')
    fig.colorbar(cs, ax=ax, pad=0.1, label='Name [units]')
    plt.show()


def CONTOUR_PLOT_AVG_BEST(qubitrange, barray=bdl_array,ctol=c_tol, NMODEL="bitflipcirq=0.05", numpoints=6, saveplot=False):
    Z=AVG_BEST_Z(qubitrange, barray, ctol, NMODEL, numpoints)
    print(Z)
    X, Y = np.meshgrid(barray, qubitrange)
    
    fig, ax = plt.subplots()
    
    cs = ax.contourf(X, Y, Z, levels=[-1, 0, 1], colors=['seagreen', 'tab:blue', 'tab:orange'])
    
    cs.cmap.set_over('red')
    cs.cmap.set_under('red')
    cs.changed()
    #ax.clabel(cs, inline=True, fontsize=10)
    ax.set_title('AAVQE vs VQE average relative success')
    fig.colorbar(cs, ax=ax, pad=0.2, label='Optimal strategy')
    if saveplot==True:
        SAVE_PLOT('penny_contour_3_8.pdf')
    plt.show()

def CHECK_CONSIST(qubitrange=np.array([3, 5]), barray=bdl_array,ctol=c_tol, NMODEL="bitflipcirq=0.05", numpoints=6):
    Hreflist=['XX', '1XX', '2XX', '3XX']
    for xind, X in enumerate(Hreflist):
        print(Z_FCN_BEST(qubitrange,barray,ctol, NMODEL, numpoints, HPREF=X))
    

def AVG_BEST_Z(qubitrange=np.array([3, 5]), barray=bdl_array,ctol=c_tol, NMODEL="bitflipcirq=0.05", numpoints=6, Hreflist=['0XX', '1XX', '2XX', '3XX', '4XX', '5XX', '6XX', '7XX', '8XX', '9XX']):
    
    AVG=np.zeros([len(qubitrange), len(barray)])
    for xind, X in enumerate(Hreflist):
        AVG=AVG+Z_FCN_BEST(qubitrange,barray, NMODEL, numpoints, HPREF=X)
    AVG=AVG/len(Hreflist)
    
    return AVG


def n_FCN_BEST(qubitrange,barray,  NMODEL, numpoints, HPREF='XX', noisy=True):
    iters=np.zeros([  len(qubitrange), len(barray)])
    successes=np.zeros([ len(qubitrange), len(barray)])
    Niters=np.zeros([  len(qubitrange), len(barray)])
    Nsuccesses=np.zeros([ len(qubitrange), len(barray)])
    for q, qubit in enumerate(qubitrange):
        HNAME=HPREF+str(int(qubit))
        
        data=DATA_EXTRACT(NMODEL,HNAME, numpoints)
        for b, bdl in enumerate(barray):
            gsE, AA, nAA, NAA, NnAA=EXTRACT_ITERATIONS(data, bdl, noisy)
            
            successes[q, b]=SUCC_TEST(AA, gsE, token_val=1)
            Nsuccesses[q, b]=SUCC_TEST(NAA,gsE, token_val=1)
            
            iters[q, b]=nAA
            Niters[q, b]=NnAA

    return successes, iters, Nsuccesses, Niters

def AVG_COMP_n(qubitrange=np.array([3, 5]), barray=bdl_array,ctol=c_tol, NMODEL="bitflipcirq=0.05",numpoints=6, Hreflist=['0XX', '1XX', '2XX', '3XX', '4XX']):
    AVG=np.zeros([len(qubitrange), len(barray)]), np.zeros([len(qubitrange), len(barray)]), np.zeros([len(qubitrange), len(barray)]), np.zeros([len(qubitrange), len(barray)])
    
    for xind, X in enumerate(Hreflist):
        #print('this is instance', X)
        AVG=tuple(map(lambda i, j: i + j, AVG, n_FCN_BEST(qubitrange,barray, NMODEL, numpoints, HPREF=X)))
        #AVG=AVG+n_FCN_BEST(qubitrange,barray, NMODEL, numpoints, HPREF=X)
        #print(n_FCN_BEST(qubitrange,barray, NMODEL, numpoints, HPREF=X))
    
    return AVG[0], AVG[1]/len(Hreflist), AVG[2], AVG[3]/len(Hreflist)

def GET_pt_AVG(qubit, bdl,Hreflist, ctol=c_tol, NMODEL="bitflipcirq=0.05",numpoints=6):
    
    kiters=np.zeros([len(Hreflist)])
    ksuccesses=np.zeros([ len(Hreflist)])
    AAiters=np.zeros([len(Hreflist)])
    AAsuccesses=np.zeros([len(Hreflist)])
    
    for xind, X in enumerate(Hreflist):
        gsE,  AAE,  kiters[xind], NAAE, AAiters[xind]=EXTRACT_ITERATIONS(data, bdl, noisy, getnoisykandala=True)
        ksuccesses=SUCC_TEST(AAE, gsE, token_val=1)
        AAsuccesses=SUCC_TEST(NAAE, gsE, token_val=1)
    Nkstd=np.std(kiters)
    Nkavgn=np.mean(kiters)
    NAAstd=np.std(AAiters)
    NAAavgn=np.mean(AAiters)

    return Nkavgn, Nkstd, NAAavgn, NAAstd, ksuccesses, AAsuccesses 

def GET_qubit_AVG(qubit, barray,Hreflist, ctol=c_tol, NMODEL="bitflippenny=0.05",numpoints=6):
    kiters=np.zeros([len(Hreflist)+len(bdl_array)])
    ksuccesses=np.zeros([ len(Hreflist)+len(bdl_array)])
    AAiters=np.zeros([len(Hreflist)+len(bdl_array)])
    AAsuccesses=np.zeros([len(Hreflist)+len(bdl_array)])
    
    for bind, bdl in enumerate(barray):
        for xind, X in enumerate(Hreflist):
            data=DATA_EXTRACT(NMODEL,X+str(qubit), numpoints)
            gsE,  AAE,  kiters[bind+xind], NAAE, AAiters[bind+xind]=EXTRACT_ITERATIONS(data, bdl, getnoisykandala=True)
            ksuccesses[bind+xind]=SUCC_TEST(AAE, gsE, token_val=1)
            
            AAsuccesses[bind+xind]=SUCC_TEST(NAAE, gsE, token_val=1)
            
    Nkstd=np.std(kiters)
    Nkavgn=np.mean(kiters)
    NAAstd=np.std(AAiters)
    NAAavgn=np.mean(AAiters)

    return Nkavgn, Nkstd, NAAavgn, NAAstd, ksuccesses, AAsuccesses 

def PLOT_AVG_ITER(qubitrange, barray=bdl_array,ctol=c_tol, NMODEL="bitflipcirq=0.05", numpoints=6, ifsucc=False):
    sols, AVGn, Nsols, NAVGn =AVG_COMP_n(qubitrange, barray, ctol, NMODEL, numpoints)
    #X, Y = np.meshgrid(barray, qubitrange)
    if ifsucc==True:
        AVGn[np.where(sols==0)]=0
        NAVGn[np.where(Nsols==0)]=0
    fig, ax = plt.subplots()
    ax.scatter(qubitrange, np.sum(NAVGn, axis=1)/len(bdl_array),marker=7, color='seagreen', label='Noisy average iterations')
    ax.scatter(qubitrange, np.sum(AVGn, axis=1)/len(bdl_array),marker=5,color= 'gold', label='Noiseless average iterations')
    #cs = ax.contourf(X, Y, Z, levels=[0,  1, 2], colors=['seagreen', 'gold', 'blue'], extend='both')
    
    ax.set_title('')
    plt.show()

def PLOT_AVG_SUCC(qubitrange, barray=bdl_array,ctol=c_tol, NMODEL="bitflipcirq=0.05", numpoints=6):
    sols, AVGn, Nsols, NAVGn =AVG_COMP_n(qubitrange, barray, ctol, NMODEL, numpoints, Hreflist=['0XX', '1XX', '2XX', '3XX', '4XX', '5XX', '7XX', '8XX', '9XX'])
    numsucc=np.sum(sols, axis=1)
    Nnumsucc=np.sum(Nsols, axis=1)

    fig, ax = plt.subplots()

    ax.plot(qubitrange, Nnumsucc/len(bdl_array),marker=7, color='seagreen', label='Noisy average iterations')
    
    ax.plot(qubitrange, numsucc/len(bdl_array),marker=5,color= 'gold', label='Noiseless average iterations')

    ax.set_title('Qubit number vs ratio of success')
    plt.show()

def PLOT_AVG_ITERS_W_STD(qubitrange, barray=bdl_array,ctol=c_tol, NMODEL="bitflippenny=0.05", numpoints=6, Hreflist=['0XX', '1XX', '2XX', '3XX', '4XX', '5XX', '6XX', '7XX', '8XX', '9XX']):
    fig=plt.figure()
    Nkavgnlist=np.zeros(len(qubitrange))
    Nkstdlist=np.zeros(len(qubitrange))
    NAAavgnlist=np.zeros(len(qubitrange))
    NAAstdlist=np.zeros(len(qubitrange))
    ksuccesseslist=np.zeros(len(qubitrange))
    AAsuccesseslist=np.zeros(len(qubitrange))
    bar_labels = ['VQE', 'AAVQE']
    centering_help=0.25*np.ones(len(qubitrange))
    bar_colors = ['tab:blue', 'tab:orange']
    for q, qubit in enumerate(qubitrange):
        
        Nkavgnlist[q], Nkstd, NAAavgn, NAAstd, ksuccesses, AAsuccesses =GET_qubit_AVG(qubit, barray, Hreflist, numpoints=numpoints)
        
        Nkstdlist[q]=Nkstd
        NAAavgnlist[q]=NAAavgn
        NAAstdlist[q]=NAAstd
        ksuccesseslist[q]=np.sum([ksuccesses])/len(ksuccesses)
        AAsuccesseslist[q]=np.sum([AAsuccesses])/len(AAsuccesses)
        
        probVQE=np.around(100*np.sum([ksuccesses])/len(ksuccesses),4)
        probAAVQE=np.around(100*np.sum([AAsuccesses])/len(AAsuccesses),4)
        print('for '+str(qubit)+ 'qubits, the VQE convergence probability is' + str(probVQE) )
        
        print('for '+str(qubit)+ 'qubits, the AAVQE convergence probability is' + str(probAAVQE) )
        #if probVQE>probAAVQE:
        plt.text(qubit-centering_help[q], Nkavgnlist[q]+5, r'$\pm$'+str(np.around(Nkstd, 2)))
        #else:
        plt.text(qubit+centering_help[q], NAAavgn+5,r'$\pm$'+str(np.around(NAAstd, 2)))

    #plt.ylim(0, max(max(NAAavgnlist), max(Nkavgnlist))+10)
    #plt.xlim(qubitrange[0], qubitrange[-1]+1)
    plt.bar(qubitrange-centering_help, Nkavgnlist, label=bar_labels[0], color=bar_colors[0], width=0.4)
    #plt.errorbar(qubitrange, Nkavgnlist, yerr=Nkstdlist, label='VQE', color=bar_colors[0])
    plt.bar(qubitrange+centering_help, NAAavgnlist, label=bar_labels[1], color=bar_colors[1], width=0.4)
    #plt.errorbar(qubitrange, NAAavgnlist, NAAstdlist, label='AAVQE', color=bar_colors[1])
    
    plt.ylabel('average number of iterations to convergence')
    plt.xlabel('number of qubits')
    plt.legend()
    plt.show()
    #SAVE_PLOT('aavqe_avg_iters.pdf')

def PLOT_AVG_SUCC(qubitrange, barray=bdl_array,ctol=c_tol, NMODEL="bitflippenny=0.05", numpoints=6):
    Hreflist=['0XX', '1XX', '2XX', '3XX', '4XX', '5XX', '6XX', '7XX', '8XX', '9XX']
    fig=plt.figure()
    Nkavgnlist=np.zeros(len(qubitrange))
    Nkstdlist=np.zeros(len(qubitrange))
    NAAavgnlist=np.zeros(len(qubitrange))
    NAAstdlist=np.zeros(len(qubitrange))
    probVQE=np.zeros(len(qubitrange))
    probAAVQE=np.zeros(len(qubitrange))
    bar_labels = ['VQE', 'AAVQE']
    patterns = [ "\\" , "",  "/",]
    centering_help=0.25*np.ones(len(qubitrange))
    bar_colors = ['tab:blue', 'tab:orange']
    for q, qubit in enumerate(qubitrange):
        Nkavgnlist[q], Nkstd, NAAavgn, NAAstd, ksuccesses, AAsuccesses =GET_qubit_AVG(qubit, barray, Hreflist, numpoints=numpoints)
        
        Nkstdlist[q]=Nkstd
        NAAavgnlist[q]=NAAavgn
        NAAstdlist[q]=NAAstd
        ksuccesseslist=np.sum([ksuccesses])/len(ksuccesses)
        AAsuccesseslist=np.sum([AAsuccesses])/len(AAsuccesses)
        
        probVQE[q]=np.around(100*np.sum([ksuccesses])/len(ksuccesses),4)
        probAAVQE[q]=np.around(100*np.sum([AAsuccesses])/len(AAsuccesses),4)

    #plt.ylim(0,1)
    plt.xlim(qubitrange[0]-1, qubitrange[-1]+1)
    plt.bar(qubitrange-centering_help, probVQE, label=bar_labels[0], color=bar_colors[0], width=0.4, hatch=patterns[0])
    #plt.errorbar(qubitrange, Nkavgnlist, yerr=Nkstdlist, label='VQE', color=bar_colors[0])
    plt.bar(qubitrange+centering_help, probAAVQE, label=bar_labels[1], color=bar_colors[1], width=0.4,  hatch=patterns[1])
    #plt.errorbar(qubitrange, NAAavgnlist, NAAstdlist, label='AAVQE', color=bar_colors[1])
    plt.ylabel('Average success probability')
    plt.xlabel('Number of qubits')
    plt.legend()
    SAVE_PLOT('aavqe_avg_succ.pdf')


#PLOT_AVG_ITERS_W_STD(np.array([3, 4, 5, 6, 7, 8]), numpoints=8)

#GET_PLOT1(np.array([100]), NMODELS=["FakeManila"],barray=bdl_array,  numpoints=8, HPREF='0XX', ifsave=False,  bar_labels = ['red'], bar_colors = ['tab:red'])
#
#GET_PLOT2(100, NMODEL="FakeManila",  numpoints=8, HPREF='0XX')
#print(DATA_EXTRACT(NMODEL="FakeManila",HNAME='0XX100', numpoints=8).keys())
#
D=DATA_EXTRACT(NMODEL="FakeManila",HNAME='0XX100', numpoints=8)
bdict=D['b_-1.0_data']
print(bdict['Nsdata'].keys())
print(bdict['Nsdata']['fulln'])
#print(bdict['Nsdata']['fullenergy'])
print(bdict['Nsdata']['fulltimes'])
#PLOT_AVG_SUCC(np.array([3, 4, 5, 6, 7, 8]), numpoints=8)
#CONTOUR_PLOT_AVG_BEST(qubitrange=np.array([3, 4, 5, 6, 7, 8]),NMODEL="bitflippenny=0.05",numpoints=8, saveplot=True)
#plt.show()
