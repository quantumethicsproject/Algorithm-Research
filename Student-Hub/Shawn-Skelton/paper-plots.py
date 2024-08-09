import matplotlib.pyplot as plt
import numpy as np

'''DEFINE THE FIGURE AND DOMAIN'''
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)

import pickle
import os.path
import json 
patterns = [ "\\" , "/", ""]
bar_colors = ['tab:blue', 'mediumseagreen', 'tab:orange']
noiselist=['noiseless', "bit-flip", "FakeManila"]
aavqelist=[12,8, 3]
vqsdlist=[16, 4, 0]
vqlslist=[5, 5, 0]
qabomlist=[8, 8, 4]


def FIGURE1(nmodels, qubits,vqaname='AAVQE', ifsave=False):
    fig, ax=plt.subplots(layout='constrained')
    ax.bar(nmodels,qubits,   color=bar_colors, hatch=patterns)
    ax.set_ylabel('Qubits')
    ax.set_xlabel('Noise models')
    plt.title(vqaname+' largest instances')

    if ifsave==True:
        filename=vqaname+'_max_implementation.pdf'
        SAVE_PLOT(filename)
    else:
        plt.show()


def FIGURE2(n,cfvals, exactval=0, vqaname='AAVQE', witherror=False, ifsave=False):
    itererror=exactval*np.ones(len(n))-cfvals
    
    fig, ax=plt.subplots(layout='constrained')
    ax.plot(n, cfvals, label=r'$C(\theta_j)$', color=bar_colors[0])
    if witherror==True:
        ax.plot(n, itererror, label=r'Error in $C(\theta_j)$', linestyle='dashed', color=bar_colors[1])
        plt.axhline(y=exactval,xmin=0,xmax=3,c=bar_colors[2],linewidth=1,zorder=0, label=r"$C(\theta_{ideal})$")
    ax.set_ylabel(r'Cost function $C(\theta_j)$')
    ax.set_xlabel(vqaname+' iteration step j')
    ax.legend()
    plt.title(vqaname+' convergence with FakeManila noise')
    if ifsave==True:
        filename=vqaname+'_fakemanila_convergence.pdf'
        SAVE_PLOT(filename)
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

def RUN_FIGURE2_VQSD(ifsave=True):
    vqaname='VQSD'
    f=open('VQSD_2_FakeManila_2.json')
    data=json.load(f)
    cfvals=data['cost_history']
    num_its=data['Iterations to Solution']
    n=np.linspace(0, num_its-1, num_its)

    fig, ax=plt.subplots(layout='constrained')
    ax.plot(n, cfvals, label=r'$C(\theta_j)$', color=bar_colors[0])
    ax.plot(n, data['error_history'], label=r'Error in $C(\theta_j)$', linestyle='dashed', color=bar_colors[1])
    ax.set_ylabel(r'Cost function $C(\theta_j)$')
    ax.set_xlabel(vqaname+' iteration step j')
    ax.legend()
    plt.title(vqaname+' convergence with FakeManila noise')
    if ifsave==True:
        filename=vqaname+'_fakemanila_convergence.pdf'
        SAVE_PLOT(filename)
    else:
        plt.show()
RUN_FIGURE2_VQSD(ifsave=True)
def RUN_FIGURE2_VQLS(ifsave=True):
    vqaname='VQLS'
    f=open('benchmarking_data.json')
    data=json.load(f)
    cfvals=data["FakeManila"]['cost_history']
    
    num_its=len(cfvals)
    n=np.linspace(0, num_its-1, num_its)
    
    fig, ax=plt.subplots(layout='constrained')
    ax.plot(n, cfvals, label=r'$C(\theta_j)$', color=bar_colors[0])
    #ax.plot(n, data["FakeManila"]['error_history'], label=r'Error in $C(\theta_j)$', linestyle='dashed', color=bar_colors[1])
    ax.set_ylabel(r'Cost function $C(\theta_j)$')
    ax.set_xlabel(vqaname+' iteration step j')
    ax.set_ylim([-0.01, 0.1])
    ax.legend()
    plt.title(vqaname+' convergence with FakeManila noise')
    if ifsave==True:
        filename=vqaname+'_fakemanila_convergence.pdf'
        SAVE_PLOT(filename)
    else:
        plt.show()



def RUN_FIGURE2_VQSD(ifsave=True):
    f=open('VQSD_2_FakeManila_2.json')
    data=json.load(f)
    cfvals=data['cost_history']
    num_its=data['Iterations to Solution']
    n=np.linspace(0, num_its-1, num_its)

    fig, ax=plt.subplots(layout='constrained')
    ax.plot(n, cfvals, label=r'$C(\theta_j)$', color=bar_colors[0])
    ax.plot(n, data['error_history'], label=r'Error in $C(\theta_j)$', linestyle='dashed', color=bar_colors[1])
    ax.set_ylabel(r'Cost function $C(\theta_j)$')
    ax.set_xlabel(vqaname+' iteration step j')
    ax.legend()
    plt.title(vqaname+' convergence with FakeManila noise')
    if ifsave==True:
        filename=vqaname+'_fakemanila_convergence.pdf'
        SAVE_PLOT(filename)
    else:
        plt.show()

#FIGURE1(noiselist, aavqelist, "AAVQE", ifsave=True)
RUN_FIGURE2_VQLS(ifsave=True)

def RUN_FIGURE2_QABoM(ifsave=False):
    ####get the helliger distance which is the global cost function###
    ##for now, I just copied the first array from the hellinger_results1.txt file
    cfvals=[
    -0.04925354247836494,
    -0.05990846029917061,
    -0.07085884902386204,
    -0.08210223566181524,
    -0.09362629559488647,
    -0.10529141652979897,
    -0.11735486653109302,
    -0.1297580319282278,
    -0.14251892317895223,
    -0.15549970578798178,
    -0.168812977728585,
    -0.18244713627225623,
    -0.19644882934071856,
    -0.21085603499743677,
    -0.22560017915954195,
    -0.2407378682599007,
    -0.2562977284879892,
    -0.27211368532906627,
    -0.2884061091152304,
    -0.30497099372546693,
    -0.3219800796526684,
    -0.3394822377069479,
    -0.357205118637376,
    -0.3754750827998336,
    -0.3942746957768317,
    -0.4133417653122919,
    -0.4329837803264143,
    -0.45309549953482414,
    -0.47381307407341866,
    -0.4949729102631002,
    -0.516612351134909,
    -0.5386424103964682,
    -0.5612354249033865,
    -0.5844014469583119,
    -0.6080820581231067,
    -0.6322952657050278,
    -0.6569810037700855,
    -0.6821653643564161,
    -0.7080992643878224,
    -0.7345407268400616,
    -0.7616032398990319,
    -0.7891484112262872,
    -0.8174347404730798,
    -0.8462765502782976,
    -0.8757179890110683,
    -0.9057449925645107,
    -0.9363523906230068,
    -0.9676026800053741,
    -0.999353929707663,
    -1.0319392706616715,
    -1.065072736649311,
    -1.0987843155562373,
    -1.1332504315466694,
    -1.1683571477494636,
    -1.2040950661590593,
    -1.2405320320890945,
    -1.2775410523218445,
    -1.3151289097897405,
    -1.3535752990357366,
    -1.3924607146538421,
    -1.4320318500711302,
    -1.4721579173601322,
    -1.513047214338636,
    -1.5543636851998994,
    -1.5963515879727466,
    -1.638867607484833,
    -1.6820099837279432,
    -1.7257508621213398,
    -1.7701689222828143,
    -1.8151773368517836,
    -1.8605216838591698,
    -1.9065008175302796,
    -1.9529774322601958,
    -2.000027182283047,
    -2.047491941099417,
    -2.095455867140317,
    -2.143906236813457,
    -2.1928203300107114,
    -2.2421458562702234,
    -2.292074195898656,
    -2.342192920563081,
    -2.3928381506126573,
    -2.4440266518703746,
    -2.4955311926304486,
    -2.5473280496880375,
    -2.5994912583567196,
    -2.65210539142992,
    -2.7048944550958356,
    -2.758010704472694,
    -2.811568606168893,
    -2.8653417433158364,
    -2.9193762034838637,
    -2.9737089287029437,
    -3.0283578625909477,
    -3.0831655830292917,
    -3.13820870876447,
    -3.1935548208682065,
    -3.2490182189449057,
    -3.304705828840654,
    -3.360510521562962,
    -3.416588848841409,
    -3.472785746301044,
    -3.529238981234473,
    -3.585794113606018,
    -3.6425114498989912,
    -3.699344389382933,
    -3.7564809243802952,
    -3.8136703480405845,
    -3.8709937103190097,
    -3.928386073272775,
    -3.985968217025902,
    -4.043509878122987,
    -4.101220110844442,
    -4.158917988286546,
    -4.216774110870925,
    -4.274774051007943,
    -4.332786394896827,
    -4.390953213270866,
    -4.449084951284823,
    -4.507353359578804,
    -4.565745048855051,
    -4.624149139008741,
    -4.682591535482596,
    -4.741176475747337,
    -4.799793613005327,
    -4.858332657533302,
    -4.916898233929947,
    -4.9754975471043,
    -5.034352789799858,
    -5.093080959892236,
    -5.151962985575111,
    -5.2109773570026645,
    -5.269770621680132,
    -5.328565416590015,
    -5.387379531050911,
    -5.446504297394075,
    -5.505362122605654,
    -5.564459229764006,
    -5.623364636487897,
    -5.682399221431379,
    -5.741522729943125,
    -5.800685191174709,
    -5.859905040041995,
    -5.919083590023497,
    -5.978190544235085,
    -6.037322609429542,
    -6.0964984464166205,
    -6.155766065859128,
    -6.2150563464499555,
    -6.274378319298703,
    -6.333623871121277,
    -6.392870455199059,
    -6.45209792144832,
    -6.511422870707367,
    -6.570678751850576,
    -6.629933394436849,
    -6.689147246536384,
    -6.748407732464127,
    -6.807694892165269,
    -6.8670180841696515,
    -6.926249972211575,
    -6.985380416582291,
    -7.044555549125633,
    -7.103862941932737,
    -7.163194879559202,
    -7.222590138167643,
    -7.281960559840343,
    -7.341452370424834,
    -7.400860257797759,
    -7.46022305048353,
    -7.51955029643416,
    -7.5789394475684695,
    -7.638448910254032,
    -7.697892963450417,
    -7.757193308926695,
    -7.816613452313389,
    -7.8760555920115225,
    -7.935431696134529,
    -7.994858815813682,
    -8.054268467380266,
    -8.113641000073185,
    -8.173034894493783,
    -8.232479342664117,
    -8.291739871474869,
    -8.35118747619128,
    -8.410685354543059,
    -8.470077174258693,
    -8.529480041905705,
    -8.588913415041343,
    -8.648435818305732,
    -8.707861640657471,
    -8.767268943383042,
    -8.826804152222628,
    -8.886174245353738,
    -8.945418230995616,
    -9.004760668149103,
    -9.064201512288761,
    -9.12367236193421,
    -9.183173176527616,
    -9.242479308437462,
    -9.30190321940481,
    -9.36141557957274,
    -9.420733154356562,
    -9.480178177353787,
    -9.539574839778476,
    -9.599040301803049,
    -9.658535476010933,
    -9.717835729243513,
    -9.77727306896599,
    -9.836554506035363,
    -9.89600228520172,
    -9.955450372888718,
    -10.014762032543214,
    -10.074142324902624,
    -10.133454515767381,
    -10.192913433750947,
    -10.252294456049661,
    -10.311714756652337,
    -10.371291510748783,
    -10.430702441465328,
    -10.4901916778195,
    -10.549700615860365,
    -10.608965573921619,
    -10.668318573370414,
    -10.727788902611717,
    -10.787308194328434,
    -10.846710425240126,
    -10.906073712798436,
    -10.96550547193572,
    -11.02496663380363,
    -11.08432047359238,
    -11.143772063501414,
    -11.203155382637854,
    -11.26274386329476,
    -11.322146875927634,
    -11.381667150125807,
    -11.441089838001519,
    -11.50059071665935,
    -11.559984235508464,
    -11.61922156273624,
    -11.678595663585954,
    -11.738077238142496,
    -11.797519799128835,
    -11.856972171815112,
    -11.91632693167432,
    -11.975740326134934,
    -12.035183056004728,
    -12.09463558782837,
    -12.154049091389803,
    -12.2135309865673,
    -12.27284689653506,
    -12.332240960084077,
    -12.391742472477784,
    -12.451253775955621,
    -12.510726040925762,
    -12.570139734737667,
    -12.629553449816585,
    -12.689152731810784,
    -12.748605548371527,
    -12.808146273360334,
    -12.867647952648692,
    -12.92707152275547,
    -12.986426748372915,
    -13.045830816126582,
    -13.10526419396958,
    -13.164746412395752,
    -13.224170048820605,
    -13.283603461930102,
    -13.34299782296285,
    -13.402362896936147,
    -13.461767042654554,
    -13.521180962715777,
    -13.580702312868334,
    -13.64020413952687,
    -13.699686442240884,
    -13.759032033086173,
    -13.818485052275896,
    -13.877879483823003,
    -13.937273921119472,
    -13.996619535708765,
    -14.055994452274392,
    -14.115486561149723,
    -14.174910315191179,
    -14.234304776631705,
    -14.293630882722793,
    -14.353025351978072]
    ###infer the number of iterations, beginning the count at the 0th iteration###
    num_its=len(cfvals)
    n=np.linspace(0, num_its-1, num_its)

    ###Since the Hellinger distance is used as a cost function, the 'ideal' cost function value is 0
    FIGURE2(n, cfvals, exactval=0, vqaname='QABoM', ifsave=ifsave)
    return

#RUN_FIGURE2_QABoM(ifsave=True)

