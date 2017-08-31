"""plot output from gaussian process for sugar paper."""

import numpy as np
import pylab as plt
import sugar
import os
import cosmogp

path = os.path.dirname(sugar.__file__)

#rcParams['font.size'] = 26.
#font = {'family': 'normal', 'size': 26}
#rc('axes', linewidth=1.5)
#rc("text", usetex=True)
#rc('font', family='serif')
#rc('font', serif='Times')
#rc('legend', fontsize=25)
#rc('xtick.major', size=5, width=1.5)
#rc('ytick.major', size=5, width=1.5)
#rc('xtick.minor', size=3, width=1)
#rc('ytick.minor', size=3, width=1)


def plot_line_positon(ylim):
    """
    Draw position of SNIa line at maximum.
    """
    #Ca H&K
    plt.fill_between([3580,3880],[ylim[0],ylim[0]], [ylim[1],ylim[1]],color='k',alpha=0.1)
    #Si II 4131
    plt.fill_between([3940,4060],[ylim[0],ylim[0]], [ylim[1],ylim[1]],color='k',alpha=0.1)
    #Mg II
    plt.fill_between([4120,4420],[ylim[0],ylim[0]], [ylim[1],ylim[1]],color='k',alpha=0.1)
    #Fe 4800
    plt.fill_between([4480,5137],[ylim[0],ylim[0]], [ylim[1],ylim[1]],color='k',alpha=0.1)
    #S II W
    plt.fill_between([5197,5560],[ylim[0],ylim[0]], [ylim[1],ylim[1]],color='k',alpha=0.1)
    #Si II 5972
    plt.fill_between([5620,5902],[ylim[0],ylim[0]], [ylim[1],ylim[1]],color='k',alpha=0.1)
    #Si II 6355
    plt.fill_between([5962,6277],[ylim[0],ylim[0]], [ylim[1],ylim[1]],color='k',alpha=0.1)
    #O I
    plt.fill_between([7205,7840],[ylim[0],ylim[0]], [ylim[1],ylim[1]],color='k',alpha=0.1)
    #Ca II IR
    plt.fill_between([7880,8530],[ylim[0],ylim[0]], [ylim[1],ylim[1]],color='k',alpha=0.1)

def plot_gp_output(plot_mean=False):
    """
    plot gp output. 
    
    kernel amplitude
    correlation length
    pull mean
    pull std of residual
    """
    try: gp_output = np.loadtxt(path+'/data_output/gaussian_process/gp_info.dat',comments='#')
    except ValueError:
        "%s does not exist"%(path+'/sugar/data_output/gaussian_process/gp_info.dat')

    if plot_mean:
        add_subplot = 1
    else:
        add_subplot = 0
        
    xlim = [3300,8600]
        
    plt.figure(figsize=(12,10))
    plt.subplots_adjust(top=0.85,bottom=0.08,left=0.08,right=0.99,hspace=0.0)
    
    plt.subplot(3+add_subplot,1,1)
    
    ylim = plt.ylim(0,0.7)
    plot_line_positon(ylim)
        
    plt.plot(gp_output[:,0],gp_output[:,1],'b',linewidth=3)


    #plt.xticks([],[])
    nsil = [r'Ca II H&K', r'Si II $\lambda$4131', r'Mg II',
            r'Fe $\lambda$4800', r'S II W', r'Si II $\lambda$5972',
            r'Si II $\lambda$6355', r'O I $\lambda$7773', r'Ca II IR']
    pos = [3730.0+200,4000.0+200,4270.0+100,4808.5+200,5378.5+100,
           5761.0+200,6119.5+200,7522.5+200,8205.0+100]
    
    plt.gca().xaxis.tick_top()
    plt.xticks(pos,nsil,fontsize=20,rotation=45)
    plt.yticks(np.linspace(0.1,0.6,6))    
    plt.xlim(xlim[0],xlim[1])
    plt.ylabel('$\sigma(\lambda)$ (mag)',fontsize=20)
    
    plt.subplot(3+add_subplot,1,2)

    ylim=plt.ylim(0,13)
    plot_line_positon(ylim)
    
    plt.plot(gp_output[:,0],gp_output[:,2],'b',linewidth=3)

    plt.xticks([],[])
    plt.yticks(np.linspace(2,12,6))
    plt.xlim(xlim[0],xlim[1])
    plt.ylabel('$l(\lambda)$ (days)',fontsize=20)

    if plot_mean:
        plt.subplot(3+add_subplot,1,3)

        ylim=plt.ylim(-0.2,0.18)
        plot_line_positon(ylim)
    
        plt.plot(xlim,np.zeros_like(xlim),'k-.')
        plt.plot(gp_output[:,0],gp_output[:,3],'b',linewidth=3)
        err = gp_output[:,4]/np.sqrt(gp_output[:,5])
        plt.fill_between(gp_output[:,0],gp_output[:,3]-err,gp_output[:,3]+err,color='b',alpha=0.3)

        plt.xticks([],[])
        plt.yticks(np.linspace(-0.15,0.15,7))
        plt.xlim(xlim[0],xlim[1])
        plt.ylabel('$<\mathrm{pull}(\lambda)>$',fontsize=20)

    plt.subplot(3+add_subplot,1,3+add_subplot)

    ylim=plt.ylim(0.8,1.6)
    plot_line_positon(ylim)
    
    plt.plot(xlim,np.ones_like(xlim),'k-.')
    plt.plot(gp_output[:,0],gp_output[:,4],'b',linewidth=3)
    err = gp_output[:,4]/np.sqrt(2.*(gp_output[:,5]-1.))
    plt.fill_between(gp_output[:,0],gp_output[:,4]-err,gp_output[:,4]+err,color='b',alpha=0.3)

    plt.xlim(xlim[0],xlim[1])
    plt.yticks(np.linspace(0.9,1.5,7))
    plt.xlabel('wavelength [$\AA$]',fontsize=20)
    plt.ylabel('$\mathrm{RMS}[\mathrm{pull}(\lambda)]$',fontsize=20)


def plot_snia_interpolation(sn_name):
    """
    Plot one example of SNIa interpolation. 
    """
    try: gp_output = np.loadtxt(path+'/data_output/gaussian_process/gp_info.dat',comments='#')
    except ValueError:
        "%s does not exist"%(path+'/sugar/data_output/gaussian_process/gp_info.dat')
    
    lds = sugar.load_data_sugar()
    lds.load_spectra()

    gp_interp = np.zeros((len(lds.spectra_phases[sn_name].keys()),
                          len(lds.spectra_wavelength[sn_name]['0'])))

    ldbg = sugar.load_data_bin_gp()
    ldbg.build_difference_mean()
    diff = ldbg.diff

    ind_sn = list(ldbg.sn_name).index(sn_name)
    
    for i in range(len(gp_interp[0])):
        ldbg.load_data_bin(i)
        ldbg.load_mean_bin(i,hsiao_empca=True)
        
        gpr = cosmogp.gaussian_process_nobject(ldbg.y, ldbg.time, kernel='RBF1D',
                                               y_err=ldbg.y_err, diff=diff, Mean_Y=ldbg.mean,
                                               Time_mean=ldbg.mean_time, substract_mean=False)

        gpr.nugget = 0.03
        gpr.hyperparameters = [gp_output[i,1],gp_output[i,2]]
        gpr.get_prediction(new_binning=ldbg.time[ind_sn], COV=True, svd_method=False)
        gp_interp[:,i] = gpr.Prediction[ind_sn]
        
    plt.figure(figsize=(8,12))
    plt.subplots_adjust(top=0.98,bottom=0.08,left=0.08,right=0.85,hspace=0.0)

    CST = 19.2

    y_label_position = []
    y_label = []
    
    for i in range(len(lds.spectra_phases[sn_name].keys())):
        if i == 0:
            plt.plot(lds.spectra_wavelength[sn_name]['%i'%(i)],
                     lds.spectra[sn_name]['%i'%(i)]+CST,'r',linewidth=4,label=sn_name)
            plt.plot(lds.spectra_wavelength[sn_name]['%i'%(i)],
                     gp_interp[i]+CST,'b',linewidth=2,label='gaussian process interpolation')
        else:
            plt.plot(lds.spectra_wavelength[sn_name]['%i'%(i)],
                     lds.spectra[sn_name]['%i'%(i)]+CST,'r',linewidth=4)
            plt.plot(lds.spectra_wavelength[sn_name]['%i'%(i)],
                     gp_interp[i]+CST,'b',linewidth=2)
        y_label_position.append(gp_interp[i,-1]+CST)
        y_label.append('%.2f days'%(ldbg.time[ind_sn][i]))
            
        CST +=1

    plt.legend(loc=4)
    plt.ylim(-0.5,14.2)
    plt.gca().invert_yaxis()
    xlim = [3300,8600]
    ax1 = plt.gca()
    ax2 = plt.gca().twinx()
    ax2.set_yticks(y_label_position)
    ax2.set_yticklabels(y_label,fontsize=20)
    ax2.set_ylim(-0.5,14.2)
    ax2.invert_yaxis()
    ax1.set_xlim(xlim[0],xlim[1])
    ax1.set_xlabel('wavelength [$\AA$]',fontsize=20)
    ax1.set_ylabel('mag AB + cst.',fontsize=20)


    
if __name__=='__main__':

    plot_gp_output()
    plt.savefig('plot_paper/gaussian_processes.pdf')
    #plot_snia_interpolation('PTF09foz')
