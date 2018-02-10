"""plot emfa result on spectral feaAtures at max."""

from ToolBox import Statistics
from ToolBox import MPL
from ToolBox.Signal import loess
import numpy as np
import pylab as plt
import cPickle
import sugar
import copy
import os

path = os.path.dirname(sugar.__file__)

def SP_set_kwargs(inkw, label, **kwargs):

    """
    Read dictionnary *label* from *inkw*, and set some default
    values from *kwargs*.
    """
    outkw = inkw.get(label, {})
    for key in kwargs:
        outkw.setdefault(key, kwargs[key])

    return outkw


class emfa_plot:
    """
    this class will make all the plot for the emfa part.

    before to try to make this plot you should run the
    emfa_analysis.py in order to have the emfa_output.pkl
    in your data_output within the sugar forders
    """
    def __init__(self):

        dic_emfa = cPickle.load(open(path+'/data_output/emfa_output.pkl'))
        self.val = dic_emfa['val']
        self.vec = dic_emfa['vec']
        self.si_norm = dic_emfa['Norm_data'][dic_emfa['filter']]
        self.si_norm_err = dic_emfa['Norm_err'][dic_emfa['filter']]
        self.sn_name = dic_emfa['sn_name']
        self.filtre = dic_emfa['filter']
        self.si = dic_emfa['data'][dic_emfa['filter']]
        self.si_err = dic_emfa['error'][dic_emfa['filter']]


    def no_linear(self):

        from mpl_toolkits.mplot3d import Axes3D
                
        sil = ['pEWCa II H&K', r'pEWSi II $\lambda$4131', 'pEWMg II',
               'pEWFe $\lambda$4800', 'pEWS II W', 'pEWSi II $\lambda$5972',
               'pEWSi II $\lambda$6355', 'pEWO I $\lambda$7773', 'pEWCa II IR',
               'VSi II $\lambda$4131','VWS II $\lambda$5454','VWS II $\lambda$5640',
               'VSi II $\lambda$6355']

        data = self.si_norm
        err = self.si_norm_err

        new_base = sugar.passage(data,err,self.vec,sub_space=3)
                                 

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        plt.subplots_adjust(bottom=0.01,top=0.99,left=0.01,right=1.01)
        scat = ax.scatter(self.si[:,1],self.si[:,5],self.si[:,6],s=50,c=new_base[:,0],vmin=-3.5,vmax=3.5,cmap=plt.cm.jet)
        ax.set_xlabel(sil[1] + ' ($\AA$)',fontsize=15)
        ax.set_xlim(0,40)
        ax.set_ylabel(sil[5] + ' ($\AA$)',fontsize=15)
        ax.set_ylim(0,50)
        ax.set_zlabel(sil[6] + ' ($\AA$)',fontsize=15)
        ax.set_zlim(35,140)
        cb = fig.colorbar(scat)
        cb.set_label('SUGAR $q_1$',fontsize=20)

    def plot_eigenvalues(self,noise=False):
        """
        plot eigenvalue in terms of eigenvectors.

        if noise set to true, it will plot the noise
        level from emfa (variability in psi matrix).
        """
        plt.figure(figsize=(6,6))
        plt.subplots_adjust(top=0.97,bottom=0.1,left=0.1,right=0.99,hspace=0.0)

        x_axis = np.linspace(1,len(self.val),len(self.val))

        val_100 = (self.val/(np.sum(self.val)))*100.

        xlim = plt.xlim(0,len(self.val)+1)
        plt.plot([xlim[0],xlim[1]],[0,0],'k',linewidth=2)
        
        plt.plot(x_axis,val_100,'b-',linewidth=3,label='eigenvalue')
        plt.scatter(x_axis,val_100,c='b',s=75)

        if noise:
            val_100 = (self.val/(len(self.val)))*100.
            noise_level = (len(self.val) - np.sum(self.val)) / len(self.val) * 100.
            plt.plot([xlim[0],xlim[1]],noise_level*np.ones(2),'r',linewidth=3,label='noise level')
        
        plt.xticks(x_axis)
        plt.yticks(np.linspace(0,40,9))
        plt.ylim(-1,42)
        plt.ylabel('variability on each component (%)',fontsize=20)
        plt.xlabel('Factor analysis component',fontsize=20)
        plt.legend()


    def plot_pf_corr_factor_si(self,split=5):
        """
        plot corr coeff between fa component and si.

        split: int, number of component that you want to draw. 
        """
        data = self.si_norm
        err = self.si_norm_err

        new_base = sugar.passage(data,err,self.vec,sub_space=10)
        new_err = sugar.passage_error(err,self.vec,sub_space=10,return_std=True)
        
        new_base = new_base[:,:split]
        new_err = new_err[:,:split]

        #new_base[:,0] *= -1
        #new_base[:,1] *= -1

        nsil = ['pEWCa II H&K', r'pEWSi II $\lambda$4131', 'pEWMg II',
                'pEWFe $\lambda$4800', 'pEWS II W', 'pEWSi II $\lambda$5972',
                'pEWSi II $\lambda$6355', 'pEWO I $\lambda$7773', 'pEWCa II IR',
                'VSi II $\lambda$4131','VWS II $\lambda$5454','VWS II $\lambda$5640',
                'VSi II $\lambda$6355']

        dic_corr_vec = {}
        dic_corr_vece = {}
        neff = []
        X = []
        Y = []
    
        for i in range(len(new_base[0])):
            dic_corr_vec.update({'corr_vec%i'%(i):np.zeros(len(nsil))})
            dic_corr_vece.update({'corr_vec%ie'%(i):np.zeros(len(nsil))})
            neff.append([])
            X.append([])
            Y.append([])
    
        for j in range(len(new_base[0])):
            for i in range(len(nsil)):
                dic_corr_vec['corr_vec%i'%(j)][i],dic_corr_vece['corr_vec%ie'%(j)][i]=Statistics.correlation_weighted(data[:,i],new_base[:,j], w=1./(err[:,i]*new_err[:,j]),error=True, symmetric=True)

                neff[j].append(Statistics.neff_weighted(1./(err[:,i]*new_err[:,j])))
            
                X[j].append(data[:,i])
                
                Y[j].append(new_base[:,j])


                
        cmap = plt.matplotlib.cm.get_cmap('Blues',9)
    
    

        fig = plt.figure(figsize=(12,6.5),dpi=100)
        ax = fig.add_axes([0.05,0.07,0.9,0.72])
        #plt.subplots_adjust(top=0.5,bottom=0.2,left=0.1,right=1.1,hspace=0.0)
        rotation=45
        xstart, xplus, ystart = 0.03, 0.0777 ,1.01
 
        cmap.set_over('r')
        bounds = [0, 1, 2, 3, 4, 5]
        norm = plt.matplotlib.colors.BoundaryNorm(bounds, cmap.N)
                
        ylabels=[]
        corrs=[]
        Ticks=[]
        for j in range(len(new_base[0])):
            corrs.append(dic_corr_vec['corr_vec%i'%(j)])
            ylabels.append(r'$q_{%i}$'%(j+1))
            Ticks.append(4-j)
        
        for i,corr in enumerate(corrs):
            sig = np.array([Statistics.correlation_significance(np.abs(c),n, sigma=True) for c, n in zip(corr,neff[i])])
            Sig = copy.deepcopy(sig)
            sig /= bounds[-1]
            cols = cmap(sig)
            mat = [[[0.25,rho*0.25],[rho*0.25,0.25]] for rho in corr]
            MPL.errorellipses(ax, range(1, len(nsil)+1), [4-i]*len(corr),
                              mat, color=cols, alpha=1, **{'ec':'k'})
            for j,c in enumerate(corr):
                x = (X[i][j]-np.min(X[i][j]))/np.max(X[i][j]-np.min(X[i][j]))-0.5
                y = (Y[i][j]-np.min(Y[i][j]))/np.max(Y[i][j]-np.min(Y[i][j]))-0.5
                x += (j+1)
                y += -np.mean(y) + (4-i)
                esty = loess(x, y)
                isort = np.argsort(x)
                lkwargs = SP_set_kwargs({}, 'loess', c='b', alpha=0.7, ls='-', lw=1)
                if Sig[j]>4 and Sig[j]<5:
                    if c<0.9:
                        ax.annotate('%.2f'%c,(j+1,4-i),color='w',ha='center',va='center',)
                    else:
                        ax.annotate('%.2f'%c,(j+1,4-i),color='w',fontsize=9,ha='center',va='center',)
                
                else:
                    ax.annotate('%.2f'%c,(j+1,4-i),ha='center',va='center',)
        x = xstart
        toto = 1
        for leg in nsil:
            ax.annotate(leg, (x,ystart), xycoords='axes fraction',
                        size='large', rotation=rotation, ha='left', va='bottom')
            if toto == 12:
                toto += 1
                x += 0.06
            else:
                toto += 1
                x += xplus    
        
        ax.set_xticks([])

        ax.set_yticks(Ticks)
        ax.set_yticklabels(ylabels, size='xx-large', rotation=90)
        ax.set_ylim(ymin=4.4-len(new_base[0]),ymax=4.6)
        ax.set_xlim(xmin=0.4, xmax=len(nsil)+0.6)
        ax.set_aspect('equal', adjustable='box-forced', anchor='C')
        
        im = ax.imshow([[0,5]], cmap=cmap,
                       extent=None, origin='upper',
                       interpolation='none', visible=False)
        cax, kw = plt.matplotlib.colorbar.make_axes(ax, orientation='horizontal',
                                                    pad=0.02)

        cb = plt.matplotlib.colorbar.ColorbarBase(cax, cmap=cmap,
                                                  norm=norm,
                                                  boundaries=bounds+[9],
                                                  extend='max',
                                                  ticks=bounds,
                                                  spacing='proportional',
                                                  orientation='horizontal')

        cb.set_label('Pearson correlation coefficient significance ($\sigma$)',
                     size='x-large')


    def plot_pf_corr_factor_salt2(self,split=5):
        """
        plot corr coeff between fa component and si.

        split: int, number of component that you want to draw. 
        """
        data = self.si_norm
        err = self.si_norm_err

        new_base = sugar.passage(data,err,self.vec,sub_space=10)
        new_err = sugar.passage_error(err,self.vec,sub_space=10,return_std=True)
        
        new_base = new_base[:,:split]
        new_err = new_err[:,:split]

        #new_base[:,0] *= -1
        #new_base[:,1] *= -1
        
        data_sugar = sugar.load_data_sugar()
        data_sugar.load_salt2_data()

        delta_mu = copy.deepcopy(data_sugar.mb)

        for i in range(len(data_sugar.zhelio)):
            delta_mu[i] -= sugar.distance_modulus(data_sugar.zhelio[i],data_sugar.zcmb[i])

        data = np.array([data_sugar.X1,data_sugar.C,delta_mu]).T[self.filtre]
        err = np.array([data_sugar.X1_err,data_sugar.C_err,data_sugar.mb_err]).T[self.filtre]

        self.data = data
        self.err = err
        self.new = new_base
        self.new_err = new_err
        nsil = [r'$X_1$', r'$C$', r'$\Delta \mu_B$']

        dic_corr_vec = {}
        dic_corr_vece = {}
        neff = []
        X = []
        Y = []
    
        for i in range(len(new_base[0])):
            dic_corr_vec.update({'corr_vec%i'%(i):np.zeros(len(nsil))})
            dic_corr_vece.update({'corr_vec%ie'%(i):np.zeros(len(nsil))})
            neff.append([])
            X.append([])
            Y.append([])
    
        for j in range(len(new_base[0])):
            for i in range(len(nsil)):
                print j,i
                dic_corr_vec['corr_vec%i'%(j)][i],dic_corr_vece['corr_vec%ie'%(j)][i]=Statistics.correlation_weighted(data[:,i],new_base[:,j] ,error=True, symmetric=True)

                neff[j].append(Statistics.neff_weighted(1./(np.ones_like(data[:,i]))))
            
                X[j].append(data[:,i])
                
                Y[j].append(new_base[:,j])


                
        cmap = plt.matplotlib.cm.get_cmap('Blues',9)
    
    

        fig = plt.figure(figsize=(5,5),dpi=100)
        ax = fig.add_axes([0.05,0.07,0.9,0.85])
        #plt.subplots_adjust(top=0.5,bottom=0.2,left=0.1,right=1.1,hspace=0.0)
        xstart, xplus, ystart = 0.1, 0.37 ,1.01
 
        cmap.set_over('r')
        bounds = [0, 1, 2, 3, 4, 5]
        norm = plt.matplotlib.colors.BoundaryNorm(bounds, cmap.N)
                
        ylabels=[]
        corrs=[]
        Ticks=[]
        for j in range(len(new_base[0])):
            corrs.append(dic_corr_vec['corr_vec%i'%(j)])
            ylabels.append(r'$q_{%i}$'%(j+1))
            Ticks.append(4-j)
        
        for i,corr in enumerate(corrs):
            sig = np.array([Statistics.correlation_significance(np.abs(c),n, sigma=True) for c, n in zip(corr,neff[i])])
            Sig = copy.deepcopy(sig)
            sig /= bounds[-1]
            cols = cmap(sig)
            mat = [[[0.25,rho*0.25],[rho*0.25,0.25]] for rho in corr]
            MPL.errorellipses(ax, range(1, len(nsil)+1), [4-i]*len(corr),
                              mat, color=cols, alpha=1, **{'ec':'k'})
            for j,c in enumerate(corr):
                x = (X[i][j]-np.min(X[i][j]))/np.max(X[i][j]-np.min(X[i][j]))-0.5
                y = (Y[i][j]-np.min(Y[i][j]))/np.max(Y[i][j]-np.min(Y[i][j]))-0.5
                x += (j+1)
                y += -np.mean(y) + (4-i)
                esty = loess(x, y)
                isort = np.argsort(x)
                lkwargs = SP_set_kwargs({}, 'loess', c='b', alpha=0.7, ls='-', lw=1)
                if Sig[j]>4 and Sig[j]<5:
                    if c<0.9:
                        ax.annotate('%.2f'%c,(j+1,4-i),color='w',ha='center',va='center',)
                    else:
                        ax.annotate('%.2f'%c,(j+1,4-i),color='w',fontsize=9,ha='center',va='center',)
                
                else:
                    ax.annotate('%.2f'%c,(j+1,4-i),ha='center',va='center',)
        x = xstart
        toto = 1
        for leg in nsil:
            ax.annotate(leg, (x,ystart), xycoords='axes fraction',
                        size='large', ha='left', va='bottom')
    
            toto += 1
            x += xplus    
        
        ax.set_xticks([])

        ax.set_yticks(Ticks)
        ax.set_yticklabels(ylabels, size='xx-large', rotation=90)
        ax.set_ylim(ymin=4.4-len(new_base[0]),ymax=4.6)
        ax.set_xlim(xmin=0.4, xmax=len(nsil)+0.6)
        ax.set_aspect('equal', adjustable='box-forced', anchor='C')
        
        im = ax.imshow([[0,5]], cmap=cmap,
                       extent=None, origin='upper',
                       interpolation='none', visible=False)
        cax, kw = plt.matplotlib.colorbar.make_axes(ax, orientation='horizontal',
                                                    pad=0.02)

        cb = plt.matplotlib.colorbar.ColorbarBase(cax, cmap=cmap,
                                                  norm=norm,
                                                  boundaries=bounds+[9],
                                                  extend='max',
                                                  ticks=bounds,
                                                  spacing='proportional',
                                                  orientation='horizontal')

        cb.set_label('Pearson correlation coefficient significance ($\sigma$)',
                     fontsize=14)        
    
        

if __name__=='__main__':

    faplot = emfa_plot()
    #faplot.no_linear()
    faplot.plot_eigenvalues(noise=True)
    faplot.plot_pf_corr_factor_si(split=5)
    faplot.plot_pf_corr_factor_salt2(split=5)
