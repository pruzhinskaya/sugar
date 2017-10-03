import matplotlib.pyplot as P
import numpy as N
import cPickle
import scipy.interpolate as inter
from ToolBox import MPL
from ToolBox import Astro
from ToolBox import Statistics 
import copy
from matplotlib.patches import Ellipse
from ToolBox.Signal import loess
from ToolBox.Plots import scatterPlot as SP
import matplotlib.gridspec as gridspec
from matplotlib.gridspec import GridSpec
from ToolBox import MPL
from matplotlib.widgets import Slider, Button, RadioButtons
import sys,os,optparse     
from scipy.stats import norm as NORMAL_LAW
import sugar



def Compare_TO_SUGAR_parameter(emfa_pkl='../sugar/data_output/sugar_paper_output/emfa_3_sigma_clipping.pkl',
                               SED_max='../sugar/data_output/sugar_paper_output/model_at_max_3_eigenvector_without_grey_with_sigma_clipping_save_before_PCA.pkl',
                               SUGAR_parameter_pkl='../sugar/data_output/sugar_parameters.pkl'):

    dic = cPickle.load(open(emfa_pkl))

    FILTRE=dic['filter']
    data=sugar.passage(dic['Norm_data'][FILTRE],dic['Norm_err'][FILTRE],dic['vec'],sub_space=3)
    error=sugar.passage_error(dic['Norm_err'][FILTRE],dic['vec'],3,return_std=True)


    dic_SUGAR=cPickle.load(open(SUGAR_parameter_pkl))
    dic_sed_max=cPickle.load(open(SED_max))
    Av_max=dic_sed_max['Av_cardelli']

    data_SUGAR=N.zeros(N.shape(data))
    error_SUGAR=N.zeros(N.shape(error))
    Av_SUGAR=N.zeros(len(Av_max))
    sn_name=N.array(dic['sn_name'])[dic['filter']]


    for i in range(len(sn_name)):
        data_SUGAR[i,0]=dic_SUGAR[sn_name[i]]['q1']
        data_SUGAR[i,1]=dic_SUGAR[sn_name[i]]['q2']
        data_SUGAR[i,2]=dic_SUGAR[sn_name[i]]['q3']
        Av_SUGAR[i]=dic_SUGAR[sn_name[i]]['Av']
        error_SUGAR[i]=N.sqrt(N.diag(dic_SUGAR[sn_name[i]]['cov_q'][2:,2:]))

    P.figure(figsize=(15,8))
    P.subplots_adjust(left=0.05, bottom=0.10, right=0.99, top=0.98,wspace=0.2,hspace=0.4)
    for i in range(3):
        RHO=N.corrcoef(data[:,i],data_SUGAR[:,i])[0,1]
        P.subplot(2,2,i+1)
        P.scatter(data[:,i],data_SUGAR[:,i],label=r'$\rho=%.2f$'%(RHO),zorder=10)
        P.errorbar(data[:,i],data_SUGAR[:,i], linestyle='', xerr=error[:,i],yerr=error_SUGAR[:,i],ecolor='grey',alpha=0.7,marker='.',zorder=0)
        P.legend(loc=2)
        P.xlabel('$q_{%i}$ from spectral indicators @ max'%(i+1),fontsize=20)
        P.ylabel('$q_{%i}$ from SUGAR fitting'%(i+1),fontsize=20)
        xlim = P.xlim()
        P.plot(xlim,xlim,'r',linewidth=3,zorder=5)
        P.ylim(xlim[0],xlim[1])
        P.xlim(xlim[0],xlim[1])


    RHO=N.corrcoef(Av_max,Av_SUGAR)[0,1]
    P.subplot(2,2,4)
    P.scatter(Av_max,Av_SUGAR,label=r'$\rho=%.2f$'%(RHO),zorder=10)
    P.plot(Av_max,Av_max,'r',linewidth=3)
    P.legend(loc=2)
    P.xlabel('$A_{V}$ from fit SED @ max',fontsize=20)
    P.ylabel('$A_{V}$ from SUGAR fitting',fontsize=20)
    xlim = P.xlim()
    P.plot(xlim,xlim,'r',linewidth=3,zorder=5)
    P.ylim(xlim[0],xlim[1])
    P.xlim(xlim[0],xlim[1])

class SUGAR_plot:

    def __init__(self,dico_hubble_fit):
                
        dico = cPickle.load(open(dico_hubble_fit))
        #self.key=dico['key']
        self.alpha=dico['alpha']
        self.M0=dico['m0']
        self.X=dico['X']
        self.data = dico['h']


    def plot_spectrophtometric_effec_time(self,comp=0):

        reorder = N.arange(190*19).reshape(190, 19).T.reshape(-1)
        X=self.X[reorder]
        M0=self.M0[reorder]
        ALPHA=self.alpha[:,comp][reorder]
     
        CST=N.mean(M0)
        fig,ax1=P.subplots(figsize=(7,8))
        P.subplots_adjust(left=0.1, right=0.85,bottom=0.1,top=0.99)
        Time=N.linspace(-12,42,19)
        Y2_label=[]
        Y2_pos=[]
        for i in range(19):
            
            if i%2==0:
                if (-12+(3*i))!=0:
                    Y2_label.append('%i days'%(Time[i]))
                else:
                    Y2_label.append('%i day'%(Time[i]))

                Y2_pos.append(M0[i*190:(i+1)*190][-1]-CST)

                ax1.plot(X[i*190:(i+1)*190],M0[i*190:(i+1)*190]-CST,'b')

                
                y_moins=M0[i*190:(i+1)*190]-ALPHA[i*190:(i+1)*190]*(N.mean(self.data[:,comp])+N.sqrt(N.var(self.data[:,comp])))
                y_plus=M0[i*190:(i+1)*190]+ALPHA[i*190:(i+1)*190]*(N.mean(self.data[:,comp])+N.sqrt(N.var(self.data[:,comp])))

                ax1.fill_between(X[i*190:(i+1)*190],M0[i*190:(i+1)*190]-CST,y_plus-CST,color='m',alpha=0.7 )
                ax1.fill_between(X[i*190:(i+1)*190],M0[i*190:(i+1)*190]-CST,y_moins-CST,color='g',alpha=0.7)

                if i==0:
                    CST-=2.2
                else:
                    CST-=1


        ax1.set_ylabel('$M_0(t,\lambda)$ + cst',fontsize=20)
        ax1.set_xlim(3300,8620)
        ax1.set_ylim(-1,14.8)
        ax1.set_xlabel('wavelength $[\AA]$',fontsize=20)
        p1 = P.Rectangle((0, 0), 1, 1, fc="magenta")
        p2 = P.Rectangle((0, 0), 1, 1, fc="green")
        ax1.legend([p1, p2], ['+1$\sigma_{q_%i}$'%(comp+1), '-1$\sigma_{q_%i}$'%(comp+1)],loc=4)
        P.gca().invert_yaxis()

        ax2=ax1.twinx()
        ax2.set_ylim(-1,14.8)
        ax2.yaxis.set_ticks(Y2_pos)
        ax2.yaxis.set_ticklabels(Y2_label,fontsize=15)
        ax2.set_ylim(ax2.get_ylim()[::-1])



class residual_plot:

    def __init__(self,DIC_spectra,SUGAR_model,Dic_sugar_parameter,dic_at_max,dic_salt=None):

        self.dic=cPickle.load(open(DIC_spectra))
        self.sn_name=self.dic.keys()
        self.dicS=cPickle.load(open(Dic_sugar_parameter))
        if dic_salt is not None :
            self.dicSA=cPickle.load(open(dic_salt))
        else:
            self.dicSA=None
        dic_at_max=cPickle.load(open(dic_at_max))
	self.Rv=dic_at_max['RV']
	SUGAR=N.loadtxt(SUGAR_model)
	self.M0=SUGAR[:,2]
	self.alpha=SUGAR[:,3:6]


    def plot_spectra_reconstruct(self,sn,T_min=-12,T_max=42):
        
        fig,ax1=P.subplots(figsize=(10,8))
        P.subplots_adjust(left=0.08, right=0.88,bottom=0.07,top=0.95)
        IND=None
        for i,SN in enumerate(self.sn_name):
            if SN==sn:
                IND=i

        OFFSET=[]
        Off=0
        Phase=[]
        Time=N.linspace(-12,42,19)
        DAYS=[-999]
        Y2_pos=[]
        Y2_label=[]
        for j in range(len(self.dic[sn].keys())):
            days=self.dic[sn]['%i'%(j)]['phase_salt2']

            if days>T_min and days<T_max:
                DAYS.append(days)
                if '%10.1f'%DAYS[-2]!='%10.1f'%DAYS[-1]:
                    OFFSET.append(Off+0.2-N.mean(self.dic[sn]['0']['Y']))
                    if Off==0:
                        ax1.plot(self.dic[sn]['%i'%(j)]['X'],self.dic[sn]['%i'%(j)]['Y']+OFFSET[Off],'k',
                                 label='Observed spectra')
                        if self.dicSA:
                            YS=Astro.Coords.flbda2ABmag(self.dicSA[sn]['%i'%(j)]['X'],self.dicSA[sn]['%i'%(j)]['PF_flux'])
                            YS+=-5.*N.log10(sugar.cosmology.luminosity_distance(self.dicSA[sn]['%i'%(j)]['z_helio'],self.dicSA[sn]['%i'%(j)]['z_cmb']))+5.
                            ax1.plot(self.dic[sn]['%i'%(j)]['X'],YS+OFFSET[Off],'r',lw=2,label='SALT2.4')
                    else:
                        ax1.plot(self.dic[sn]['%i'%(j)]['X'],self.dic[sn]['%i'%(j)]['Y']+OFFSET[Off],'k')
                        if self.dicSA:
                            YS=Astro.Coords.flbda2ABmag(self.dicSA[sn]['%i'%(j)]['X'],self.dicSA[sn]['%i'%(j)]['PF_flux'])
                            YS+=-5.*N.log10(sugar.cosmology.luminosity_distance(self.dicSA[sn]['%i'%(j)]['z_helio'],self.dicSA[sn]['%i'%(j)]['z_cmb']))+5.
                            ax1.plot(self.dicSA[sn]['%i'%(j)]['X'],YS+OFFSET[Off],'r',lw=2)

                    moins=self.dic[sn]['%i'%(j)]['Y']+OFFSET[Off]-N.sqrt(self.dic[sn]['%i'%(j)]['V'])
                    plus=self.dic[sn]['%i'%(j)]['Y']+OFFSET[Off]+N.sqrt(self.dic[sn]['%i'%(j)]['V'])
                    ax1.fill_between(self.dic[sn]['%i'%(j)]['X'],moins,plus,color='k',alpha=0.5 )
                    Y2_pos.append(self.dic[sn]['%i'%(j)]['Y'][-1]+OFFSET[Off])
                    if abs(self.dic[sn]['%i'%(j)]['phase_salt2'])<2:
                        Y2_label.append('%.1f day'%(self.dic[sn]['%i'%(j)]['phase_salt2']))
                    else:
                        Y2_label.append('%.1f days'%(self.dic[sn]['%i'%(j)]['phase_salt2']))
                    
                    Phase.append(self.dic[sn]['%i'%(j)]['phase_salt2'])
                    Off+=1


        Phase=N.array(Phase)
    

        Reconstruction=N.zeros((len(Phase),190))
        for Bin in range(190):
            SPLINE_Mean=inter.InterpolatedUnivariateSpline(Time,self.M0[Bin*19:(Bin+1)*19])
            Reconstruction[:,Bin]+=SPLINE_Mean(Phase)
            for i in range(3):
                SPLINE=inter.InterpolatedUnivariateSpline(Time,self.alpha[:,i][Bin*19:(Bin+1)*19])
            
                Reconstruction[:,Bin]+=self.dicS[sn]['q%i'%(i+1)]*SPLINE(Phase)

            Reconstruction[:,Bin]+=self.dicS[sn]['grey']
            Reconstruction[:,Bin]+=self.dicS[sn]['Av']*sugar.extinctionLaw(self.dic[sn]['0']['X'][Bin],
                                                                           Rv=self.Rv)

        MIN=10**25
        MAX=-10**25

        for j in range(len(Reconstruction[:,0])):
            if j == 0:
                ax1.plot(self.dic[sn]['%i'%(j)]['X'],Reconstruction[j]+OFFSET[j],'b',label='SUGAR model',lw=2)
            else:
                ax1.plot(self.dic[sn]['%i'%(j)]['X'],Reconstruction[j]+OFFSET[j],'b',lw=2)

            if N.min(Reconstruction[j]+OFFSET[j])<MIN:
                MIN=N.min(Reconstruction[j]+OFFSET[j])
                
            if N.max(Reconstruction[j]+OFFSET[j])>MAX:
                MAX=N.max(Reconstruction[j]+OFFSET[j])

        P.title(sn)
        ax1.set_xlim(3300,8600)
        ax1.set_ylim(MIN-0.5,MAX+3)
        ax1.set_ylabel(r'Mag AB $(t,\lambda)$ +cst',fontsize=12)
        ax1.set_xlabel('wavelength $[\AA]$',fontsize=12)
        ax1.legend(loc=4)
        P.gca().invert_yaxis()
        
        ax2=ax1.twinx()
        ax2.set_ylim(MIN-0.5,MAX+3)
        ax2.yaxis.set_ticks(Y2_pos)
        ax2.yaxis.set_ticklabels(Y2_label)
        ax2.set_ylim(ax2.get_ylim()[::-1])



    def plot_spectra_reconstruct_residuals(self,sn,T_min=-12,T_max=42):

        fig,ax1=P.subplots(figsize=(10,8))
        P.subplots_adjust(left=0.08, right=0.88,bottom=0.07,top=0.95)

        RESIDUAL_SUGAR=[]
        RESIDUAL_SALT2=[]
        ERROR_SPECTRA=[]

        IND=None
        for i,SN in enumerate(self.sn_name):
            if SN==sn:
                IND=i

        OFFSET=[]
        Off=0
        Phase=[]
        Time=N.linspace(-12,42,19)
        DAYS=[-999]
        JJ=[]
        Y2_pos=[]
        Y2_label=[]
        for j in range(len(self.dic[sn].keys())):
            days=self.dic[sn]['%i'%(j)]['phase_salt2']

            if days>T_min and days<T_max:
                DAYS.append(days)
                if '%10.1f'%DAYS[-2]!='%10.1f'%DAYS[-1]:
                    JJ.append(j)
                    OFFSET.append(Off+0.2)
                    if Off==0:
                        ax1.plot(self.dic[sn]['%i'%(j)]['X'],N.zeros(len(self.dic[sn]['%i'%(j)]['X']))+OFFSET[Off],
                                 'k',label='Observed spectra')
                        if self.dicSA:
                            YS=Astro.Coords.flbda2ABmag(self.dicSA[sn]['%i'%(j)]['X'],self.dicSA[sn]['%i'%(j)]['PF_flux'])
                            YS+=-5.*N.log10(sugar.cosmology.luminosity_distance(self.dicSA[sn]['%i'%(j)]['z_helio'],self.dicSA[sn]['%i'%(j)]['z_cmb']))+5.
                            if not N.isfinite(N.sum(YS)):
                                SPLINE=inter.InterpolatedUnivariateSpline(self.dicSA[sn]['%i'%(j)]['X'][N.isfinite(YS)],
                                                                          YS[N.isfinite(YS)])
                                YS[~N.isfinite(YS)]=SPLINE(self.dicSA[sn]['%i'%(j)]['X'][~N.isfinite(YS)])
                            RESIDUAL_SALT2.append(self.dic[sn]['%i'%(j)]['Y']-YS)
                            ax1.plot(self.dic[sn]['%i'%(j)]['X'],RESIDUAL_SALT2[Off]+OFFSET[Off],'r',lw=2,label='SALT2.4')
                    else:
                        ax1.plot(self.dic[sn]['%i'%(j)]['X'],N.zeros(len(self.dic[sn]['%i'%(j)]['X']))+OFFSET[Off],'k')
                        if self.dicSA:


                            
                            YS=Astro.Coords.flbda2ABmag(self.dicSA[sn]['%i'%(j)]['X'],self.dicSA[sn]['%i'%(j)]['PF_flux'])
                            YS+=-5.*N.log10(sugar.cosmology.luminosity_distance(self.dicSA[sn]['%i'%(j)]['z_helio'],self.dicSA[sn]['%i'%(j)]['z_cmb']))+5.
                            if not N.isfinite(N.sum(YS)):
                                SPLINE=inter.InterpolatedUnivariateSpline(self.dicSA[sn]['%i'%(j)]['X'][N.isfinite(YS)],
                                                                          YS[N.isfinite(YS)])
                                YS[~N.isfinite(YS)]=SPLINE(self.dicSA[sn]['%i'%(j)]['X'][~N.isfinite(YS)])
                            
                            RESIDUAL_SALT2.append(self.dic[sn]['%i'%(j)]['Y']-YS)
                            ax1.plot(self.dic[sn]['%i'%(j)]['X'],RESIDUAL_SALT2[Off]+OFFSET[Off],'r',lw=2)
                            
                    moins=OFFSET[Off]-N.sqrt(self.dic[sn]['%i'%(j)]['V'])
                    plus=OFFSET[Off]+N.sqrt(self.dic[sn]['%i'%(j)]['V'])
                    ax1.fill_between(self.dic[sn]['%i'%(j)]['X'],moins,plus,color='k',alpha=0.5 )
                    Y2_pos.append(OFFSET[Off])
                    if abs(self.dic[sn]['%i'%(j)]['phase_salt2'])<2:
                        Y2_label.append('%.1f day'%(self.dic[sn]['%i'%(j)]['phase_salt2']))
                    else:
                        Y2_label.append('%.1f days'%(self.dic[sn]['%i'%(j)]['phase_salt2']))
                    Phase.append(self.dic[sn]['%i'%(j)]['phase_salt2'])
                    Off+=1
                    

                    ERROR_SPECTRA.append(N.sqrt(self.dic[sn]['%i'%(j)]['V']))

        Phase=N.array(Phase)

        Reconstruction=N.zeros((len(Phase),190))
        for Bin in range(190):
            SPLINE_Mean=inter.InterpolatedUnivariateSpline(Time,self.M0[Bin*19:(Bin+1)*19])
            Reconstruction[:,Bin]+=SPLINE_Mean(Phase)
            for i in range(3):
                SPLINE=inter.InterpolatedUnivariateSpline(Time,self.alpha[:,i][Bin*19:(Bin+1)*19])
                Reconstruction[:,Bin]+=self.dicS[sn]['q%i'%(i+1)]*SPLINE(Phase)

            Reconstruction[:,Bin]+=self.dicS[sn]['grey']
            Reconstruction[:,Bin]+=self.dicS[sn]['Av']*sugar.extinctionLaw(self.dic[sn]['0']['X'][Bin],
                                                                           Rv=self.Rv)
        MIN=10**25
        MAX=-10**25
                
        for j in range(len(Reconstruction[:,0])):
            if j == 0:
                ax1.plot(self.dic[sn]['%i'%(j)]['X'],
                         Reconstruction[j]+OFFSET[j]-self.dic[sn]['%i'%(JJ[j])]['Y'],'b',label='SUGAR model',lw=2)
            else:
                ax1.plot(self.dic[sn]['%i'%(j)]['X'],
                         Reconstruction[j]+OFFSET[j]-self.dic[sn]['%i'%(JJ[j])]['Y'],'b',lw=2)

            RESIDUAL_SUGAR.append(Reconstruction[j]-self.dic[sn]['%i'%(JJ[j])]['Y'])
                            
        
        MIN=N.min(OFFSET[0])
        MAX=N.max(OFFSET[-1])

        P.title(sn)
        ax1.set_xlim(3300,8600)
        ax1.set_ylim(MIN-2,MAX+3)
        ax1.set_ylabel(r'Mag AB $(t,\lambda)$ + Cst.',fontsize=12)
        ax1.set_xlabel('wavelength $[\AA]$',fontsize=12)
        ax1.legend(loc=4)
        P.gca().invert_yaxis()

        ax2=ax1.twinx()
        ax2.set_ylim(MIN-2,MAX+3)
        ax2.yaxis.set_ticks(Y2_pos)
        ax2.yaxis.set_ticklabels(Y2_label)
        ax2.set_ylim(ax2.get_ylim()[::-1])

        return RESIDUAL_SUGAR, RESIDUAL_SALT2, ERROR_SPECTRA


if __name__=='__main__':

    #SED=SUGAR_plot('../sugar/data_output/sugar_model.pkl')
    #SED.plot_spectrophtometric_effec_time(comp=0)
    #SED.plot_spectrophtometric_effec_time(comp=1)
    #SED.plot_spectrophtometric_effec_time(comp=2)


    #Compare_TO_SUGAR_parameter()

    dic = cPickle.load(open('../sugar/data_output/sugar_parameters.pkl'))
    
    rp = residual_plot('../sugar/data_input/spectra_snia.pkl',
                       '../sugar/data_output/SUGAR_model_v1.asci',
                       '../sugar/data_output/sugar_parameters.pkl',
                       '../sugar/data_output/sugar_paper_output/model_at_max_3_eigenvector_without_grey_with_sigma_clipping_save_before_PCA.pkl',
                       dic_salt = '../sugar/data_input/file_pf_bis.pkl')

    residual_sel = []
    residual_sucre = []
    nspectra = 0

    i=0
    for sn in dic.keys():
        print sn 
        #rp.plot_spectra_reconstruct(sn)
        #P.savefig('plot_paper/reconstruct/'+sn+'.pdf')
        sucre, sel, residual_error = rp.plot_spectra_reconstruct_residuals(sn,T_min=-5,T_max=30)
        residual_sel.append(sel)
        residual_sucre.append(sucre)
        nspectra += len(N.array(residual_sucre[i])[:,0])
        i+=1
        #P.savefig('plot_paper/reconstruct/'+sn+'_residual.pdf')
    P.close('all')

    res_sel = N.zeros((nspectra,190))
    res_sucre = N.zeros((nspectra,190))

    t = 0
    for i in range(len(dic.keys())):
        for j in range(len(residual_sel[i])):
            res_sucre[t] = residual_sucre[i][j]
            res_sel[t] = residual_sel[i][j]
            t+=1

    dic = cPickle.load(open('../sugar/data_input/spectra_snia.pkl'))
    wave = dic['PTF09dnl']['0']['X']
            
    P.plot(wave,N.std(res_sel,axis=0),'r',linewidth=3,label='SALT2.4')
    P.plot(wave,N.std(res_sucre,axis=0),'b',linewidth=3,label='SUGAR')            
    P.xlabel('wavelength $[\AA]$',fontsize=20)
    P.ylabel('STD (mag)',fontsize=20)
    P.ylim(0,0.5)
    P.xlim(3300,8700)
    P.legend(loc=3)
    P.show()
    
                                    
